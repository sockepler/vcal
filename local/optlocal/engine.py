"""Local constrained Bayesian-optimization engine.

Same algorithm as the Windows client (Sobol init -> GP / deep-kernel GP
+ TuRBO + batched constrained qLogNEI) but:
  * evaluation through an Evaluator (local in-process by default)
  * flexible Objective (maximize / minimize / approach-target terms)
  * pause/stop + per-record callbacks for the GUI
  * device auto-detect: CUDA and AMD ROCm both appear as torch.cuda
"""
import csv
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Mapping

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from optclient.space import Space
from optclient.turbo import TurboState

from .i18n import tr

from .objective import Objective, _finite
from .proposer import LocalProposer, RemoteProposer

DTYPE = torch.double


def pick_device(pref="auto"):
    """auto 默认 CPU：2026-07-06 microbench 实测，在典型 EDA sizing 规模
    （几十~几百个采样点、10-20 维）下，GP/DKL 的小矩阵计算 CPU 明显快于
    GPU（GPU 的 kernel launch 开销压过微小算量）——40点纯 GP：CPU ~7s vs
    AMD RX6800 ~15s。GPU 只有在点数上千 / DKL 网络更大时才可能翻盘。
    想用本地 GPU 或远端强卡显式指定 device=cuda/gpu 或 --remote-gpu。"""
    if pref not in ("auto", "cpu", "cuda", "gpu", "rocm"):
        raise ValueError("device must be auto/cpu/cuda/gpu/rocm")
    if pref in ("cuda", "gpu", "rocm"):
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
        return "cpu", tr("cpu (指定GPU但未检测到，回退CPU)")
    return "cpu", "cpu"


class Constraint:
    """max / min bound, or two-sided |m - target| <= tol."""

    def __init__(self, cfg):
        if not isinstance(cfg, dict) or not isinstance(cfg.get("metric"), str):
            raise ValueError("constraint must contain a metric name")
        if set(cfg) & {"min", "max", "target", "tol"} not in (
                {"min"}, {"max"}, {"target", "tol"}):
            raise ValueError("constraint requires min, max, or target + tol")
        self.metric = cfg["metric"]
        if "tol" in cfg and "target" in cfg:
            self.kind = "band"
            self.target = _finite(cfg["target"], "constraint target")
            self.tol = _finite(cfg["tol"], "constraint tol")
            if self.tol < 0:
                raise ValueError("constraint tol must be nonnegative")
        elif "max" in cfg:
            self.kind = "max"
            self.bound = _finite(cfg["max"], "constraint max")
        else:
            self.kind = "min"
            self.bound = _finite(cfg["min"], "constraint min")

    def value(self, metrics):
        m = float(metrics[self.metric])
        if self.kind == "band":
            return abs(m - self.target) - self.tol
        if self.kind == "max":
            return m - self.bound
        return self.bound - m

    def describe(self):
        if self.kind == "band":
            return "|%s - %g| <= %g" % (self.metric, self.target, self.tol)
        op = "<=" if self.kind == "max" else ">="
        return "%s %s %g" % (self.metric, op, self.bound)


class Engine:
    def __init__(self, evaluator, outdir, objective=None, constraints=None,
                 batch=4, n_init=24, device="auto", use_dkl=True,
                 dkl_after=48, seed=0, on_log=None, on_record=None,
                 spec_params=None, fixed=None, remote_gpu=None,
                 initial_points=None, stagnation_rounds=0):
        for name, value in (("batch", batch), ("n_init", n_init), ("dkl_after", dkl_after)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if (isinstance(stagnation_rounds, bool) or
                not isinstance(stagnation_rounds, int) or stagnation_rounds < 0):
            raise ValueError("stagnation_rounds must be a nonnegative integer")
        self.ev = evaluator
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.spec = evaluator.spec()
        # spec_params: optional subset/override of the search space
        # (GUI: enabled params with edited bounds); fixed: params pinned
        # to a constant value, merged into every evaluation.
        self.fixed = dict(self.spec.get("fixed", {}) if fixed is None else fixed)
        params = ([p for p in self.spec["params"] if p.get("enabled", True)
                   and p["name"] not in self.fixed] if spec_params is None else spec_params)
        self.space = Space(params)
        all_params = {p["name"]: p for p in self.spec["params"]}
        for name, value in self.fixed.items():
            if name not in all_params or name in self.space.names:
                raise ValueError("fixed parameter must be known and disabled: %s" % name)
            v = _finite(value, "fixed." + name)
            p = all_params[name]
            if not p["lo"] <= v <= p["hi"] or (p.get("integer") and not v.is_integer()):
                raise ValueError("fixed.%s is outside its allowed range" % name)
            self.fixed[name] = v
        self.obj = Objective(objective if objective is not None
                             else self.spec["objective"])
        cons_cfg = (constraints if constraints is not None
                    else self.spec.get("constraints", []))
        self.cons = [Constraint(c) for c in cons_cfg]
        self.cons += [Constraint(c) for c in
                      self.obj.implicit_constraints()]
        if "metrics" in self.spec:
            for metric in [t.metric for t in self.obj.terms] + [c.metric for c in self.cons]:
                if metric not in self.spec["metrics"]:
                    raise ValueError("unknown objective/constraint metric: %s" % metric)
        raw_initial_points = (self.spec.get("initial_points", [])
                              if initial_points is None else initial_points)
        self.initial_points = self._validate_initial_points(raw_initial_points,
                                                            all_params)
        self.batch = batch
        self.n_init = n_init
        self.stagnation_rounds = stagnation_rounds
        self.use_dkl = use_dkl
        self.dkl_after = dkl_after
        self.device, self.device_name = pick_device(device)
        self._log = on_log or (lambda s: print(s, flush=True))
        if remote_gpu:
            self.proposer = RemoteProposer(remote_gpu,
                                           device_fallback=self.device,
                                           log=self._log)
        else:
            self.proposer = LocalProposer(self.device)
        torch.manual_seed(seed)
        self.sobol = SobolEngine(self.space.dim, scramble=True, seed=seed)
        self.seed = seed
        self.turbo = TurboState(self.space.dim, batch)
        self.stop_event = threading.Event()
        self._log = on_log or (lambda s: print(s, flush=True))
        self._on_record = on_record or (lambda rec, st: None)
        # observations
        self.X, self.y, self.C, self.ok, self.records = [], [], [], [], []
        self.status = "ready"
        self.budget = None
        self._seen = set()
        self._resumed_ids = set()
        self.stagnation_count = 0
        self.exploration_restarts = 0
        self.exploration_pending = False

    def _validate_initial_points(self, points, all_params):
        """Validate and complete physical seed points before any evaluation."""
        if points is None:
            points = []
        if not isinstance(points, list):
            raise ValueError("initial_points must be a list")
        nominal = self.space.to_physical(self.space.nominal_unit())
        active = {p["name"]: p for p in self.space.params}
        out = []
        for index, raw in enumerate(points):
            if not isinstance(raw, Mapping):
                raise ValueError("initial_points[%d] must be a mapping" % index)
            point = dict(nominal)
            for name, value in raw.items():
                if name not in all_params:
                    raise ValueError("initial_points[%d] has unknown parameter: %s"
                                     % (index, name))
                # Fixed parameters may be repeated in a complete physical
                # point, but their value must agree with the run configuration.
                if name in self.fixed:
                    value = _finite(value, "initial_points[%d].%s" % (index, name))
                    if not math.isclose(float(value), float(self.fixed[name]),
                                        rel_tol=1e-9, abs_tol=0.0):
                        raise ValueError("initial_points[%d].%s conflicts with fixed parameter"
                                         % (index, name))
                    continue
                if name not in active:
                    raise ValueError("initial_points[%d] parameter is inactive: %s"
                                     % (index, name))
                value = _finite(value, "initial_points[%d].%s" % (index, name))
                p = active[name]
                if (not p["lo"] <= value <= p["hi"] or
                        (p.get("integer") and not value.is_integer())):
                    raise ValueError("initial_points[%d].%s is outside its allowed range"
                                     % (index, name))
                point[name] = value
            point.update(self.fixed)
            out.append(point)
        return out

    # ---------- data ----------
    def _inactive_params_compatible(self, params):
        """Keep history backgrounds outside the selected scope compatible.

        LocalEvaluator records a known nominal for an omitted inactive
        parameter, while it omits inactive parameters whose nominal is
        unknown.  Match that representation so a changed background cannot
        enter the active-parameter model.
        """
        selected = set(self.space.names) | set(self.fixed)
        for spec in self.spec.get("params", []):
            name = spec.get("name") if isinstance(spec, dict) else None
            if not name or name in selected:
                continue
            nominal = spec.get("nominal")
            if nominal is None:
                if name in params:
                    return False
                continue
            if name not in params:
                return False
            try:
                value = _finite(params[name], "param " + name)
                nominal = _finite(nominal, "nominal " + name)
            except (TypeError, ValueError, OverflowError):
                return False
            if not math.isclose(value, nominal, rel_tol=1e-9, abs_tol=0.0):
                return False
        return True

    def _ingest(self, result, history=False):
        params = result.get("params") or {}
        if history and not self._inactive_params_compatible(params):
            return False
        full = {p["name"]: params.get(p["name"], p.get("nominal"))
                for p in self.space.params}
        if any(v is None for v in full.values()):
            return False
        try:
            for p in self.space.params:
                v = _finite(full[p["name"]], "param " + p["name"])
                if not p["lo"] <= v <= p["hi"] or (p.get("integer") and not v.is_integer()):
                    return False
                full[p["name"]] = v
            defaults = {p["name"]: p.get("nominal") for p in self.spec["params"]}
            for name, value in self.fixed.items():
                if not math.isclose(float(params.get(name, defaults.get(name))), value,
                                    rel_tol=1e-9, abs_tol=0.0):
                    return False
        except (TypeError, ValueError, OverflowError):
            return False
        x = self.space.to_unit(full)
        good = bool(result.get("ok")) and result.get("metrics")
        yv = cv = None
        if good:
            try:
                m = result["metrics"]
                if not isinstance(m, dict) or not all(math.isfinite(float(v)) for v in m.values()):
                    raise ValueError("non-finite or invalid metrics")
                yv = self.obj.scalar(m)
                cv = [c.value(m) for c in self.cons]
                if not math.isfinite(yv) or not all(math.isfinite(v) for v in cv):
                    raise ValueError("non-finite objective/constraint")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                good = False
                result = {**result, "ok": False, "error": str(exc)}
                yv = cv = None
        if not good:
            yv = cv = None
        self.X.append(x)
        self.y.append(yv)
        self.C.append(cv)
        self.ok.append(bool(good))
        self.records.append(result)
        self._seen.add(self._key(full))
        self._resumed_ids.add(self._record_id(result))
        self._on_record(result, self.summary())
        return True

    def resume(self):
        n = skipped = 0
        for r in self.ev.history():
            identity = self._record_id(r)
            if identity in self._resumed_ids:
                continue
            self._resumed_ids.add(identity)
            if self._ingest(r, history=True):
                n += 1
            else:
                skipped += 1
        state_path = os.path.join(self.outdir, "state.json")
        restored = False
        if os.path.isfile(state_path):
            try:
                with open(state_path) as stream:
                    state = json.load(stream)
                if state.get("signature") == self._signature() and state.get("n") == len(self.X):
                    draws = state["sobol_draws"]
                    if draws > self.sobol.num_generated:
                        self.sobol.fast_forward(draws - self.sobol.num_generated)
                    for key in ("length", "success", "failure", "restarts", "best_value"):
                        value = state["turbo"][key]
                        if value is not None:
                            setattr(self.turbo, key, value)
                    stagnation_count = state.get("stagnation_count", 0)
                    exploration_restarts = state.get("exploration_restarts", 0)
                    exploration_pending = state.get("exploration_pending", False)
                    if (isinstance(stagnation_count, bool) or
                            not isinstance(stagnation_count, int) or stagnation_count < 0):
                        raise ValueError("invalid stagnation_count in checkpoint")
                    if (isinstance(exploration_restarts, bool) or
                            not isinstance(exploration_restarts, int) or exploration_restarts < 0):
                        raise ValueError("invalid exploration_restarts in checkpoint")
                    if not isinstance(exploration_pending, bool):
                        raise ValueError("invalid exploration_pending in checkpoint")
                    self.stagnation_count = stagnation_count
                    self.exploration_restarts = exploration_restarts
                    self.exploration_pending = exploration_pending
                    restored = True
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self._log(tr("checkpoint unavailable; rebuilding from history: %s") % exc)
        if not restored:
            self.stagnation_count = 0
            self.exploration_restarts = 0
            self.exploration_pending = False
            seed_count = len(self.initial_points) if self.initial_points else 0
            draws = max(len(self.X) - 1 - seed_count, 0)
            if draws > self.sobol.num_generated:
                self.sobol.fast_forward(draws - self.sobol.num_generated)
            feasible = [y for y, f in zip(self.y, self.feasible_mask()) if f]
            if feasible:
                self.turbo.best_value = max(feasible)
        if n:
            self._log(tr("resumed %d evaluations from history") % n)
        if skipped:
            self._log(tr("skipped %d history records incompatible with bounds/fixed parameters") % skipped)
        return n

    @staticmethod
    def _record_id(record):
        identity = (record.get("workdir"), record.get("trial"))
        return json.dumps(record, sort_keys=True) if identity == (None, None) else identity

    def _signature(self):
        cfg = {"space": self.space.params, "fixed": self.fixed,
               "objective": self.obj.to_cfg(), "constraints": [vars(c) for c in self.cons],
               "seed": self.seed, "batch": self.batch, "n_init": self.n_init}
        # Keep the old signature for the default configuration so checkpoints
        # written before seed support remain resumable.
        if self.initial_points:
            cfg["initial_points"] = self.initial_points
        if self.stagnation_rounds:
            cfg["stagnation_rounds"] = self.stagnation_rounds
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    def _key(self, params):
        return tuple(float(format(float(params[name]), ".12g")) for name in self.space.names)

    def _space_exhausted(self):
        return all(p.get("integer") for p in self.space.params) and len(self._seen) >= math.prod(
            math.floor(p["hi"]) - math.ceil(p["lo"]) + 1 for p in self.space.params)

    @staticmethod
    def _constraint_scale(constraint):
        if constraint.kind == "band":
            scale = max(abs(constraint.target), abs(constraint.tol))
        else:
            scale = abs(constraint.bound)
        return scale if math.isfinite(scale) and scale > 0 else 1.0

    def _progress_metric(self):
        """Return the best objective, or the best normalized violation so far."""
        feasible = self.feasible_mask()
        objective_values = [y for y, is_feasible in zip(self.y, feasible)
                            if is_feasible and y is not None and math.isfinite(y)]
        if objective_values:
            return "objective", max(objective_values)

        violations = []
        for ok, values in zip(self.ok, self.C):
            if not ok or values is None:
                continue
            total = 0.0
            for value, constraint in zip(values, self.cons):
                total += max(float(value), 0.0) / self._constraint_scale(constraint)
            if math.isfinite(total):
                violations.append(total)
        if violations:
            return "violation", min(violations)
        return None, None

    @staticmethod
    def _progress_improved(before, after):
        before_kind, before_value = before
        after_kind, after_value = after
        if after_kind is None:
            return False
        if before_kind is None:
            return True
        if after_kind == "objective":
            return before_kind != "objective" or after_value > before_value
        return before_kind == "violation" and after_value < before_value

    def _update_stagnation(self, before):
        if not self.stagnation_rounds:
            return
        if self._progress_improved(before, self._progress_metric()):
            self.stagnation_count = 0
            return
        self.stagnation_count += 1
        if (self.stagnation_count >= self.stagnation_rounds and
                not self.exploration_pending):
            self.exploration_pending = True
            self._log(tr("-- stagnation threshold reached; global Sobol exploration queued --"))

    def feasible_mask(self):
        return [o and c is not None and all(v <= 0 for v in c)
                for o, c in zip(self.ok, self.C)]

    def best(self):
        feas = self.feasible_mask()
        idx, val = None, -float("inf")
        for i, (y, f) in enumerate(zip(self.y, feas)):
            if f and y is not None and y > val:
                idx, val = i, y
        if idx is None:
            for i, (y, o) in enumerate(zip(self.y, self.ok)):
                if o and y is not None and y > val:
                    idx, val = i, y
        return idx, (self.records[idx] if idx is not None else None)

    def summary(self):
        bi, brec = self.best()
        return {"n": len(self.X), "n_ok": sum(self.ok),
                "n_failed": len(self.X) - sum(self.ok),
                "n_feasible": sum(self.feasible_mask()),
                "best_index": bi,
                "best_record": brec,
                "best_scalar": (self.y[bi] if bi is not None else None),
                "best_feasible": bool(bi is not None and self.feasible_mask()[bi]),
                "last_ok": bool(self.ok and self.ok[-1]),
                "last_feasible": bool(self.ok and self.feasible_mask()[-1]),
                "status": self.status, "budget": self.budget,
                "tr_length": self.turbo.length,
                "stagnation_count": self.stagnation_count,
                "exploration_restarts": self.exploration_restarts,
                "exploration_pending": self.exploration_pending}

    # ---------- model / proposal (same as optclient) ----------
    def _worst_objective(self):
        vals = [v for v, o in zip(self.y, self.ok) if o]
        if not vals:
            return 0.0
        lo, hi = min(vals), max(vals)
        return lo - 0.25 * max(hi - lo, 1.0)

    def _arrays(self):
        y_bad = self._worst_objective()
        c_bad = []
        for j in range(len(self.cons)):
            vals = [c[j] for c, o in zip(self.C, self.ok) if o and c]
            c_bad.append(max(1.0, max(vals) + 0.25 * max(1.0, abs(max(vals)))) if vals else 1.0)
        X = np.array(self.X, float)
        y = np.array([v if v is not None else y_bad for v in self.y],
                     float)
        C = (np.array([c if c is not None else c_bad for c in self.C],
                      float) if self.cons else None)
        return X, y, C

    def _unique_candidates(self, candidates, count):
        out, seen = [], set(self._seen)
        for params in candidates:
            key = self._key(params)
            if key not in seen:
                out.append(params)
                seen.add(key)
            if len(out) == count:
                return out
        for _ in range(max(256, 64 * count)):
            if len(out) == count:
                break
            params = {**self.fixed, **self.space.to_physical(self.sobol.draw(1).numpy().ravel())}
            key = self._key(params)
            if key not in seen:
                out.append(params)
                seen.add(key)
        if not out:
            self._log(tr("no unseen parameter points available; stopping without repeated simulations"))
        return out

    def propose(self, batch=None):
        batch = self.batch if batch is None else batch
        if self.stop_event.is_set():
            return [], False
        if self._space_exhausted():
            return [], False
        if self.exploration_pending:
            points = self._unique_candidates([], batch)
            if not points:
                return [], False
            self.turbo.restart()
            self.exploration_restarts += 1
            self.exploration_pending = False
            self.stagnation_count = 0
            self._log(tr("-- restarting trust region for global Sobol exploration --"))
            return points, False
        if not any(self.ok):
            self._log(tr("no valid metrics yet; continuing Sobol exploration"))
            return self._unique_candidates([], batch), False
        X, y, C = self._arrays()
        bi, _ = self.best()
        center = self.X[bi] if bi is not None else \
            self.X[int(np.argmax([v if v is not None else -1e9
                                  for v in self.y]))]
        try:
            cand, used_dkl = self.proposer.propose(
                X, y, C, batch, self.turbo.length, center,
                self.use_dkl, self.dkl_after)
            cand = np.asarray(cand, float)
            if cand.ndim != 2 or cand.shape[1] != self.space.dim or not np.isfinite(cand).all():
                raise ValueError("proposer returned invalid candidates")
            points = [{**self.fixed, **self.space.to_physical(x)} for x in cand]
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            self._log(tr("model proposal failed; using Sobol exploration: %s") % exc)
            points, used_dkl = [], False
        return self._unique_candidates(points, batch), used_dkl

    # ---------- run ----------
    def _evaluate(self, points):
        try:
            results = list(self.ev.evaluate_batch(points, parallel=self.batch))
        except Exception as exc:
            results = [{"ok": False, "error": str(exc)} for _ in points]
        for i, params in enumerate(points):
            result = results[i] if i < len(results) and isinstance(results[i], dict) else {
                "ok": False, "error": "evaluator returned no result"}
            result = {**result, "params": result.get("params") or params}
            if not self._ingest(result):
                self._ingest({**result, "params": params, "ok": False,
                              "metrics": None, "error": "evaluator returned incompatible parameters"})

    def initialize(self, budget=None):
        target = self.n_init if budget is None else min(self.n_init, budget)
        if self.initial_points:
            seen = set(self._seen)
            pending_keys = {self._key(point) for point in self.initial_points}
            pending = sum(key not in seen for key in pending_keys)
            nominal = {**self.fixed,
                        **self.space.to_physical(self.space.nominal_unit())}
            nominal_key = self._key(nominal)
            if nominal_key not in seen and nominal_key not in pending_keys:
                pending += 1
            target = max(target, len(self.X) + pending)
            if budget is not None:
                target = min(budget, target)
        need = target - len(self.X)
        if need <= 0:
            return
        initial = list(self.initial_points)
        if self.initial_points or not self.X:
            initial.append({**self.fixed,
                            **self.space.to_physical(self.space.nominal_unit())})
        pts = self._unique_candidates(initial, need)
        for k in range(0, len(pts), self.batch):
            if self.stop_event.is_set():
                return
            chunk = pts[k:k + self.batch]
            self._evaluate(chunk)
            self.save_best()
            s = self.summary()
            self._log(tr("init %d/%d  ok=%d feas=%d")
                      % (s["n"], target, s["n_ok"],
                         s["n_feasible"]))
        feasible = [y for y, f in zip(self.y, self.feasible_mask()) if f]
        if feasible:
            self.turbo.best_value = max(feasible)

    def step(self, batch=None):
        t0 = time.time()
        before_progress = self._progress_metric()
        plist, used_dkl = self.propose(batch)
        if self.stop_event.is_set() or not plist:
            return False
        start = len(self.X)
        self._evaluate(plist)
        batch_best = -float("inf")
        for y, feasible in zip(self.y[start:], self.feasible_mask()[start:]):
            if feasible:
                batch_best = max(batch_best, y)
        self.turbo.update(batch_best)
        if self.turbo.needs_restart:
            self.turbo.restart()
            self._log(tr("-- trust region collapsed, restarting --"))
        self._update_stagnation(before_progress)
        s = self.summary()
        head = self.obj.headline_metric()
        bh = "-"
        if s["best_record"] and s["best_record"].get("metrics"):
            bh = "%.4g" % s["best_record"]["metrics"].get(head,
                                                          float("nan"))
        self._log(tr("n=%d  %s  TR=%.3f  feas=%d  best[%s]=%s  %.1fs")
                  % (s["n"], "DKL-GP" if used_dkl else "GP",
                     self.turbo.length, s["n_feasible"], head, bh,
                     time.time() - t0))
        return True

    def run(self, budget):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("budget must be a positive integer")
        self.budget = budget
        self.status = "running"
        self._log(tr("device=%s (%s)  dim=%d  proposer=%s")
                  % (self.device, self.device_name, self.space.dim,
                     self.proposer.name))
        self._log(tr("objective: %s") % self.obj.describe())
        for c in self.cons:
            self._log(tr("constraint: %s") % c.describe())
        try:
            if not self.stop_event.is_set():
                self.initialize(budget)
            while len(self.X) < budget and not self.stop_event.is_set():
                if not self.step(min(self.batch, budget - len(self.X))):
                    self.status = "exhausted"
                    break
                self.save_best()
            if self.stop_event.is_set():
                self.status = "stopped"
            elif self.status != "exhausted":
                self.status = "completed"
        except BaseException:
            self.status = "interrupted"
            raise
        finally:
            self.save_best()
        return self.best()

    def save_best(self):
        bi, rec = self.best()
        path = os.path.join(self.outdir, "best.json")
        if rec is not None:
            out = {"trial": rec.get("trial"),
               "feasible": bool(self.feasible_mask()[bi]),
               "objective_scalar": self.y[bi],
               "objective": self.obj.describe(),
               "params": rec["params"], "metrics": rec["metrics"]}
            self._atomic_json(path, out)
        elif os.path.exists(path):
            os.unlink(path)
        self._atomic_json(os.path.join(self.outdir, "summary.json"), self.summary())
        turbo = {key: getattr(self.turbo, key) for key in
                 ("length", "success", "failure", "restarts", "best_value")}
        if not math.isfinite(turbo["best_value"]):
            turbo["best_value"] = None
        self._atomic_json(os.path.join(self.outdir, "state.json"), {
            "signature": self._signature(), "n": len(self.X),
            "sobol_draws": self.sobol.num_generated, "turbo": turbo,
            "stagnation_count": self.stagnation_count,
            "exploration_restarts": self.exploration_restarts,
            "exploration_pending": self.exploration_pending})
        self.export_history(os.path.join(self.outdir, "history.csv"))

    @staticmethod
    def _atomic_json(path, value):
        fd, tmp = tempfile.mkstemp(prefix=".vcal-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def export_history(self, path):
        params = sorted({k for r in self.records for k in (r.get("params") or {})})
        metrics = sorted({k for r in self.records for k in (r.get("metrics") or {})})
        fields = ["iteration", "trial", "ok", "feasible", "objective_scalar", "sim_s", "error"]
        fields += ["param." + k for k in params] + ["metric." + k for k in metrics]
        fd, tmp = tempfile.mkstemp(prefix=".vcal-", dir=os.path.dirname(os.path.abspath(path)))
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i, (r, ok, feasible) in enumerate(zip(self.records, self.ok, self.feasible_mask())):
                    row = {"iteration": i + 1, "trial": r.get("trial"), "ok": ok,
                           "feasible": feasible, "objective_scalar": self.y[i],
                           "sim_s": r.get("sim_s"), "error": r.get("error")}
                    row.update({"param." + k: v for k, v in (r.get("params") or {}).items()})
                    row.update({"metric." + k: v for k, v in (r.get("metrics") or {}).items()})
                    writer.writerow(row)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
