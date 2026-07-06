"""Local constrained Bayesian-optimization engine.

Same algorithm as the Windows client (Sobol init -> GP / deep-kernel GP
+ TuRBO + batched constrained qLogNEI) but:
  * evaluation through an Evaluator (local in-process by default)
  * flexible Objective (maximize / minimize / approach-target terms)
  * pause/stop + per-record callbacks for the GUI
  * device auto-detect: CUDA and AMD ROCm both appear as torch.cuda
"""
import json
import os
import threading
import time

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from optclient.space import Space
from optclient.turbo import TurboState

from .objective import Objective
from .proposer import LocalProposer, RemoteProposer

DTYPE = torch.double


def pick_device(pref="auto"):
    """auto 默认 CPU：2026-07-06 microbench 实测，在典型 EDA sizing 规模
    （几十~几百个采样点、10-20 维）下，GP/DKL 的小矩阵计算 CPU 明显快于
    GPU（GPU 的 kernel launch 开销压过微小算量）——40点纯 GP：CPU ~7s vs
    AMD RX6800 ~15s。GPU 只有在点数上千 / DKL 网络更大时才可能翻盘。
    想用本地 GPU 或远端强卡显式指定 device=cuda/gpu 或 --remote-gpu。"""
    if pref in ("cuda", "gpu", "rocm"):
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
        return "cpu", "cpu (指定GPU但未检测到，回退CPU)"
    return "cpu", "cpu"


class Constraint:
    """max / min bound, or two-sided |m - target| <= tol."""

    def __init__(self, cfg):
        self.metric = cfg["metric"]
        if "tol" in cfg and "target" in cfg:
            self.kind = "band"
            self.target = float(cfg["target"])
            self.tol = float(cfg["tol"])
        elif "max" in cfg:
            self.kind = "max"
            self.bound = float(cfg["max"])
        else:
            self.kind = "min"
            self.bound = float(cfg["min"])

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
                 spec_params=None, fixed=None, remote_gpu=None):
        self.ev = evaluator
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.spec = evaluator.spec()
        # spec_params: optional subset/override of the search space
        # (GUI: enabled params with edited bounds); fixed: params pinned
        # to a constant value, merged into every evaluation.
        self.fixed = dict(fixed or {})
        self.space = Space(spec_params or self.spec["params"])
        self.obj = Objective(objective if objective is not None
                             else self.spec["objective"])
        cons_cfg = (constraints if constraints is not None
                    else self.spec.get("constraints", []))
        self.cons = [Constraint(c) for c in cons_cfg]
        self.cons += [Constraint(c) for c in
                      self.obj.implicit_constraints()]
        self.batch = batch
        self.n_init = n_init
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
        self.turbo = TurboState(self.space.dim, batch)
        self.stop_event = threading.Event()
        self._log = on_log or (lambda s: print(s, flush=True))
        self._on_record = on_record or (lambda rec, st: None)
        # observations
        self.X, self.y, self.C, self.ok, self.records = [], [], [], [], []

    # ---------- data ----------
    def _ingest(self, result):
        params = result.get("params") or {}
        full = {p["name"]: params.get(p["name"], p.get("nominal"))
                for p in self.space.params}
        if any(v is None for v in full.values()):
            return
        x = self.space.to_unit(full)
        good = bool(result.get("ok")) and result.get("metrics")
        yv = cv = None
        if good:
            try:
                m = result["metrics"]
                yv = self.obj.scalar(m)
                cv = [c.value(m) for c in self.cons]
            except (KeyError, ValueError):
                good = False
        self.X.append(x)
        self.y.append(yv)
        self.C.append(cv)
        self.ok.append(bool(good))
        self.records.append(result)
        self._on_record(result, self.summary())

    def resume(self):
        n = 0
        for r in self.ev.history():
            self._ingest(r)
            n += 1
        if n:
            self._log("resumed %d evaluations from history" % n)
        return n

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
                "n_feasible": sum(self.feasible_mask()),
                "best_index": bi,
                "best_record": brec,
                "best_scalar": (self.y[bi] if bi is not None else None),
                "tr_length": self.turbo.length}

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
            c_bad.append(max(vals) if vals else 1.0)
        X = np.array(self.X, float)
        y = np.array([v if v is not None else y_bad for v in self.y],
                     float)
        C = (np.array([c if c is not None else c_bad for c in self.C],
                      float) if self.cons else None)
        return X, y, C

    def propose(self):
        X, y, C = self._arrays()
        bi, _ = self.best()
        center = self.X[bi] if bi is not None else \
            self.X[int(np.argmax([v if v is not None else -1e9
                                  for v in self.y]))]
        cand, used_dkl = self.proposer.propose(
            X, y, C, self.batch, self.turbo.length, center,
            self.use_dkl, self.dkl_after)
        return ([{**self.fixed, **self.space.to_physical(x)}
                 for x in cand], used_dkl)

    # ---------- run ----------
    def initialize(self):
        need = self.n_init - len(self.X)
        if need <= 0:
            return
        pts = [self.space.nominal_unit()] if not self.X else []
        while len(pts) < need:
            pts.append(self.sobol.draw(1).numpy().ravel())
        for k in range(0, len(pts), self.batch):
            if self.stop_event.is_set():
                return
            chunk = pts[k:k + self.batch]
            for r in self.ev.evaluate_batch(
                    [{**self.fixed, **self.space.to_physical(x)}
                     for x in chunk],
                    parallel=self.batch):
                self._ingest(r)
            s = self.summary()
            self._log("init %d/%d  ok=%d feas=%d"
                      % (s["n"], self.n_init, s["n_ok"],
                         s["n_feasible"]))

    def step(self):
        t0 = time.time()
        plist, used_dkl = self.propose()
        results = self.ev.evaluate_batch(plist, parallel=self.batch)
        batch_best = -float("inf")
        for r in results:
            self._ingest(r)
            if r.get("ok") and r.get("metrics"):
                try:
                    cv = [c.value(r["metrics"]) for c in self.cons]
                    if all(v <= 0 for v in cv):
                        batch_best = max(batch_best,
                                         self.obj.scalar(r["metrics"]))
                except (KeyError, ValueError):
                    pass
        self.turbo.update(batch_best)
        if self.turbo.needs_restart:
            self.turbo.restart()
            self._log("-- trust region collapsed, restarting --")
        s = self.summary()
        head = self.obj.headline_metric()
        bh = "-"
        if s["best_record"] and s["best_record"].get("metrics"):
            bh = "%.4g" % s["best_record"]["metrics"].get(head,
                                                          float("nan"))
        self._log("n=%d  %s  TR=%.3f  feas=%d  best[%s]=%s  %.1fs"
                  % (s["n"], "DKL-GP" if used_dkl else "GP",
                     self.turbo.length, s["n_feasible"], head, bh,
                     time.time() - t0))

    def run(self, budget):
        self._log("device=%s (%s)  dim=%d  proposer=%s"
                  % (self.device, self.device_name, self.space.dim,
                     self.proposer.name))
        self._log("objective: %s" % self.obj.describe())
        for c in self.cons:
            self._log("constraint: %s" % c.describe())
        self.initialize()
        while len(self.X) < budget and not self.stop_event.is_set():
            self.step()
            self.save_best()
        self.save_best()
        return self.best()

    def save_best(self):
        bi, rec = self.best()
        if rec is None:
            return
        out = {"trial": rec.get("trial"),
               "feasible": bool(self.feasible_mask()[bi]),
               "objective_scalar": self.y[bi],
               "objective": self.obj.describe(),
               "params": rec["params"], "metrics": rec["metrics"]}
        with open(os.path.join(self.outdir, "best.json"), "w") as f:
            json.dump(out, f, indent=1)
