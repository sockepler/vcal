"""Constrained Bayesian optimization loop.

Strategy:
  * Sobol initialization (nominal design included as the first point)
  * objective surrogate: plain Matern GP, switching to a deep-kernel GP
    (MLP features + GP, trained on the local GPU) once enough data exists
  * one plain GP per constraint metric
  * batched qLogNEI with outcome constraints, optimized inside a TuRBO
    trust region centered on the best feasible point
  * failed simulations are imputed with a pessimistic objective and
    violated constraints, so the surrogates learn to avoid the region
"""
import json
import os
import time

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import ModelList
from botorch.optim import optimize_acqf
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch.quasirandom import SobolEngine

from .models import fit_dkl, fit_plain_gp
from .space import Space
from .turbo import TurboState

DTYPE = torch.double


class Constraint:
    """metric <= max  (or metric >= min) -> c(x) = signed violation <= 0."""

    def __init__(self, cfg):
        self.metric = cfg["metric"]
        self.is_max = "max" in cfg
        self.bound = float(cfg["max"] if self.is_max else cfg["min"])

    def value(self, metrics):
        v = float(metrics[self.metric])
        return (v - self.bound) if self.is_max else (self.bound - v)


class Optimizer:
    def __init__(self, server, outdir, batch=4, n_init=24, device=None,
                 use_dkl=True, dkl_after=48, seed=0):
        self.server = server
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.spec = server.spec()
        self.space = Space(self.spec["params"])
        self.obj_metric = self.spec["objective"]["metric"]
        self.sign = 1.0 if self.spec["objective"]["goal"] == "maximize" \
            else -1.0
        self.cons = [Constraint(c) for c in self.spec.get("constraints",
                                                          [])]
        self.batch = batch
        self.n_init = n_init
        self.use_dkl = use_dkl
        self.dkl_after = dkl_after
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.sobol = SobolEngine(self.space.dim, scramble=True, seed=seed)
        self.turbo = TurboState(self.space.dim, batch)
        # observations
        self.X = []        # unit-cube arrays
        self.y = []        # signed objective (always maximize)
        self.C = []        # constraint violation rows (<=0 feasible)
        self.ok = []       # sim success flags
        self.records = []
        self.histfile = os.path.join(outdir, "history.jsonl")

    # ---------- bookkeeping ----------
    def _worst_objective(self):
        vals = [v for v, o in zip(self.y, self.ok) if o]
        if not vals:
            return 0.0
        lo, hi = min(vals), max(vals)
        return lo - 0.25 * max(hi - lo, 1.0)

    def _ingest(self, result):
        params = result.get("params") or {}
        if not params:
            params = {p["name"]: p.get("nominal") for p in
                      self.spec["params"]}
        # fill unspecified params with nominal for the unit-cube encoding
        full = {p["name"]: params.get(p["name"], p.get("nominal"))
                for p in self.spec["params"]}
        if any(v is None for v in full.values()):
            return
        x = self.space.to_unit(full)
        good = bool(result.get("ok")) and result.get("metrics")
        if good:
            m = result["metrics"]
            yv = self.sign * float(m[self.obj_metric])
            cv = [c.value(m) for c in self.cons]
        else:
            yv = None      # imputed lazily at fit time
            cv = None
        self.X.append(x)
        self.y.append(yv)
        self.C.append(cv)
        self.ok.append(bool(good))
        self.records.append(result)
        with open(self.histfile, "a") as f:
            f.write(json.dumps(result) + "\n")

    def resume_from_server(self):
        hist = self.server.history()
        for r in hist:
            self._ingest(r)
        return len(hist)

    def _tensors(self):
        y_bad = self._worst_objective()
        c_bad = []
        for j in range(len(self.cons)):
            vals = [c[j] for c, o in zip(self.C, self.ok) if o]
            c_bad.append(max(vals) if vals else 1.0)
        X = torch.tensor(np.array(self.X), dtype=DTYPE,
                         device=self.device)
        y = torch.tensor([[v if v is not None else y_bad]
                          for v in self.y], dtype=DTYPE,
                         device=self.device)
        C = torch.tensor([c if c is not None else c_bad
                          for c in self.C], dtype=DTYPE,
                         device=self.device) \
            if self.cons else None
        return X, y, C

    def feasible_mask(self):
        return [o and c is not None and all(v <= 0 for v in c)
                for o, c in zip(self.ok, self.C)]

    def best(self):
        """(index, record) of best feasible point, or best objective if
        nothing is feasible yet."""
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

    # ---------- phases ----------
    def initialize(self, log=print):
        need = self.n_init - len(self.X)
        if need <= 0:
            return
        pts = [self.space.nominal_unit()] if not self.X else []
        while len(pts) < need:
            pts.append(self.sobol.draw(1).numpy().ravel())
        for k in range(0, len(pts), self.batch):
            chunk = pts[k:k + self.batch]
            plist = [self.space.to_physical(x) for x in chunk]
            results = self.server.evaluate_batch(plist,
                                                 parallel=self.batch)
            for r in results:
                self._ingest(r)
            nfeas = sum(self.feasible_mask())
            log("init %d/%d  ok=%d feas=%d"
                % (len(self.X), self.n_init, sum(self.ok), nfeas))

    def _fit_models(self, X, y, C):
        n = len(self.X)
        dkl = self.use_dkl and n >= self.dkl_after
        if dkl:
            obj_model = fit_dkl(X, y)
        else:
            obj_model = fit_plain_gp(X, y)
        con_models = [fit_plain_gp(X, C[:, j:j + 1])
                      for j in range(len(self.cons))]
        return obj_model, con_models, dkl

    def propose(self, log=print):
        X, y, C = self._tensors()
        obj_model, con_models, used_dkl = self._fit_models(X, y, C)
        model = ModelList(obj_model, *con_models)
        n_out = 1 + len(con_models)

        def obj_cb(Z, X=None):
            return Z[..., 0]

        cons_cb = [lambda Z, j=j: Z[..., j] for j in range(1, n_out)]
        sampler = ListSampler(*[
            SobolQMCNormalSampler(torch.Size([128]))
            for _ in range(n_out)])
        acq = qLogNoisyExpectedImprovement(
            model=model, X_baseline=X,
            sampler=sampler,
            objective=GenericMCObjective(obj_cb),
            constraints=cons_cb if con_models else None,
            prune_baseline=not used_dkl)

        # trust region around best (feasible) point
        bi, _ = self.best()
        center = self.X[bi] if bi is not None else \
            self.X[int(np.argmax([v if v is not None else -1e9
                                  for v in self.y]))]
        try:
            ls = obj_model.covar_module.base_kernel.lengthscale
            w = None if used_dkl else \
                ls.detach().cpu().numpy().ravel()
        except AttributeError:
            w = None
        lo, hi = self.turbo.bounds(center, w)
        bounds = torch.tensor(np.stack([lo, hi]), dtype=DTYPE,
                              device=self.device)
        cand, _ = optimize_acqf(
            acq, bounds=bounds, q=self.batch, num_restarts=8,
            raw_samples=256, options={"maxiter": 200, "batch_limit": 4})
        return [self.space.to_physical(x)
                for x in cand.detach().cpu().numpy()], used_dkl

    def step(self, log=print):
        t0 = time.time()
        plist, used_dkl = self.propose(log)
        results = self.server.evaluate_batch(plist, parallel=self.batch)
        batch_best = -float("inf")
        for r in results:
            self._ingest(r)
            if r.get("ok") and r.get("metrics"):
                cvals = [c.value(r["metrics"]) for c in self.cons]
                if all(v <= 0 for v in cvals):
                    batch_best = max(batch_best, self.sign *
                                     r["metrics"][self.obj_metric])
        self.turbo.update(batch_best)
        if self.turbo.needs_restart:
            self.turbo.restart()
            log("-- trust region collapsed, restarting exploration --")
        bi, brec = self.best()
        bstr = "none"
        if brec is not None and brec.get("metrics"):
            bstr = "%.4g" % brec["metrics"][self.obj_metric]
        log("n=%d  model=%s  TR=%.3f  batch_best=%s  best(%s)=%s  %.1fs"
            % (len(self.X), "DKL-GP" if used_dkl else "GP",
               self.turbo.length,
               ("%.4g" % (self.sign * batch_best)
                if batch_best > -float("inf") else "infeas"),
               self.obj_metric, bstr, time.time() - t0))

    def run(self, budget, log=print):
        log("device=%s  dim=%d  objective=%s(%s)  constraints=%d"
            % (self.device, self.space.dim,
               self.spec["objective"]["goal"], self.obj_metric,
               len(self.cons)))
        self.initialize(log)
        while len(self.X) < budget:
            self.step(log)
            self.save_best()
        self.save_best()
        return self.best()

    def save_best(self):
        bi, rec = self.best()
        if rec is None:
            return
        out = {"trial": rec.get("trial"),
               "feasible": self.feasible_mask()[bi],
               "params": rec["params"],
               "metrics": rec["metrics"]}
        with open(os.path.join(self.outdir, "best.json"), "w") as f:
            json.dump(out, f, indent=1)
