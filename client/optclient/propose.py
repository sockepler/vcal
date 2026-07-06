"""Shared candidate-proposal core: GP / deep-kernel GP fitting + batched
constrained qLogNEI inside a TuRBO trust region.

Used by three callers:
  * the Windows remote-BO client (optclient.optimize)
  * the local engine (optlocal.engine)
  * the remote GPU compute server (optclient.gpuserver)

Pure function of numpy inputs -> numpy candidates, so it can sit behind
an HTTP API.
"""
import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import ModelList
from botorch.optim import optimize_acqf
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from .models import fit_dkl, fit_plain_gp

DTYPE = torch.double


def tr_bounds(center, length, dim, weights=None):
    """TuRBO trust-region box in [0,1]^d (lengthscale-weighted)."""
    center = np.asarray(center, float)
    if weights is None:
        weights = np.ones(dim)
    w = np.asarray(weights, float)
    w = w / w.mean()
    w = w / np.prod(np.power(w, 1.0 / dim))
    half = 0.5 * float(length) * w
    return (np.clip(center - half, 0.0, 1.0),
            np.clip(center + half, 0.0, 1.0))


def propose_candidates(X, y, C, batch, tr_length, center,
                       use_dkl=True, dkl_after=48, device="cpu"):
    """X: n×d unit-cube, y: n (objective, maximize, failures imputed),
    C: n×m violations (<=0 feasible) or None. Returns (q×d ndarray,
    used_dkl)."""
    X = torch.tensor(np.asarray(X, float), dtype=DTYPE, device=device)
    y = torch.tensor(np.asarray(y, float).reshape(-1, 1), dtype=DTYPE,
                     device=device)
    n, dim = X.shape
    used_dkl = bool(use_dkl and n >= dkl_after)
    obj_model = fit_dkl(X, y) if used_dkl else fit_plain_gp(X, y)
    con_models = []
    if C is not None and np.size(C):
        Ct = torch.tensor(np.asarray(C, float), dtype=DTYPE,
                          device=device)
        con_models = [fit_plain_gp(X, Ct[:, j:j + 1])
                      for j in range(Ct.shape[1])]
    model = ModelList(obj_model, *con_models)
    n_out = 1 + len(con_models)
    sampler = ListSampler(*[SobolQMCNormalSampler(torch.Size([128]))
                            for _ in range(n_out)])
    acq = qLogNoisyExpectedImprovement(
        model=model, X_baseline=X, sampler=sampler,
        objective=GenericMCObjective(lambda Z, X=None: Z[..., 0]),
        constraints=[lambda Z, j=j: Z[..., j]
                     for j in range(1, n_out)] or None,
        prune_baseline=not used_dkl)
    try:
        ls = obj_model.covar_module.base_kernel.lengthscale
        weights = None if used_dkl else ls.detach().cpu().numpy().ravel()
    except AttributeError:
        weights = None
    lo, hi = tr_bounds(center, tr_length, dim, weights)
    bounds = torch.tensor(np.stack([lo, hi]), dtype=DTYPE, device=device)
    cand, _ = optimize_acqf(
        acq, bounds=bounds, q=int(batch), num_restarts=8,
        raw_samples=256, options={"maxiter": 200, "batch_limit": 4})
    return cand.detach().cpu().numpy(), used_dkl
