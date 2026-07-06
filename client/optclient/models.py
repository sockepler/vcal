"""Surrogate models: plain Matern GP and deep-kernel-learning GP.

The DKL GP feeds the inputs through a small MLP feature extractor and
puts a Matern GP on the learned features; extractor + GP hyperparameters
are trained jointly by maximizing the exact marginal log likelihood with
Adam — this is where the RTX 4090 earns its keep once the trial count
grows into the hundreds.

Both model types return posteriors in RAW objective units, so outcome
constraint thresholds can be specified in physical units either way.
"""
import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, dim_in, dim_feat=8, width=64):
        super().__init__(
            torch.nn.Linear(dim_in, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, dim_feat),
        )
        self.dim_feat = dim_feat


class DKLGP(gpytorch.models.ExactGP):
    """ExactGP over MLP features, trained on standardized targets;
    posterior() rescales back to raw units. Implements the minimal
    botorch Model API surface used by the acquisition functions."""

    def __init__(self, X, y_raw, dim_feat=8, width=64):
        y_raw = y_raw.reshape(-1)
        self._y_mean = y_raw.mean()
        self._y_std = y_raw.std().clamp_min(1e-6)
        yn = (y_raw - self._y_mean) / self._y_std
        lik = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
        super().__init__(X, yn, lik)
        self.extractor = FeatureExtractor(X.shape[-1], dim_feat, width)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=dim_feat))

    def forward(self, x):
        z = self.extractor(x)
        return MultivariateNormal(self.mean_module(z),
                                  self.covar_module(z))

    # --- botorch Model API ---
    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, observation_noise=False,
                  posterior_transform=None, **kwargs):
        self.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            mvn = self(X)          # ExactGP broadcasts batched test inputs
            if observation_noise:
                mvn = self.likelihood(mvn)
            mean = mvn.mean * self._y_std + self._y_mean
            cov = mvn.covariance_matrix * self._y_std ** 2
            mvn = MultivariateNormal(
                mean, cov + 1e-8 * torch.eye(cov.shape[-1], dtype=cov.dtype,
                                             device=cov.device))
        post = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            post = posterior_transform(post)
        return post


def fit_dkl(X, y, steps=400, lr=5e-3, dim_feat=8, width=64):
    """X: n x d in [0,1] (cuda ok), y: n x 1 raw units."""
    model = DKLGP(X, y, dim_feat=dim_feat, width=width).to(X)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    opt = torch.optim.Adam([
        {"params": model.extractor.parameters(), "lr": lr},
        {"params": model.mean_module.parameters(), "lr": lr * 4},
        {"params": model.covar_module.parameters(), "lr": lr * 4},
        {"params": model.likelihood.parameters(), "lr": lr * 4},
    ])
    for _ in range(steps):
        opt.zero_grad()
        out = model(model.train_inputs[0])
        loss = -mll(out, model.train_targets)
        loss.backward()
        opt.step()
    model.eval()
    return model


def fit_plain_gp(X, y):
    model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model
