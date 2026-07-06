"""TuRBO-1 trust region bookkeeping (Eriksson et al., 2019).

The acquisition is optimized inside a hyper-rectangle centered on the
best (feasible) point; the box grows on consecutive successes and
shrinks on failures. When it collapses, the run restarts exploration.
"""
import math

import numpy as np


class TurboState:
    def __init__(self, dim, batch_size=4):
        self.dim = dim
        self.batch = batch_size
        self.length = 0.8
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.success = 0
        self.failure = 0
        self.success_tol = 3
        self.failure_tol = max(4, int(math.ceil(dim / batch_size)))
        self.best_value = -float("inf")
        self.restarts = 0

    def update(self, batch_best):
        if batch_best > self.best_value + 1e-4 * abs(self.best_value):
            self.success += 1
            self.failure = 0
        else:
            self.success = 0
            self.failure += 1
        self.best_value = max(self.best_value, batch_best)
        if self.success >= self.success_tol:
            self.length = min(2.0 * self.length, self.length_max)
            self.success = 0
        elif self.failure >= self.failure_tol:
            self.length /= 2.0
            self.failure = 0

    @property
    def needs_restart(self):
        return self.length < self.length_min

    def restart(self):
        self.length = 0.8
        self.success = self.failure = 0
        self.best_value = -float("inf")
        self.restarts += 1

    def bounds(self, x_center, weights=None):
        """Trust-region box in [0,1]^d around x_center, side scaled per
        dimension by the GP lengthscale weights (TuRBO trick)."""
        x_center = np.asarray(x_center)
        if weights is None:
            weights = np.ones(self.dim)
        w = weights / weights.mean()
        w = w / np.prod(np.power(w, 1.0 / self.dim))
        half = 0.5 * self.length * w
        lo = np.clip(x_center - half, 0.0, 1.0)
        hi = np.clip(x_center + half, 0.0, 1.0)
        return lo, hi
