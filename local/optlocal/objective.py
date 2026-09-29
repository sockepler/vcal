"""Flexible optimization objective.

The objective is a weighted sum of terms, each over one metric:

    goal: maximize   ->  + w * m / scale
    goal: minimize   ->  - w * m / scale
    goal: target     ->  - w * |m - target| / scale   (approach a value)

`scale` normalizes different metric magnitudes (default: |target| for
target terms, otherwise 1). The engine always MAXIMIZES the total.

YAML forms accepted:
    objective: {metric: gain_db, goal: maximize}                # legacy
    objective:
      - {metric: gain,     goal: target,  target: 8, weight: 1, tol: 0.2}
      - {metric: power_mw, goal: minimize, weight: 0.3, scale: 5}

A target term with `tol` ALSO produces an implicit constraint
|m - target| <= tol (checked for the feasibility report).
"""
import math


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number" % label)
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number" % label) from None
    if not math.isfinite(v):
        raise ValueError("%s must be a finite number" % label)
    return v


class Term:
    def __init__(self, cfg):
        if not isinstance(cfg, dict) or not isinstance(cfg.get("metric"), str):
            raise ValueError("objective term must contain a metric name")
        self.metric = cfg["metric"]
        self.goal = cfg.get("goal", "maximize")
        if self.goal not in ("maximize", "minimize", "target"):
            raise ValueError("goal must be maximize/minimize/target")
        if self.goal == "target" and "target" not in cfg:
            raise ValueError("target objective requires target")
        self.target = _finite(cfg.get("target", 0.0), "target")
        self.weight = _finite(cfg.get("weight", 1.0), "weight")
        self.tol = _finite(cfg["tol"], "tol") if cfg.get("tol") is not None else None
        if self.weight < 0 or (self.tol is not None and self.tol < 0):
            raise ValueError("weight and tol must be nonnegative")
        scale = cfg.get("scale")
        if scale is None:
            scale = abs(self.target) if (self.goal == "target"
                                         and self.target != 0) else 1.0
        self.scale = _finite(scale, "scale")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def value(self, metrics):
        m = float(metrics[self.metric])
        if self.goal == "maximize":
            return self.weight * m / self.scale
        if self.goal == "minimize":
            return -self.weight * m / self.scale
        return -self.weight * abs(m - self.target) / self.scale

    def describe(self):
        if self.goal == "target":
            s = "%s -> %g" % (self.metric, self.target)
            if self.tol is not None:
                s += " (±%g)" % float(self.tol)
        else:
            s = "%s %s" % (self.goal, self.metric)
        if self.weight != 1.0:
            s += " ×%g" % self.weight
        return s

    def to_cfg(self):
        d = {"metric": self.metric, "goal": self.goal,
             "weight": self.weight}
        if self.goal == "target":
            d["target"] = self.target
            if self.tol is not None:
                d["tol"] = float(self.tol)
        if self.scale != (abs(self.target) if self.goal == "target"
                          and self.target else 1.0):
            d["scale"] = self.scale
        return d


class Objective:
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = [cfg]
        if not isinstance(cfg, list):
            raise ValueError("objective must be a mapping or list of terms")
        self.terms = [Term(t) for t in cfg]
        if not self.terms:
            raise ValueError("empty objective")

    def scalar(self, metrics):
        return sum(t.value(metrics) for t in self.terms)

    def implicit_constraints(self):
        """[{metric, target, tol}] from target terms with tolerance."""
        return [{"metric": t.metric, "target": t.target,
                 "tol": float(t.tol)}
                for t in self.terms if t.goal == "target"
                and t.tol is not None]

    def describe(self):
        return "  +  ".join(t.describe() for t in self.terms)

    def headline_metric(self):
        """Metric to plot on the convergence chart (first term's)."""
        return self.terms[0].metric

    def to_cfg(self):
        lst = [t.to_cfg() for t in self.terms]
        return lst[0] if len(lst) == 1 else lst
