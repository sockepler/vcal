"""Parameter space: physical values <-> unit cube [0,1]^d."""
import math

import numpy as np


class Space:
    def __init__(self, spec_params):
        if not isinstance(spec_params, list) or not spec_params:
            raise ValueError("at least one parameter must be enabled")
        names = set()
        for p in spec_params:
            name = p["name"]
            if name in names:
                raise ValueError("duplicate parameter: %s" % name)
            names.add(name)
            lo, hi = p["lo"], p["hi"]
            if not all(math.isfinite(v) for v in (lo, hi)) or not lo < hi:
                raise ValueError("param %s: bounds must be finite and lo < hi" % name)
            if p.get("log") and lo <= 0:
                raise ValueError("param %s: log bounds must be positive" % name)
            if p.get("integer") and math.ceil(lo) > math.floor(hi):
                raise ValueError("param %s: range contains no integer" % name)
        self.params = spec_params
        self.names = [p["name"] for p in spec_params]
        self.dim = len(spec_params)

    def to_unit(self, values):
        """values: {name: physical} -> np.ndarray in [0,1]^d."""
        x = np.zeros(self.dim)
        for i, p in enumerate(self.params):
            v = float(values[p["name"]])
            lo, hi = p["lo"], p["hi"]
            if p.get("log"):
                x[i] = (math.log(v) - math.log(lo)) / \
                       (math.log(hi) - math.log(lo))
            else:
                x[i] = (v - lo) / (hi - lo)
        return np.clip(x, 0.0, 1.0)

    def to_physical(self, x):
        """x: array in [0,1]^d -> {name: physical}."""
        out = {}
        for i, p in enumerate(self.params):
            u = float(np.clip(x[i], 0.0, 1.0))
            lo, hi = p["lo"], p["hi"]
            if p.get("log"):
                v = math.exp(math.log(lo) + u * (math.log(hi) -
                                                 math.log(lo)))
            else:
                v = lo + u * (hi - lo)
            if p.get("integer"):
                v = min(max(int(round(v)), math.ceil(lo)), math.floor(hi))
            else:
                v = min(max(v, lo), hi)
            out[p["name"]] = v
        return out

    def nominal_unit(self):
        vals = {}
        for p in self.params:
            v = p.get("nominal")
            if v is None:
                v = math.sqrt(p["lo"] * p["hi"]) if p.get("log") \
                    else 0.5 * (p["lo"] + p["hi"])
            vals[p["name"]] = min(max(v, p["lo"]), p["hi"])
        return self.to_unit(vals)
