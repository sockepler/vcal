"""Metric extraction from transient waveforms.

Metrics are defined in the circuit YAML as expressions over saved signals,
evaluated with numpy and the helpers below. Example:

    metrics:
      vod_final:  "avg(V('VOP') - V('VON'), 90n, 100n)"
      gain:       "m['vod_final'] / 0.1"
      tsettle:    "settle_time(V('VOP') - V('VON'), start=15n, tol=0.005)"
      power_mw:   "avg(-I('V_VDD'), 20n, 100n) * 1.8 * 1e3"

Earlier metrics are visible to later ones through `m`.
Suffixed literals like `90n` are rewritten to floats before eval.
"""
import math
import re

import numpy as np

_SUFFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3, "m": 1e-3,
           "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}

_LIT = re.compile(r"\b([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)([TGMKkmunpfa])\b")


def _expand_literals(expr):
    """Rewrite '90n' -> '(90*1e-9)' so eng notation works in expressions."""
    return _LIT.sub(lambda m: "(%s*%g)" % (m.group(1), _SUFFIX[m.group(2)]),
                    expr)


class Waves:
    def __init__(self, t, sigs):
        self.t = t
        self.sigs = sigs

    # ---- signal access ----
    def V(self, name):
        for k in (name, name + "!", name.upper(), name.lower()):
            if k in self.sigs:
                return self.sigs[k]
        raise KeyError("signal %r not in saved traces %s"
                       % (name, sorted(self.sigs)[:20]))

    def I(self, srcname):
        """Terminal current of a source, saved as e.g. 'V_VDD:p'."""
        for base in (srcname, srcname.lower(), srcname.upper()):
            for k in (base + ":p", base, base + ":1"):
                if k in self.sigs:
                    return self.sigs[k]
        raise KeyError("current %r not in saved traces" % srcname)

    # ---- window helpers ----
    def _win(self, t0=None, t1=None):
        sel = np.ones(len(self.t), bool)
        if t0 is not None:
            sel &= self.t >= t0
        if t1 is not None:
            sel &= self.t <= t1
        return sel

    def avg(self, x, t0=None, t1=None):
        sel = self._win(t0, t1)
        if not sel.any():
            return float("nan")
        return float(np.trapezoid(x[sel], self.t[sel])
                     / (self.t[sel][-1] - self.t[sel][0])) \
            if sel.sum() > 1 else float(x[sel][0])

    def rms(self, x, t0=None, t1=None):
        sel = self._win(t0, t1)
        return float(np.sqrt(np.mean(np.square(x[sel])))) if sel.any() \
            else float("nan")

    def vmax(self, x, t0=None, t1=None):
        sel = self._win(t0, t1)
        return float(np.max(x[sel])) if sel.any() else float("nan")

    def vmin(self, x, t0=None, t1=None):
        sel = self._win(t0, t1)
        return float(np.min(x[sel])) if sel.any() else float("nan")

    def at(self, x, tq):
        return float(np.interp(tq, self.t, x))

    def slice(self, x, t0, t1):
        sel = self._win(t0, t1)
        return x[sel]

    def settle_time(self, x, start=0.0, end=None, tol=0.005, final=None):
        """Time (from `start`) after which x stays within tol (relative to
        the settling step) of its final value. Returns end-start if it
        never settles."""
        sel = self._win(start, end)
        if sel.sum() < 4:
            return float("nan")
        ts, xs = self.t[sel], x[sel]
        xf = float(np.mean(xs[max(1, int(0.95 * len(xs))):])) \
            if final is None else final
        step = abs(xf - xs[0])
        band = tol * (step if step > 1e-9 else max(abs(xf), 1e-9))
        outside = np.abs(xs - xf) > band
        if not outside.any():
            return 0.0
        last_out = np.max(np.nonzero(outside)[0])
        if last_out >= len(ts) - 1:
            return float(ts[-1] - ts[0])   # never settles in window
        return float(ts[last_out + 1] - ts[0])

    def enob(self, x, fs, fund, nsamp, t0_lo, t0_hi, nphase=48):
        """相干 FFT ENOB：在 [t0_lo, t0_hi] 扫采样相位（避开这个项目反复
        踩坑的 measurement-aliasing），每个相位取 nsamp 个 @1/fs 间隔的
        样本，去均值后 FFT，基波在 bin=fund，SNDR=sig/其余噪声，
        ENOB=(SNDR-1.76)/6.02。返回相位扫描到的最佳 ENOB。"""
        Ts = 1.0 / fs
        best = float("-inf")
        for t0 in np.linspace(t0_lo, t0_hi, int(nphase)):
            ts = t0 + np.arange(int(nsamp)) * Ts
            if ts[-1] > self.t[-1]:
                continue
            xs = np.interp(ts, self.t, x)
            xs = xs - xs.mean()
            p = np.abs(np.fft.rfft(xs)) ** 2
            if int(fund) >= len(p):
                continue
            sig = p[int(fund)]
            noise = float(p[1:int(nsamp) // 2].sum()) - sig
            if sig <= 0 or noise <= 0:
                continue
            sndr = 10 * math.log10(sig / noise)
            e = (sndr - 1.76) / 6.02
            if e > best:
                best = e
        return best if best > float("-inf") else float("nan")

    def sndr(self, x, fs, fund, nsamp, t0_lo, t0_hi, nphase=48):
        e = self.enob(x, fs, fund, nsamp, t0_lo, t0_hi, nphase)
        return e * 6.02 + 1.76

    def overshoot(self, x, start=0.0, end=None):
        """Peak deviation beyond final value, relative to step size."""
        sel = self._win(start, end)
        if sel.sum() < 4:
            return float("nan")
        xs = x[sel]
        xf = float(np.mean(xs[max(1, int(0.95 * len(xs))):]))
        step = xf - xs[0]
        if abs(step) < 1e-12:
            return 0.0
        peak = np.max(xs) if step > 0 else np.min(xs)
        return float(max(0.0, (peak - xf) / step)) if step > 0 \
            else float(max(0.0, (xf - peak) / -step))


def compute_metrics(defs, t, sigs):
    """defs: ordered {name: expr}. Returns {name: float}."""
    w = Waves(t, sigs)
    env = {"__builtins__": {},
           "np": np, "abs": abs, "min": min, "max": max, "float": float,
           "log10": math.log10, "log": math.log, "sqrt": math.sqrt,
           "db": lambda v: 20 * math.log10(abs(v)) if v else float("-inf"),
           "V": w.V, "I": w.I, "avg": w.avg, "rms": w.rms,
           "vmax": w.vmax, "vmin": w.vmin, "at": w.at, "slice": w.slice,
           "settle_time": w.settle_time, "overshoot": w.overshoot,
           "enob": w.enob, "sndr_fft": w.sndr,
           "t": t}
    out = {}
    env["m"] = out
    for name, expr in defs.items():
        try:
            v = eval(_expand_literals(str(expr)), env)   # noqa: S307
            out[name] = float(v)
        except Exception as e:
            raise RuntimeError("metric %r failed: %s" % (name, e))
    return out
