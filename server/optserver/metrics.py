"""Reusable transient, DC and ADC measurements for numerical circuit objectives.

Saved signals are sampled only within the available data range. Empty windows,
unsettled responses and undefined ratios are explicit measurement failures.
See docs/METRICS.md for units, conventions and legacy FFT phase-search behavior.
"""
import hashlib
import json
import math
import warnings

import numpy as np

from .measurements import WaveMeasurements
from .metricexpr import (expand_literals as _expand_literals, prepare_definitions,
                         validate_metric_definitions)

MEASUREMENT_VERSION = 2


def measurement_signature(defs, analyses, save=()):
    payload = {"version": MEASUREMENT_VERSION, "metrics": defs,
               "analyses": analyses, "save": sorted(save)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return value


def _count(value, name, minimum=1):
    parsed = _finite(value, name)
    if parsed < minimum or not parsed.is_integer():
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(parsed)


class Waves(WaveMeasurements):
    def __init__(self, t, sigs, analysis="tran"):
        if np.iscomplexobj(t) or any(np.iscomplexobj(v) for v in sigs.values()):
            raise ValueError("waveform axis and signals must be real")
        self.t = np.asarray(t, float)
        if self.t.ndim != 1 or len(self.t) < 2 or not np.isfinite(self.t).all():
            raise ValueError("waveform axis needs at least two finite samples")
        self.sigs = {name: np.asarray(values, float) for name, values in sigs.items()}
        if any(values.shape != self.t.shape for values in self.sigs.values()):
            raise ValueError("signal and sweep lengths differ")
        if analysis == "dc" and np.all(np.diff(self.t) < 0):
            self.t = self.t[::-1]
            self.sigs = {name: values[::-1] for name, values in self.sigs.items()}
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("waveform axis must be strictly monotonic")
        self.analysis = analysis

    def V(self, name):
        for key in (name, name + "!", name.upper(), name.lower()):
            if key in self.sigs:
                return self.sigs[key]
        matches = [value for key, value in self.sigs.items() if key.casefold() == name.casefold()]
        if len(matches) == 1:
            return matches[0]
        raise KeyError("signal %r not in saved traces %s" % (name, sorted(self.sigs)[:20]))

    def I(self, srcname):
        for suffix in (":p", "", ":1"):
            matches = [value for key, value in self.sigs.items()
                       if key.casefold() == (srcname + suffix).casefold()]
            if len(matches) == 1:
                return matches[0]
        raise KeyError("current %r not in saved traces" % srcname)

    def _vector(self, x):
        if np.iscomplexobj(x):
            raise ValueError("measurement requires a real signal")
        values = np.asarray(x, float)
        if values.shape != self.t.shape or not np.isfinite(values).all():
            raise ValueError("measurement requires a finite signal matching the waveform axis")
        return values

    def _range(self, t0=None, t1=None):
        lo = self.t[0] if t0 is None else _finite(t0, "window start")
        hi = self.t[-1] if t1 is None else _finite(t1, "window end")
        epsilon = 8 * np.finfo(float).eps * max(abs(self.t[0]), abs(self.t[-1]), self.t[-1] - self.t[0])
        if lo < self.t[0] - epsilon or hi > self.t[-1] + epsilon or lo > hi:
            raise ValueError("measurement window [%g, %g] is outside [%g, %g] or reversed" %
                             (lo, hi, self.t[0], self.t[-1]))
        return max(lo, self.t[0]), min(hi, self.t[-1])

    def _window(self, x, t0=None, t1=None, min_points=2):
        values = self._vector(x)
        lo, hi = self._range(t0, t1)
        interior = (self.t > lo) & (self.t < hi)
        axis = np.concatenate(([lo], self.t[interior], [hi])) if lo < hi else np.asarray([lo])
        if len(axis) < min_points:
            raise ValueError("measurement window has insufficient distinct samples")
        return axis, np.interp(axis, self.t, values)

    def _win(self, t0=None, t1=None):
        lo, hi = self._range(t0, t1)
        return (self.t >= lo) & (self.t <= hi)

    def at(self, x, tq):
        point = _finite(tq, "sample location")
        self._range(point, point)
        return float(np.interp(point, self.t, self._vector(x)))

    def avg(self, x, t0=None, t1=None):
        axis, values = self._window(x, t0, t1, min_points=1)
        if len(axis) == 1:
            return float(values[0])
        return float(np.sum(np.diff(axis) * (values[:-1] + values[1:]) * .5) /
                     (axis[-1] - axis[0]))

    def rms(self, x, t0=None, t1=None):
        axis, values = self._window(x, t0, t1, min_points=1)
        if len(axis) == 1:
            return abs(float(values[0]))
        a, b = values[:-1], values[1:]
        energy = np.sum(np.diff(axis) * (a*a + a*b + b*b) / 3)
        return float(np.sqrt(energy / (axis[-1] - axis[0])))

    def std(self, x, t0=None, t1=None):
        return self.rms(self._vector(x) - self.avg(x, t0, t1), t0, t1)

    def integ(self, x, t0=None, t1=None):
        axis, values = self._window(x, t0, t1)
        return float(np.sum(np.diff(axis) * (values[:-1] + values[1:]) * .5))

    def vmax(self, x, t0=None, t1=None):
        return float(np.max(self._window(x, t0, t1, min_points=1)[1]))

    def vmin(self, x, t0=None, t1=None):
        return float(np.min(self._window(x, t0, t1, min_points=1)[1]))

    def pp(self, x, t0=None, t1=None):
        return self.vmax(x, t0, t1) - self.vmin(x, t0, t1)

    def slice(self, x, t0, t1):
        return self._vector(x)[self._win(t0, t1)]

    def sample(self, x, fs, nsamp, t0=0.0, method="linear"):
        fs = _finite(fs, "sample rate")
        if fs <= 0:
            raise ValueError("sample rate must be positive")
        count = _count(nsamp, "sample count")
        times = _finite(t0, "sampling start") + np.arange(count) / fs
        self._range(times[0], times[-1])
        values = self._vector(x)
        if method == "linear":
            return np.interp(times, self.t, values)
        if method == "previous":
            return values[np.clip(np.searchsorted(self.t, times, side="right") - 1, 0, len(values)-1)]
        raise ValueError("sampling method must be linear or previous")

    def _fft_stat(self, key, x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48,
                  *, t0=None, window="rect", harmonics=5, bin_width=None, phase_mode=None):
        from .adc_metrics import spectrum
        if t0 is not None and (t0_lo is not None or t0_hi is not None):
            raise ValueError("choose fixed t0 or a phase interval, not both")
        if t0_lo is None and t0_hi is None:
            times = [self.t[0] if t0 is None else t0]
            mode = phase_mode or "fixed"
        elif t0_lo is None or t0_hi is None:
            raise ValueError("phase search requires both t0_lo and t0_hi")
        else:
            lo, hi = _finite(t0_lo, "t0_lo"), _finite(t0_hi, "t0_hi")
            if hi < lo:
                raise ValueError("phase interval is reversed")
            times = np.linspace(lo, hi, _count(nphase, "phase count"))
            mode = phase_mode or "best"
            if hi > lo and phase_mode is None:
                warnings.warn("legacy FFT phase search reports the best phase; use fixed t0 or phase_mode='worst' for robust targets",
                              RuntimeWarning, stacklevel=3)
        if mode not in ("fixed", "best", "worst") or (mode == "fixed" and len(times) != 1):
            raise ValueError("phase_mode must be fixed (one phase), best or worst")
        results = [spectrum(self.sample(x, fs, nsamp, start), fs, fund,
                            window=window, harmonics=harmonics, bin_width=bin_width)[key]
                   for start in times]
        if mode == "worst":
            return max(results) if key == "thd" else min(results)
        return min(results) if key == "thd" else max(results)

    def enob(self, x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs):
        return self._fft_stat("enob", x, fs, fund, nsamp, t0_lo, t0_hi, nphase, **kwargs)

    def sndr(self, x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs):
        return self._fft_stat("sndr", x, fs, fund, nsamp, t0_lo, t0_hi, nphase, **kwargs)

    def snr(self, x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs):
        return self._fft_stat("snr", x, fs, fund, nsamp, t0_lo, t0_hi, nphase, **kwargs)

    def thd(self, x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs):
        return self._fft_stat("thd", x, fs, fund, nsamp, t0_lo, t0_hi, nphase, **kwargs)

    def sfdr(self, x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs):
        return self._fft_stat("sfdr", x, fs, fund, nsamp, t0_lo, t0_hi, nphase, **kwargs)

    def environment(self):
        from .adc_metrics import spectrum, adc_static, transition_metrics, decode
        functions = {name: getattr(self, name) for name in (
            "V", "I", "avg", "rms", "vmax", "vmin", "at", "slice", "pp", "integ", "std",
            "cross", "rise_time", "fall_time", "delay", "slew_rate", "settle_time", "settled",
            "settling_error", "overshoot", "dc_gain", "dc_offset", "sample", "enob")}
        return {"__builtins__": {}, "np": np, "abs": abs, "min": min, "max": max,
                "float": float, "int": int, "round": round, "len": len, "sum": sum,
                "log10": math.log10, "log": math.log, "sqrt": math.sqrt,
                "db": lambda value: 20 * math.log10(abs(value)),
                **functions, "sndr_fft": self.sndr, "snr_fft": self.snr,
                "thd_fft": self.thd, "sfdr_fft": self.sfdr,
                "fft_metrics": spectrum, "adc_static": adc_static,
                "adc_transitions": transition_metrics, "decode_bits": decode,
                "t": self.t, "axis": self.t}


def compute_metrics(defs, t=None, sigs=None, *, datasets=None, default_analysis="tran"):
    data = dict(datasets or {})
    if t is not None:
        data.setdefault(default_analysis, (t, sigs))
    entries = prepare_definitions(defs, data, default_analysis)
    waves = {key: Waves(*values, analysis=key) for key, values in data.items()}
    environments = {key: value.environment() for key, value in waves.items()}
    out = {}
    for name, entry in entries.items():
        try:
            env = environments[entry["analysis"]]
            env["m"] = out
            with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
                value = eval(entry["code"], env)  # noqa: S307 -- numerical AST validated above.
            array = np.asarray(value)
            if array.shape != () or np.iscomplexobj(array):
                raise ValueError("metric must reduce to one real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("metric result is not finite")
            out[name] = value
        except Exception as exc:
            raise RuntimeError("metric %r [%s] failed: %s" % (name, entry["analysis"], exc)) from exc
    return {name: out[name] for name in defs}
