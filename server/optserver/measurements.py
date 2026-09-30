"""Reusable measurements for sampled transient and DC waveforms.

``WaveMeasurements`` is a mixin.  The waveform owner supplies ``t`` and the
boundary-aware ``_window`` and ``at`` helpers; this class only implements
measurements which are independent of how signals are stored.
"""
import math

import numpy as np


class WaveMeasurements:
    """Measurements over one strictly increasing time/scan axis."""

    _EDGES = ("rising", "falling")

    def _axis(self):
        t = np.asarray(self.t, dtype=float)
        if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
            raise ValueError("waveform axis must be a finite 1-D array with at least two points")
        if not np.all(np.diff(t) > 0):
            raise ValueError("waveform axis must be strictly increasing")
        return t

    @staticmethod
    def _finite(value, label):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("%s must be a finite number" % label)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("%s must be a finite number" % label) from None
        if not math.isfinite(value):
            raise ValueError("%s must be a finite number" % label)
        return value

    def _wave(self, x, label="waveform"):
        t = self._axis()
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.size != t.size:
            raise ValueError("%s must be a 1-D array matching the waveform axis" % label)
        if not np.isfinite(values).all():
            raise ValueError("%s must contain only finite values" % label)
        return t, values

    def _window_data(self, x, start=None, end=None, min_points=2):
        self._wave(x)
        if start is not None:
            start = self._finite(start, "start")
        if end is not None:
            end = self._finite(end, "end")
        if start is not None and end is not None and not start < end:
            raise ValueError("start must be smaller than end")
        try:
            axis, values = self._window(x, start, end, min_points=min_points)
        except TypeError:
            # Keep compatibility with a parent implementation using positional
            # min_points while retaining the same no-extrapolation contract.
            axis, values = self._window(x, start, end, min_points)
        axis = np.asarray(axis, dtype=float)
        values = np.asarray(values, dtype=float)
        if (axis.ndim != 1 or values.ndim != 1 or axis.size != values.size or
                axis.size < min_points or not np.isfinite(axis).all() or
                not np.isfinite(values).all() or not np.all(np.diff(axis) > 0)):
            raise ValueError("window must contain finite, strictly increasing samples")
        return axis, values

    @staticmethod
    def _edge(edge):
        if edge not in WaveMeasurements._EDGES:
            raise ValueError("edge must be 'rising' or 'falling'")
        return edge

    @staticmethod
    def _nth(nth):
        if isinstance(nth, (bool, np.bool_)) or not isinstance(nth, (int, np.integer)):
            raise ValueError("nth must be a positive integer")
        if nth < 1:
            raise ValueError("nth must be a positive integer")
        return int(nth)

    @staticmethod
    def _levels(low, high):
        low = WaveMeasurements._finite(low, "low")
        high = WaveMeasurements._finite(high, "high")
        if not 0 <= low < high <= 1:
            raise ValueError("low and high must satisfy 0 <= low < high <= 1")
        return low, high

    def _cross_times(self, x, level, edge="rising", start=None, end=None):
        edge = self._edge(edge)
        level = self._finite(level, "level")
        ts, xs = self._window_data(x, start, end, min_points=2)
        d = xs - level
        events = []
        i = 0
        n = len(d)
        while i < n:
            if d[i] == 0:
                j = i
                while j + 1 < n and d[j + 1] == 0:
                    j += 1
                before = d[i - 1] if i else None
                after = d[j + 1] if j + 1 < n else None
                direction = None
                if before is not None and after is not None:
                    if before < 0 < after:
                        direction = "rising"
                    elif before > 0 > after:
                        direction = "falling"
                elif before is None and after is not None:
                    if after > 0:
                        direction = "rising"
                    elif after < 0:
                        direction = "falling"
                elif before is not None:
                    if before < 0:
                        direction = "rising"
                    elif before > 0:
                        direction = "falling"
                if direction == edge:
                    # A threshold plateau is one event at its arrival, even
                    # if it extends to the right boundary of the window.
                    events.append(float(ts[i]))
                i = j + 1
                continue
            if i + 1 < n and ((d[i] < 0 < d[i + 1]) or (d[i] > 0 > d[i + 1])):
                direction = "rising" if d[i] < 0 else "falling"
                if direction == edge:
                    frac = -d[i] / (d[i + 1] - d[i])
                    events.append(float(ts[i] + frac * (ts[i + 1] - ts[i])))
            i += 1
        return events

    def cross(self, x, level, edge="rising", nth=1, start=None, end=None):
        """Return the ``nth`` linear crossing time of ``level``."""
        nth = self._nth(nth)
        events = self._cross_times(x, level, edge, start, end)
        if len(events) < nth:
            raise ValueError("waveform has fewer than %d %s crossing(s) of level %g"
                             % (nth, edge, float(level)))
        return events[nth - 1]

    def _transition(self, x, start=None, end=None, low=.1, high=.9,
                    initial=None, final=None, kind="transition"):
        low, high = self._levels(low, high)
        ts, xs = self._window_data(x, start, end, min_points=2)
        initial = float(xs[0]) if initial is None else self._finite(initial, "initial")
        final = float(xs[-1]) if final is None else self._finite(final, "final")
        step = final - initial
        if step == 0:
            raise ValueError("%s requires distinct initial and final levels" % kind)
        edge = "rising" if step > 0 else "falling"
        low_level = initial + low * step
        high_level = initial + high * step
        low_events = self._cross_times(x, low_level, edge, start, end)
        if not low_events:
            raise ValueError("waveform has no %s low-level crossing" % kind)
        for t_high in self._cross_times(x, high_level, edge, start, end):
            index = int(np.searchsorted(low_events, t_high, side="left")) - 1
            if index >= 0:
                # Skip an incomplete early glitch; use the latest low crossing.
                t_low = low_events[index]
                break
        else:
            raise ValueError("waveform has no ordered %s high-level crossing" % kind)
        if not t_high > t_low:
            raise ValueError("%s crossing interval must be positive" % kind)
        return ts, xs, initial, final, t_low, t_high

    def rise_time(self, x, start=None, end=None, low=.1, high=.9,
                  initial=None, final=None):
        ts, xs, initial, final, t_low, t_high = self._transition(
            x, start, end, low, high, initial, final, "rise_time")
        if final <= initial:
            raise ValueError("rise_time requires final greater than initial")
        return float(t_high - t_low)

    def fall_time(self, x, start=None, end=None, low=.1, high=.9,
                  initial=None, final=None):
        ts, xs, initial, final, t_low, t_high = self._transition(
            x, start, end, low, high, initial, final, "fall_time")
        if final >= initial:
            raise ValueError("fall_time requires final smaller than initial")
        return float(t_high - t_low)

    def delay(self, x, y, level_x, level_y, edge_x="rising", edge_y="rising",
              nth=1, start=None, end=None, pairing="causal"):
        """Delay from the nth input edge to its following output response.

        A response must precede the next input edge of the same direction.
        ``pairing='ordinal'`` preserves signed nth-to-nth edge subtraction.
        """
        nth = self._nth(nth)
        if pairing not in ("causal", "ordinal"):
            raise ValueError("pairing must be 'causal' or 'ordinal'")
        inputs = self._cross_times(x, level_x, edge_x, start, end)
        outputs = self._cross_times(y, level_y, edge_y, start, end)
        if len(inputs) < nth:
            raise ValueError("delay has fewer than %d input crossings" % nth)
        tx = inputs[nth - 1]
        if pairing == "ordinal":
            if len(outputs) < nth:
                raise ValueError("delay has fewer than %d output crossings" % nth)
            return float(outputs[nth - 1] - tx)
        next_input = inputs[nth] if nth < len(inputs) else math.inf
        responses = [t for t in outputs if tx <= t < next_input]
        if not responses:
            raise ValueError("delay has no causal output crossing before the next input or window end")
        return float(responses[0] - tx)

    def slew_rate(self, x, start=None, end=None, low=.1, high=.9,
                  initial=None, final=None):
        _, _, initial, final, t_low, t_high = self._transition(
            x, start, end, low, high, initial, final, "slew_rate")
        return float(abs(final - initial) * (high - low) /
                     (t_high - t_low))

    @staticmethod
    def _on_unsettled(mode):
        if mode not in ("raise", "window"):
            raise ValueError("on_unsettled must be 'raise' or 'window'")
        return mode

    @staticmethod
    def _weighted_tail(ts, xs):
        span = ts[-1] - ts[0]
        if span == 0:
            return float(xs[-1])
        tail_start = ts[-1] - .05 * span
        tail = np.interp(tail_start, ts, xs)
        tail_t = np.concatenate(([tail_start], ts[ts > tail_start]))
        tail_x = np.concatenate(([tail], xs[ts > tail_start]))
        if tail_t.size < 2 or tail_t[-1] == tail_t[0]:
            return float(tail_x[-1])
        integral = np.sum(np.diff(tail_t) * (tail_x[:-1] + tail_x[1:]) * .5)
        return float(integral / (tail_t[-1] - tail_t[0]))

    def _settle_data(self, x, start=None, end=None, final=None, initial=None,
                     tol=.005, atol=None, hold=0):
        tol = self._finite(tol, "tol")
        if tol < 0:
            raise ValueError("tol must be nonnegative")
        if atol is not None:
            atol = self._finite(atol, "atol")
            if atol < 0:
                raise ValueError("atol must be nonnegative")
        hold = self._finite(hold, "hold")
        if hold < 0:
            raise ValueError("hold must be nonnegative")
        ts, xs = self._window_data(x, start, end, min_points=2)
        initial = float(xs[0]) if initial is None else self._finite(initial, "initial")
        final = (self._weighted_tail(ts, xs) if final is None
                 else self._finite(final, "final"))
        step = abs(final - initial)
        if step == 0 and (atol is None or atol == 0):
            raise ValueError("settle_time requires a nonzero step or positive atol")
        # An explicit absolute error band must not be
        # widened by the default relative tolerance.
        band = tol * step if atol is None else atol
        return ts, xs, initial, final, band, hold

    def settle_time(self, x, start=None, end=None, tol=.005, final=None,
                    atol=None, initial=None, hold=0, on_unsettled="raise"):
        mode = self._on_unsettled(on_unsettled)
        ts, xs, initial, final, band, hold = self._settle_data(
            x, start, end, final, initial, tol, atol, hold)
        outside = np.abs(xs - final) > band
        duration = float(ts[-1] - ts[0])
        if not outside.any():
            settle_at = float(ts[0])
        else:
            last = int(np.flatnonzero(outside)[-1])
            if last == len(xs) - 1:
                if mode == "window":
                    return duration
                raise ValueError("waveform did not settle within the requested window")
            next_error = abs(xs[last + 1] - final)
            if next_error > band:
                if mode == "window":
                    return duration
                raise ValueError("waveform did not settle within the requested window")
            boundary = final + band if xs[last] > final else final - band
            delta = xs[last + 1] - xs[last]
            if delta == 0:
                if mode == "window":
                    return duration
                raise ValueError("waveform did not settle within the requested window")
            frac = (boundary - xs[last]) / delta
            settle_at = float(ts[last] + frac * (ts[last + 1] - ts[last]))
        if ts[-1] - settle_at < hold:
            if mode == "window":
                return duration
            raise ValueError("waveform did not remain settled for hold duration")
        return float(settle_at - ts[0])

    def settled(self, x, start=None, end=None, tol=.005, final=None,
                atol=None, initial=None, hold=0, on_unsettled="raise"):
        self._on_unsettled(on_unsettled)
        # Validate the numerical arguments outside the unsettled exception
        # path so ``on_unsettled='window'`` cannot hide malformed input.
        self._settle_data(x, start, end, final, initial, tol, atol, hold)
        try:
            self.settle_time(x, start, end, tol, final, atol, initial, hold,
                             "raise")
        except ValueError:
            return 0
        return 1

    def settling_error(self, x, start=None, end=None, final=None,
                       relative=False, initial=None):
        ts, xs = self._window_data(x, start, end, min_points=2)
        initial = float(xs[0]) if initial is None else self._finite(initial, "initial")
        final = (self._weighted_tail(ts, xs) if final is None
                 else self._finite(final, "final"))
        error = float(np.max(np.abs(xs - final)))
        if relative:
            step = abs(final - initial)
            if step == 0:
                raise ValueError("relative settling error requires a nonzero step")
            error /= step
        return error

    def overshoot(self, x, start=None, end=None, initial=None, final=None):
        ts, xs = self._window_data(x, start, end, min_points=2)
        initial = float(xs[0]) if initial is None else self._finite(initial, "initial")
        final = (self._weighted_tail(ts, xs) if final is None
                 else self._finite(final, "final"))
        step = final - initial
        if step == 0:
            if np.max(np.abs(xs - final)) == 0:
                return 0.0
            raise ValueError("overshoot requires a nonzero step")
        if step > 0:
            return float(max(0.0, (np.max(xs) - final) / step))
        return float(max(0.0, (final - np.min(xs)) / abs(step)))

    def _dc_data(self, y, x=None, start=None, end=None):
        t, yw = self._window_data(y, start, end, min_points=2)
        if x is None:
            xw = t.copy()
            xa = self._axis()
        else:
            _, xa = self._wave(x, "dc input")
            xw = np.interp(t, self._axis(), xa)
        if xw.size < 2 or not np.isfinite(xw).all():
            raise ValueError("dc input must contain at least two finite points")
        dx = np.diff(xw)
        if not (np.all(dx > 0) or np.all(dx < 0)):
            raise ValueError("dc input must be strictly monotonic")
        if xw[-1] == xw[0]:
            raise ValueError("dc input must have nonzero span")
        return t, xw, yw

    def dc_gain(self, y, x=None, at=None, start=None, end=None):
        _, xw, yw = self._dc_data(y, x, start, end)
        if at is None:
            xc = xw - np.mean(xw)
            yc = yw - np.mean(yw)
            denom = float(np.dot(xc, xc))
            if denom == 0:
                raise ValueError("dc input must have nonzero variance")
            return float(np.dot(xc, yc) / denom)
        at = self._finite(at, "at")
        increasing = xw[0] < xw[-1]
        xx = xw if increasing else xw[::-1]
        yy = yw if increasing else yw[::-1]
        if at < xx[0] or at > xx[-1]:
            raise ValueError("at must lie within the selected dc input range")
        i = int(np.searchsorted(xx, at, side="right") - 1)
        i = min(max(i, 0), len(xx) - 2)
        dx = xx[i + 1] - xx[i]
        if dx == 0:
            raise ValueError("dc input must have nonzero local span")
        return float((yy[i + 1] - yy[i]) / dx)

    def dc_offset(self, y, x=None, start=None, end=None):
        _, xw, yw = self._dc_data(y, x, start, end)
        xc = xw - np.mean(xw)
        denom = float(np.dot(xc, xc))
        if denom == 0:
            raise ValueError("dc input must have nonzero variance")
        slope = float(np.dot(xc, yw - np.mean(yw)) / denom)
        return float(np.mean(yw) - slope * np.mean(xw))


__all__ = ["WaveMeasurements"]
