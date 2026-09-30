"""Measurements for thresholded periodic and block-like waveforms.

``BlockMeasurements`` is a mixin.  Its waveform owner supplies the common
window and crossing primitives; this module only defines measurements whose
natural unit is a complete pulse or cycle.
"""
import math

import numpy as np


class BlockMeasurements:
    """Threshold, period, jitter, slope, and trigger/value measurements."""

    @staticmethod
    def _polarity(polarity):
        if polarity not in ("high", "low"):
            raise ValueError("polarity must be 'high' or 'low'")
        return polarity

    @staticmethod
    def _edge(edge):
        if edge not in ("rising", "falling"):
            raise ValueError("edge must be 'rising' or 'falling'")
        return edge

    def _periods(self, x, level, edge="rising", start=None, end=None):
        edge = self._edge(edge)
        crossings = list(self._cross_times(x, level, edge, start, end))
        if len(crossings) < 2:
            raise ValueError("at least two %s crossings are required" % edge)
        periods = np.diff(np.asarray(crossings, dtype=float))
        if periods.size == 0 or not np.all(periods > 0):
            raise ValueError("crossing times must be strictly increasing")
        if not np.isfinite(periods).all():
            raise ValueError("crossing times must be finite")
        return periods

    def _complete_pulses(self, x, level, polarity, start=None, end=None):
        begin_edge = "rising" if polarity == "high" else "falling"
        finish_edge = "falling" if polarity == "high" else "rising"
        begins = self._cross_times(x, level, begin_edge, start, end)
        finishes = self._cross_times(x, level, finish_edge, start, end)
        events = ([(float(t), begin_edge) for t in begins] +
                  [(float(t), finish_edge) for t in finishes])
        events.sort(key=lambda item: (item[0], item[1] != begin_edge))

        widths = []
        active = None
        for when, kind in events:
            if kind == begin_edge:
                # A begin edge while active belongs to a malformed or noisy
                # sequence; retain the earliest active edge until it closes.
                if active is None:
                    active = when
            elif active is not None:
                width = when - active
                if width > 0 and math.isfinite(width):
                    widths.append(float(width))
                active = None
        return widths

    def pulse_width(self, x, level, polarity="high", nth=1, start=None, end=None):
        """Return the ``nth`` complete high or low pulse width in the window."""
        polarity = self._polarity(polarity)
        nth = self._nth(nth)
        widths = self._complete_pulses(x, level, polarity, start, end)
        if len(widths) < nth:
            raise ValueError("fewer than %d complete %s pulse(s)" % (nth, polarity))
        return widths[nth - 1]

    def period(self, x, level, edge="rising", start=None, end=None):
        """Return the mean interval between same-direction threshold crossings."""
        periods = self._periods(x, level, edge, start, end)
        return float(np.mean(periods))

    def frequency(self, x, level, edge="rising", start=None, end=None):
        """Return the reciprocal of :meth:`period`."""
        value = self.period(x, level, edge, start, end)
        if value <= 0 or not math.isfinite(value):
            raise ValueError("period must be finite and positive")
        return float(1.0 / value)

    def _complete_cycles(self, x, level, start=None, end=None):
        rising = [float(t) for t in
                   self._cross_times(x, level, "rising", start, end)]
        falling = np.asarray(
            [float(t) for t in
             self._cross_times(x, level, "falling", start, end)],
            dtype=float,
        )
        if len(rising) < 2:
            raise ValueError("at least two rising crossings are required")
        if not np.isfinite(falling).all() or (falling.size > 1 and
                                              not np.all(np.diff(falling) > 0)):
            raise ValueError("falling crossing times must be finite and increasing")
        cycles = []
        for first, second in zip(rising[:-1], rising[1:]):
            if second <= first:
                raise ValueError("rising crossing times must be increasing")
            lo = int(np.searchsorted(falling, first, side="right"))
            hi = int(np.searchsorted(falling, second, side="left"))
            if hi - lo != 1:
                raise ValueError(
                    "each rising-to-rising cycle must contain exactly one falling crossing"
                )
            high_time = float(falling[lo] - first)
            if high_time <= 0 or not math.isfinite(high_time):
                raise ValueError("complete cycle high duration must be positive")
            cycles.append((high_time, second - first))
        return cycles

    def duty_cycle(self, x, level, start=None, end=None):
        """Return high-level time divided by period time over complete cycles."""
        cycles = self._complete_cycles(x, level, start, end)
        high = sum(item[0] for item in cycles)
        total = sum(item[1] for item in cycles)
        if total <= 0 or not math.isfinite(total):
            raise ValueError("complete cycle duration must be finite and positive")
        return float(high / total)

    def period_jitter(self, x, level, edge="rising", start=None, end=None):
        """Return population standard deviation of same-edge periods."""
        periods = self._periods(x, level, edge, start, end)
        if periods.size < 2:
            raise ValueError("at least three crossings are required for period jitter")
        return float(np.std(periods, ddof=0))

    def cycle_jitter(self, x, level, edge="rising", start=None, end=None):
        """Return RMS difference between adjacent same-edge periods."""
        periods = self._periods(x, level, edge, start, end)
        if periods.size < 2:
            raise ValueError("at least three crossings are required for cycle jitter")
        changes = np.diff(periods)
        return float(np.sqrt(np.mean(np.square(changes))))

    def slope(self, x, start=None, end=None):
        """Fit a time-weighted line to the piecewise-linear waveform.

        The fit integrates each linear segment over time.  Normalizing time to
        ``[0, 1]`` before forming moments avoids loss of precision for axes in
        nanoseconds or picoseconds, then converts the fitted slope back to
        waveform-units per second.
        """
        ts, xs = self._window_data(x, start, end)
        ts = np.asarray(ts, dtype=float)
        xs = np.asarray(xs, dtype=float)
        if (ts.ndim != 1 or xs.ndim != 1 or ts.size != xs.size or
                ts.size < 2 or not np.isfinite(ts).all() or
                not np.isfinite(xs).all() or not np.all(np.diff(ts) > 0)):
            raise ValueError("slope requires finite, increasing waveform samples")
        span = float(ts[-1] - ts[0])
        if not math.isfinite(span) or span <= 0:
            raise ValueError("slope window must have positive duration")
        u = (ts - ts[0]) / span
        v = u - .5
        z = xs - xs[0]
        du = np.diff(u)
        v0, v1 = v[:-1], v[1:]
        z0, z1 = z[:-1], z[1:]
        # The time-weighted domain is u in [0, 1].  Since v=u-.5 has
        # variance 1/12, the continuous least-squares slope is
        # 12 * integral(v*z du) / span.  Subtracting the first sample keeps
        # a large DC offset out of the covariance calculation.
        integral_vz = float(np.sum(
            du * (v0 * (2.0 * z0 + z1) + v1 * (z0 + 2.0 * z1)) / 6.0
        ))
        fitted = 12.0 * integral_vz / span
        if not math.isfinite(fitted):
            raise ValueError("slope result is not finite")
        return float(fitted)

    def value_at_cross(self, x, trigger, level, edge="rising", nth=1,
                       start=None, end=None):
        """Return ``x`` interpolated at the ``nth`` trigger crossing."""
        nth = self._nth(nth)
        # Validate both signals against the selected window.  In particular,
        # do not let a valid trigger mask a missing or malformed value signal.
        self._window_data(x, start, end)
        crossings = self._cross_times(trigger, level, edge, start, end)
        if len(crossings) < nth:
            raise ValueError("fewer than %d %s trigger crossing(s)" %
                             (nth, edge))
        when = float(crossings[nth - 1])
        value = float(self.at(x, when))
        if not math.isfinite(value):
            raise ValueError("value at crossing is not finite")
        return value


__all__ = ["BlockMeasurements"]
