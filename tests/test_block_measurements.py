import unittest

import numpy as np

from optserver.block_measurements import BlockMeasurements
from optserver.measurements import WaveMeasurements


class Wave(BlockMeasurements, WaveMeasurements):
    """Minimal waveform owner exposing the parent mixin contract."""

    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def _window(self, x, t0=None, t1=None, min_points=2):
        axis = self._axis()
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.size != axis.size or not np.isfinite(values).all():
            raise ValueError("waveform must match the axis")
        lo = axis[0] if t0 is None else self._finite(t0, "start")
        hi = axis[-1] if t1 is None else self._finite(t1, "end")
        if lo < axis[0] or hi > axis[-1] or lo >= hi:
            raise ValueError("window is outside the waveform")
        points = np.concatenate((
            np.asarray([lo]), axis[(axis > lo) & (axis < hi)], np.asarray([hi])
        ))
        if points.size < min_points:
            raise ValueError("window has too few samples")
        return points, np.interp(points, axis, values)

    def at(self, x, tq):
        axis = self._axis()
        value = np.asarray(x, dtype=float)
        if value.ndim != 1 or value.size != axis.size or not np.isfinite(value).all():
            raise ValueError("waveform must match the axis")
        tq = self._finite(tq, "time")
        if tq < axis[0] or tq > axis[-1]:
            raise ValueError("time is outside the waveform")
        return float(np.interp(tq, axis, value))


def threshold_wave(vertices, stop, step=.01):
    t = np.arange(0.0, stop + step / 2.0, step)
    vt, vx = zip(*vertices)
    return t, np.interp(t, np.asarray(vt), np.asarray(vx))


class BlockMeasurementTests(unittest.TestCase):
    def test_pulse_width_skips_window_half_pulses_and_supports_polarity(self):
        vertices = [
            (0.0, 0.0), (1.25, .5), (2.0, 1.0), (3.75, .5),
            (4.5, 0.0), (5.25, .5), (6.0, 1.0), (7.75, .5),
            (8.5, 0.0), (9.25, .5), (10.0, 1.0), (11.75, .5), (12.0, 0.0),
        ]
        t, x = threshold_wave(vertices, 12.0)
        wave = Wave(t)

        self.assertAlmostEqual(wave.pulse_width(x, .5), 2.5, places=12)
        self.assertAlmostEqual(wave.pulse_width(x, .5, nth=3), 2.5, places=12)
        self.assertAlmostEqual(
            wave.pulse_width(x, .5, start=1.537, end=10.543), 2.5, places=12
        )
        self.assertAlmostEqual(
            wave.pulse_width(x, .5, polarity="low", nth=2), 1.5, places=12
        )
        with self.assertRaises(ValueError):
            wave.pulse_width(x, .5, start=2.0, end=10.0, nth=2)

    def test_period_frequency_and_duty_cycle_use_complete_edges(self):
        vertices = [
            (0.0, 0.0), (1.25, .5), (2.0, 1.0), (3.75, .5),
            (4.5, 0.0), (5.25, .5), (6.0, 1.0), (7.75, .5),
            (8.5, 0.0), (9.25, .5), (10.0, 1.0), (11.75, .5), (12.0, 0.0),
        ]
        t, x = threshold_wave(vertices, 12.0)
        wave = Wave(t)
        self.assertAlmostEqual(wave.period(x, .5), 4.0, places=12)
        self.assertAlmostEqual(wave.period(x, .5, edge="falling"), 4.0, places=12)
        self.assertAlmostEqual(wave.frequency(x, .5), .25, places=12)
        self.assertAlmostEqual(wave.duty_cycle(x, .5), 2.5 / 4.0, places=12)
        self.assertAlmostEqual(
            wave.duty_cycle(x, .5, start=2.0, end=10.0), 2.5 / 4.0, places=12
        )

    def test_jitter_definitions_use_population_period_and_adjacent_rms(self):
        vertices = [
            (0.0, 0.0), (1.0, .5), (1.5, 1.0), (2.0, .5), (2.5, 0.0),
            (4.0, .5), (4.5, 1.0), (5.0, .5), (5.5, 0.0),
            (7.5, .5), (8.0, 1.0), (8.5, .5), (9.0, 0.0),
        ]
        t, x = threshold_wave(vertices, 9.0)
        wave = Wave(t)
        self.assertAlmostEqual(wave.period(x, .5), 3.25, places=12)
        self.assertAlmostEqual(wave.period_jitter(x, .5), .25, places=12)
        self.assertAlmostEqual(wave.cycle_jitter(x, .5), .5, places=12)

        with self.assertRaises(ValueError):
            wave.period_jitter(x, .5, start=0.0, end=5.0)
        with self.assertRaises(ValueError):
            wave.cycle_jitter(x, .5, start=0.0, end=5.0)

    def test_slope_is_time_weighted_and_stable_under_time_translation(self):
        # The piecewise-linear waveform is flat for one time unit and rises
        # from 0 to 2 over the next two.  Continuous-time least squares gives
        # 20/27; an unweighted fit to the three samples gives a different value.
        t = np.array([1.0e-6, 1.0e-6 + 1.0e-12, 1.0e-6 + 3.0e-12])
        x = np.array([0.0, 0.0, 2.0])
        wave = Wave(t)
        expected = (20.0 / 27.0) / (1.0e-12)
        self.assertAlmostEqual(wave.slope(x), expected, delta=abs(expected) * 1e-10)

        shifted = Wave(t + 2.0e-9)
        self.assertAlmostEqual(shifted.slope(x), expected, delta=abs(expected) * 1e-10)

    def test_slope_is_zero_for_constant_large_dc_and_keeps_small_droop(self):
        t = 1.0e-6 + np.array([0.0, 1.0e-12, 3.0e-12, 4.0e-12])
        constant = np.full(t.size, 1.0e12)
        self.assertAlmostEqual(Wave(t).slope(constant), 0.0, places=12)

        span = t[-1] - t[0]
        droop = 1.0e12 + 0.01 * (1.0 - (t - t[0]) / span)
        expected = -0.01 / span
        self.assertAlmostEqual(
            Wave(t).slope(droop), expected, delta=abs(expected) * 1e-2
        )

    def test_slope_is_unchanged_by_redundant_piecewise_linear_points(self):
        base_t = np.array([0.0, 1.0, 3.0, 4.0])
        base_x = np.array([0.0, 0.0, 2.0, 2.0])
        extra_t = np.array([0.0, .5, 1.0, 2.0, 3.0, 4.0])
        extra_x = np.interp(extra_t, base_t, base_x)
        self.assertAlmostEqual(Wave(base_t).slope(base_x),
                               Wave(extra_t).slope(extra_x), places=12)

    def test_value_at_cross_interpolates_signal_and_respects_window(self):
        vertices = [
            (0.0, 0.0), (1.25, .5), (2.0, 1.0), (3.75, .5),
            (4.5, 0.0), (5.25, .5), (6.0, 1.0), (7.75, .5),
            (8.5, 0.0), (9.25, .5), (10.0, 1.0), (11.75, .5), (12.0, 0.0),
        ]
        t, trigger = threshold_wave(vertices, 12.0)
        signal = 2.0 * t + 1.0
        wave = Wave(t)
        self.assertAlmostEqual(wave.value_at_cross(signal, trigger, .5), 3.5, places=12)
        self.assertAlmostEqual(
            wave.value_at_cross(signal, trigger, .5, nth=1, start=2.0, end=10.0),
            11.5, places=12
        )
        with self.assertRaises(ValueError):
            wave.value_at_cross(signal[:-1], trigger, .5)
        with self.assertRaises(ValueError):
            wave.value_at_cross(signal, trigger, .5, start=2.0, end=2.5)

    def test_missing_edges_and_bad_arguments_are_errors(self):
        t = np.linspace(0.0, 2.0, 201)
        x = t / 2.0
        wave = Wave(t)
        with self.assertRaises(ValueError):
            wave.pulse_width(x, .5, polarity="middle")
        with self.assertRaises(ValueError):
            wave.pulse_width(x, .5, nth=0)
        with self.assertRaises(ValueError):
            wave.period(x, .5)
        with self.assertRaises(ValueError):
            wave.frequency(x, .5, edge="sideways")
        with self.assertRaises(ValueError):
            wave.duty_cycle(x, .5)
        with self.assertRaises(ValueError):
            wave.slope(x, start=1.0, end=1.0)


if __name__ == "__main__":
    unittest.main()
