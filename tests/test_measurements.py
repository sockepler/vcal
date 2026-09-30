import unittest

import numpy as np

from optserver.measurements import WaveMeasurements


class Wave(WaveMeasurements):
    """Small stand-in for the waveform owner used by the production mixin."""

    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def _window(self, x, t0=None, t1=None, min_points=2):
        axis = np.asarray(self.t, dtype=float)
        values = np.asarray(x, dtype=float)
        if axis.ndim != 1 or values.ndim != 1 or len(axis) != len(values):
            raise ValueError("invalid waveform")
        lo = axis[0] if t0 is None else float(t0)
        hi = axis[-1] if t1 is None else float(t1)
        if not np.isfinite([lo, hi]).all() or lo < axis[0] or hi > axis[-1] or lo >= hi:
            raise ValueError("window is outside the waveform")
        points = [lo]
        points.extend(axis[(axis > lo) & (axis < hi)].tolist())
        points.append(hi)
        points = np.asarray(points, dtype=float)
        if points.size < min_points:
            raise ValueError("window has too few points")
        return points, np.interp(points, axis, values)

    def at(self, x, tq):
        tq = float(tq)
        if tq < self.t[0] or tq > self.t[-1]:
            raise ValueError("time is outside the waveform")
        return float(np.interp(tq, self.t, np.asarray(x, dtype=float)))


class WaveMeasurementsTests(unittest.TestCase):
    def test_cross_linear_and_plateau_crossings_are_unique(self):
        w = Wave(np.arange(7, dtype=float))
        self.assertAlmostEqual(w.cross([0, 1, 2, 2, 3, 1, 0], 2), 2.0)
        self.assertAlmostEqual(
            w.cross([0, 2, 0, 2, 0, 2, 0], 1, nth=2), 2.5)
        self.assertAlmostEqual(
            w.cross([3, 2, 2, 1, 0, 0, 0], 2, edge="falling"), 1.0)

    def test_rise_fall_delay_and_slew(self):
        t = np.linspace(0.0, 10.0, 101)
        rising = t / 10.0
        falling = 1.0 - rising
        delayed = np.clip((t - 2.0) / 10.0, 0.0, 1.0)
        w = Wave(t)
        self.assertAlmostEqual(w.rise_time(rising), 8.0, places=12)
        self.assertAlmostEqual(w.fall_time(falling), 8.0, places=12)
        self.assertAlmostEqual(w.delay(rising, delayed, .5, .5), 2.0, places=12)
        self.assertAlmostEqual(w.slew_rate(rising), .1, places=12)

    def test_rc_settling_uses_linear_entry_time(self):
        t = np.linspace(0.0, 10.0, 1001)
        x = 1.0 - np.exp(-t)
        w = Wave(t)
        value = w.settle_time(x, final=1.0, initial=0.0, tol=.01)
        self.assertGreater(value, 4.59)
        self.assertLess(value, 4.62)
        self.assertEqual(w.settled(x, final=1.0, initial=0.0, tol=.01), 1)

    def test_ringing_uses_last_out_of_band_crossing(self):
        t = np.linspace(0.0, 20.0, 2001)
        x = (1.0 - np.exp(-t / 1.2)
             + .12 * np.exp(-t / 4.0) * np.sin(2.0 * np.pi * .5 * t))
        w = Wave(t)
        value = w.settle_time(x, final=1.0, initial=0.0, tol=.02)
        self.assertGreater(value, 7.4)
        self.assertLess(value, 7.7)
        self.assertEqual(w.settled(x, final=1.0, initial=0.0, tol=.02), 1)

    def test_settle_entry_interpolates_the_actual_band_boundary(self):
        w = Wave(np.arange(4.0))
        x = np.array([0.0, 1.2, .95, 1.0])
        self.assertAlmostEqual(
            w.settle_time(x, final=1.0, initial=0.0, tol=.1), 1.4)

    def test_unsettled_window_mode_and_settled_flag(self):
        t = np.linspace(0.0, 5.0, 51)
        x = .1 * t
        w = Wave(t)
        with self.assertRaises(ValueError):
            w.settle_time(x, final=1.0, initial=0.0, tol=.01)
        self.assertEqual(w.settle_time(x, final=1.0, initial=0.0,
                                       tol=.01, on_unsettled="window"), 5.0)
        self.assertEqual(w.settled(x, final=1.0, initial=0.0, tol=.01), 0)
        self.assertEqual(w.settled(x, final=1.0, initial=0.0,
                                    tol=.01, on_unsettled="window"), 0)

    def test_hold_requires_remaining_window(self):
        t = np.arange(7.0)
        x = np.array([0.0, .5, .95, 1.0, 1.0, 1.0, 1.0])
        w = Wave(t)
        value = w.settle_time(x, final=1.0, initial=0.0, tol=.1, hold=3.0)
        self.assertGreater(value, 1.7)
        self.assertLess(value, 1.9)
        with self.assertRaises(ValueError):
            w.settle_time(x, final=1.0, initial=0.0, tol=.1, hold=6.0)
        self.assertEqual(w.settle_time(x, final=1.0, initial=0.0, tol=.1,
                                       hold=6.0, on_unsettled="window"), 6.0)

    def test_absolute_tolerance_supports_small_or_zero_steps(self):
        t = np.arange(5.0)
        x = np.array([1000.0, 1000.0002, 1000.0007, 1000.0006, 1000.0004])
        w = Wave(t)
        value = w.settle_time(x, final=1000.0, initial=1000.0,
                               tol=0.0, atol=.0005)
        self.assertGreater(value, 3.0)
        self.assertLess(value, 4.0)

    def test_downward_overshoot_and_settling_error(self):
        w = Wave(np.arange(6.0))
        x = np.array([1.0, .5, 0.0, -.2, .05, 0.0])
        self.assertAlmostEqual(w.overshoot(x, initial=1.0, final=0.0), .2)
        self.assertAlmostEqual(w.settling_error(x, initial=1.0, final=0.0), 1.0)
        self.assertAlmostEqual(
            w.settling_error(x, initial=1.0, final=0.0, relative=True), 1.0)

    def test_dc_gain_offset_and_local_difference_quotient(self):
        t = np.arange(4.0)
        w = Wave(t)
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 5.0 - 2.0 * x
        self.assertAlmostEqual(w.dc_gain(y, x), -2.0)
        self.assertAlmostEqual(w.dc_gain(y, x, at=1.5), -2.0)
        self.assertAlmostEqual(w.dc_offset(y, x), 5.0)
        self.assertAlmostEqual(w.dc_gain(y), -2.0)

        x_down = x[::-1]
        y_down = 5.0 - 2.0 * x_down
        self.assertAlmostEqual(w.dc_gain(y_down, x_down), -2.0)
        self.assertAlmostEqual(w.dc_offset(y_down, x_down), 5.0)

    def test_invalid_direction_window_and_dc_inputs_are_rejected(self):
        w = Wave(np.arange(4.0))
        with self.assertRaises(ValueError):
            w.cross([0, 1, 2, 3], 1, edge="sideways")
        with self.assertRaises(ValueError):
            w.cross([0, 1, 2, 3], 1, nth=0)
        with self.assertRaises(ValueError):
            w.rise_time([1, .7, .3, 0])
        with self.assertRaises(ValueError):
            w.settle_time([0, .5, 1, 1], start=-1, final=1)
        with self.assertRaises(ValueError):
            w.dc_gain([0, 1, 2, 3], [0, 2, 1, 3])
        with self.assertRaises(ValueError):
            w.dc_gain([0, 1, 2, 3], [1, 1, 1, 1])
        with self.assertRaises(ValueError):
            w.dc_gain([0, 1, 2, 3], [0, 1, 2, 3], at=4)


if __name__ == "__main__":
    unittest.main()
