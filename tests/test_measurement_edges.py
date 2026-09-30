"""Regression cases where a measurement window cuts through earlier events."""
import unittest

import numpy as np

from optserver.metrics import Waves, compute_metrics


class MeasurementEdgeTests(unittest.TestCase):
    def test_exact_right_boundary_and_plateau_arrival(self):
        wave = Waves([0, 1, 2, 3], {})
        self.assertEqual(wave.cross([0, 0, 0, 1], 1), 3)
        self.assertEqual(wave.cross([0, 1, 1, 1], 1), 1)
        self.assertEqual(wave.cross([2, 1, 1, 1], 1, edge="falling"), 1)
        self.assertEqual(wave._cross_times([1, 1, 2, 2], 1), [0])
        self.assertEqual(wave._cross_times([0, 1, 1, 0], 1), [])
        self.assertEqual(wave._cross_times([1, 1, 1, 1], 1), [])

    def test_window_boundary_crossing_is_an_absolute_time(self):
        result = compute_metrics({"edge": {"expr": "cross(V('x'), .5)",
                                           "window": {"start": 4.5, "end": 7}}},
                                 np.arange(9.), {"x": np.arange(9.) - 4})
        self.assertEqual(result["edge"], 4.5)

    def test_causal_delay_ignores_previous_cycles_output(self):
        t = np.arange(13.)
        x = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0.])
        y = np.asarray([0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1.])
        wave = Waves(t, {})
        # Start between the first input and its response. Ordinal pairing
        # would associate output 2.5 with input 5.5 and incorrectly return -3.
        self.assertEqual(wave.delay(x, y, .5, .5, start=2), 1)
        self.assertEqual(wave.delay(x, y, .5, .5, start=2, nth=2), 1)
        self.assertEqual(wave.delay(x, y, .5, .5, start=2, pairing="ordinal"), -3)
        with self.assertRaisesRegex(ValueError, "causal"):
            wave.delay(x, y, .5, .5, start=2, end=6)

    def test_missing_response_cannot_borrow_the_next_cycle(self):
        t = np.arange(9.)
        x = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 1.])
        y = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 1.])
        wave = Waves(t, {})
        with self.assertRaisesRegex(ValueError, "causal"):
            wave.delay(x, y, .5, .5)
        self.assertEqual(wave.delay(x, y, .5, .5, nth=2), 1)
        self.assertEqual(wave.delay(x, x, .5, .5), 0)
        with self.assertRaisesRegex(ValueError, "pairing"):
            wave.delay(x, y, .5, .5, pairing="best")

    def test_incomplete_glitch_is_not_part_of_the_next_transition(self):
        t = np.arange(8.)
        rising = np.asarray([0, .2, 0, 0, .2, 1, 1, 1.])
        wave = Waves(t, {})
        # First .1 crossing is .5 but belongs to the aborted glitch.
        self.assertAlmostEqual(wave.rise_time(rising, initial=0, final=1), 1.375)
        self.assertAlmostEqual(wave.fall_time(1-rising, initial=1, final=0), 1.375)
        self.assertAlmostEqual(wave.slew_rate(rising, initial=0, final=1), .8/1.375)

    def test_late_start_uses_window_initial_level_for_settling(self):
        t = np.arange(8.)
        x = [9, -8, 4, 0, .9, 1, 1, 1]
        definition = {"expr": "settle_time(V('x'), final=1, tol=.1, hold=2)",
                      "window": {"start": 3, "end": 7}}
        self.assertAlmostEqual(compute_metrics({"s": definition}, t, {"x": x})["s"], 1)
        definition["window"]["end"] = 5
        with self.assertRaisesRegex(RuntimeError, "hold"):
            compute_metrics({"s": definition}, t, {"x": x})


if __name__ == "__main__":
    unittest.main()
