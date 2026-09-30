import copy
import math
import unittest

import numpy as np

from optserver.metric_catalog import metric_catalog
from optserver.metricexpr import validate_metric_definitions
from optserver.metrics import Waves, compute_metrics, measurement_signature
from optserver.netlist import parse_num


def _us(values):
    return np.asarray(values, dtype=float) * 1e-6


def _tran_fixture():
    """Return a dense, piecewise-linear fixture for every transient example."""
    tu = np.linspace(0.0, 10.0, 1001)
    out_settle = np.clip(tu - 1.0, 0.0, 1.0)
    out_fall = 1.0 - np.clip(tu - 1.0, 0.0, 1.0)
    out_overshoot = np.interp(
        tu, [0.0, 1.0, 2.0, 2.5, 3.0, 10.0],
        [0.0, 0.0, 1.0, 1.2, 1.0, 1.0])
    out_ripple = np.interp(
        tu, [0.0, 9.0, 9.25, 9.75, 10.0],
        [0.0, 0.0, 2.0, 0.0, 0.0])
    out_noise = np.interp(tu, [0.0, 9.0, 10.0], [0.0, 0.0, 2.0])
    out_droop = np.interp(tu, [0.0, 9.0, 10.0], [0.0, 1.0, 0.5])
    out_delay = np.interp(tu, [0.0, 3.0, 5.0, 10.0], [0.0, 0.0, 1.0, 1.0])
    out_square = np.zeros_like(tu)
    for lo, hi in ((2.0, 4.0), (5.0, 7.0), (8.0, 10.0)):
        out_square[(tu >= lo) & (tu < hi)] = 1.0
    input_ramp = 0.25 + 0.125 * tu
    tracking_out = input_ramp + 0.5
    current = -np.interp(tu, [0.0, 5.0, 10.0], [1e-3, 3e-3, 2e-3])
    return _us(tu), {
        "out_settle": out_settle,
        "out_fall": out_fall,
        "out_overshoot": out_overshoot,
        "out_ripple": out_ripple,
        "out_noise": out_noise,
        "out_droop": out_droop,
        "out_delay": out_delay,
        "out_square": out_square,
        "in": input_ramp,
        "tracking_out": tracking_out,
        "vdd": np.full_like(tu, 1.2),
        "VDD:p": current,
    }


def catalog_cases():
    """Yield ``(catalog_entry, axis, signals, expected)`` for every template."""
    t, fixture = _tran_fixture()
    tran_cases = {
        "settling_s": (fixture["out_settle"], 0.99e-6),
        "settling_abs_s": (fixture["out_settle"], 0.999e-6),
        "settled_ok": (fixture["out_settle"], 1.0),
        "settling_error_v": (fixture["out_settle"], 0.0),
        "rise_s": (fixture["out_settle"], 0.8e-6),
        "fall_s": (fixture["out_fall"], 0.8e-6),
        "delay_s": (fixture["out_delay"], 2.0e-6),
        "slew_v_s": (fixture["out_settle"], 1.0e6),
        "overshoot_ratio": (fixture["out_overshoot"], 0.2),
        "ripple_vpp": (fixture["out_ripple"], 2.0),
        "noise_rms_v": (fixture["out_noise"], 1.0 / math.sqrt(3.0)),
        "power_w": (fixture["vdd"], 2.84e-3),
        "energy_j": (fixture["vdd"], 1.92e-9),
        "pulse_width_s": (fixture["out_square"], 2.0e-6),
        "period_s": (fixture["out_square"], 3.0e-6),
        "frequency_hz": (fixture["out_square"], 1.0 / 3.0e-6),
        "duty_ratio": (fixture["out_square"], 2.0 / 3.0),
        "period_jitter_s": (fixture["out_square"], 0.0),
        "cycle_jitter_s": (fixture["out_square"], 0.0),
        "droop_v_s": (fixture["out_droop"], -0.5e6),
        "trigger_value_v": (fixture["out_delay"], 0.75),
        "tracking_rms_v": (fixture["tracking_out"], 0.5),
        "tracking_peak_v": (fixture["tracking_out"], 0.5),
        "peak_current_a": (fixture["vdd"], 3.0e-3),
        "charge_c": (fixture["vdd"], 1.6e-9),
    }
    signal_names = {
        "settling_s": ("out_settle",),
        "settling_abs_s": ("out_settle",),
        "settled_ok": ("out_settle",),
        "settling_error_v": ("out_settle",),
        "rise_s": ("out_settle",),
        "fall_s": ("out_fall",),
        "delay_s": ("in", "out_delay"),
        "slew_v_s": ("out_settle",),
        "overshoot_ratio": ("out_overshoot",),
        "ripple_vpp": ("out_ripple",),
        "noise_rms_v": ("out_noise",),
        "power_w": ("vdd", "VDD:p"),
        "energy_j": ("vdd", "VDD:p"),
        "pulse_width_s": ("out_square",),
        "period_s": ("out_square",),
        "frequency_hz": ("out_square",),
        "duty_ratio": ("out_square",),
        "period_jitter_s": ("out_square",),
        "cycle_jitter_s": ("out_square",),
        "droop_v_s": ("out_droop",),
        "trigger_value_v": ("in", "out_delay"),
        "tracking_rms_v": ("in", "tracking_out"),
        "tracking_peak_v": ("in", "tracking_out"),
        "peak_current_a": ("VDD:p",),
        "charge_c": ("VDD:p",),
    }
    dc_axis = np.asarray([-1.0, 0.0, 1.0])
    dc_signals = {"in": dc_axis, "out": 2.0 * dc_axis + 0.3}
    dc_expected = {"dc_gain_v_v": 2.0,
                   "dc_gain_db": 20.0 * math.log10(2.0),
                   "dc_offset_v": 0.3}
    for entry in metric_catalog():
        name = entry["name"]
        if entry["analysis"] == "dc":
            yield copy.deepcopy(entry), dc_axis.copy(), {
                key: value.copy() for key, value in dc_signals.items()
            }, dc_expected[name]
            continue
        selected = {key: fixture[key].copy() for key in signal_names[name]}
        # Catalog signal names are deliberately independent of fixture names.
        rename = {"out_settle": "out", "out_fall": "out",
                  "out_overshoot": "out", "out_ripple": "out",
                  "out_noise": "out", "out_droop": "out",
                  "out_delay": "out", "out_square": "out",
                  "tracking_out": "out"}
        selected = {rename.get(key, key): value for key, value in selected.items()}
        yield copy.deepcopy(entry), t.copy(), selected, tran_cases[name][1]


def _metric_result(entry, axis, signals):
    definition = {
        "expr": entry["expr"],
        "analysis": entry.get("analysis", "tran"),
    }
    if "window" in entry:
        definition["window"] = copy.deepcopy(entry["window"])
    return compute_metrics(
        {entry["name"]: definition},
        datasets={entry["analysis"]: (axis, signals)},
    )[entry["name"]]


class CatalogWindowTests(unittest.TestCase):
    def test_every_catalog_expression_has_an_analytic_fixture(self):
        cases = list(catalog_cases())
        self.assertEqual(len(cases), len(metric_catalog()))
        for entry, axis, signals, expected in cases:
            with self.subTest(metric=entry["name"]):
                actual = _metric_result(entry, axis, signals)
                self.assertAlmostEqual(actual, expected,
                                       delta=max(1e-12, abs(expected) * 2e-8))

    def test_transient_catalog_windows_reject_outside_pollution_and_shift(self):
        for entry, axis, signals, expected in catalog_cases():
            if entry["analysis"] != "tran":
                continue
            window = entry["window"]
            start = parse_num(window["start"])
            end = parse_num(window["end"])
            before = axis[0] - 2e-6
            after = axis[-1] + 2e-6
            polluted_axis = np.concatenate(([before], axis, [after]))
            before_window = axis < start
            after_window = axis > end
            inside_window = ~(before_window | after_window)
            polluted_signals = {}
            for name, values in signals.items():
                polluted = values.copy()
                polluted[before_window] = 12345.0
                polluted[after_window] = -98765.0
                # Preserve every sample on either measurement boundary while
                # making already-recorded startup/tail samples conspicuous.
                np.testing.assert_array_equal(polluted[inside_window],
                                              values[inside_window])
                polluted_signals[name] = np.concatenate((
                    [12345.0], polluted, [-98765.0]))
            with self.subTest(metric=entry["name"], case="pollution"):
                actual = _metric_result(entry, polluted_axis, polluted_signals)
                self.assertAlmostEqual(actual, expected,
                                       delta=max(1e-12, abs(expected) * 2e-8))

            delta = 17e-6
            shifted = copy.deepcopy(entry)
            shifted["window"] = {
                key: parse_num(value) + delta
                for key, value in window.items()
            }
            with self.subTest(metric=entry["name"], case="shift"):
                actual = _metric_result(shifted, axis + delta, signals)
                self.assertAlmostEqual(actual, expected,
                                       delta=max(1e-12, abs(expected) * 2e-8))

    def test_window_start_end_interpolate_and_support_one_sided_bounds(self):
        t = _us([0.0, 2.0, 4.0])
        y = 2.0 * np.asarray([0.0, 2.0, 4.0]) + 1.0
        definition = {"analysis": "tran", "expr": "avg(V('out'))",
                      "window": {"start": "1u", "end": "3u"}}
        self.assertAlmostEqual(
            _metric_result({"name": "avg", **definition}, t, {"out": y}),
            5.0)
        one_sided = {
            "start_only": {"analysis": "tran", "expr": "avg(V('out'))",
                            "window": {"start": "1u"}},
            "end_only": {"analysis": "tran", "expr": "avg(V('out'))",
                          "window": {"end": "3u"}},
        }
        validate_metric_definitions(one_sided, {"tran": {"stop": "4u"}})
        result = compute_metrics(one_sided, datasets={"tran": (t, {"out": y})})
        self.assertAlmostEqual(result["start_only"], 6.0)
        self.assertAlmostEqual(result["end_only"], 4.0)

    def test_np_axis_and_sample_see_the_cropped_window(self):
        t = _us([0, 2, 4, 6, 8, 10])
        signals = {"out": t.copy()}
        defs = {
            "span": {"analysis": "tran",
                      "expr": "float(np.max(t) - np.min(axis))",
                      "window": {"start": "2u", "end": "8u"}},
            "first_sample": {"analysis": "tran",
                              "expr": "sample(V('out'), fs=1M, nsamp=2)[0]",
                              "window": {"start": "5u", "end": "8u"}},
        }
        result = compute_metrics(defs, datasets={"tran": (t, signals)})
        self.assertAlmostEqual(result["span"], 6e-6)
        self.assertAlmostEqual(result["first_sample"], 5e-6)

    def test_metric_dependencies_keep_their_own_windows(self):
        t = _us(np.arange(0.0, 11.0))
        signals = {"out": np.arange(0.0, 11.0)}
        defs = {
            "early": {"analysis": "tran", "expr": "avg(V('out'))",
                       "window": {"start": "1u", "end": "3u"}},
            "late": {"analysis": "tran", "expr": "avg(V('out'))",
                      "window": {"start": "8u", "end": "10u"}},
            "total": {"analysis": "tran", "expr": "m['early'] + m['late']"},
        }
        result = compute_metrics(defs, datasets={"tran": (t, signals)})
        self.assertAlmostEqual(result["early"], 2.0)
        self.assertAlmostEqual(result["late"], 9.0)
        self.assertAlmostEqual(result["total"], 11.0)

    def test_explicit_helper_ranges_outside_waveform_are_rejected(self):
        t = _us([0.0, 1.0, 2.0])
        wave = Waves(t, {"out": np.asarray([0.0, 0.0, 1.0])})
        with self.assertRaises(ValueError):
            wave.avg(wave.V("out"), t0=-1e-6, t1=1e-6)
        with self.assertRaises(ValueError):
            wave.cross(wave.V("out"), level=.5, start=0.5e-6, end=3e-6)

        row_window = {"start": "1u", "end": "2u"}
        bad_helpers = {
            "avg_bad": "avg(V('out'), t0=0, t1=2u)",
            "cross_bad": "cross(V('out'), level=.5, start=0)",
            "at_bad": "at(V('out'), 0)",
        }
        for name, expr in bad_helpers.items():
            with self.subTest(helper=name):
                with self.assertRaisesRegex(RuntimeError, "metric %r" % name):
                    compute_metrics(
                        {name: {"analysis": "tran", "expr": expr,
                                "window": row_window}},
                        datasets={"tran": (t, {"out": np.asarray([0.0, 0.0, 1.0])})},
                    )

        good = compute_metrics(
            {"avg_ok": {"analysis": "tran",
                         "expr": "avg(V('out'), t0=1u, t1=2u)",
                         "window": row_window},
             "cross_ok": {"analysis": "tran",
                           "expr": "cross(V('out'), level=.5, start=1u, end=2u)",
                           "window": row_window},
             "at_ok": {"analysis": "tran", "expr": "at(V('out'), 1u)",
                        "window": row_window}},
            datasets={"tran": (t, {"out": np.asarray([0.0, 0.0, 1.0])})},
        )
        self.assertAlmostEqual(good["avg_ok"], .5)
        self.assertAlmostEqual(good["cross_ok"], 1.5e-6)
        self.assertAlmostEqual(good["at_ok"], 0.0)

    def test_dc_start_end_are_scan_ranges_and_dc_windows_are_rejected(self):
        axis = np.asarray([-1.0, 0.0, 1.0])
        signals = {"in": axis, "out": 3.0 * axis + 2.0}
        definition = {"name": "gain", "analysis": "dc",
                      "expr": "dc_gain(V('out'), V('in'), start=-.5, end=.5)"}
        self.assertAlmostEqual(_metric_result(definition, axis, signals), 3.0)
        with self.assertRaisesRegex(ValueError, "time window requires tran"):
            validate_metric_definitions(
                {"bad": {"analysis": "dc", "expr": "dc_gain(V('out'), V('in'))",
                          "window": {"start": "0", "end": "1"}}},
                {"dc": {"source": "VIN", "start": -1,
                         "stop": 1, "step": 1}})

    def test_invalid_windows_are_preflight_errors(self):
        analyses = {"tran": {"stop": "10u"},
                    "dc": {"source": "VIN", "start": -1,
                           "stop": 1, "step": 1}}
        invalid = (
            {"start": "-1u", "end": "2u"},
            {"start": "nan", "end": "2u"},
            {"start": "3u", "end": "2u"},
            {},
            {"start": "1u", "middle": "2u"},
            {"start": "1u", "end": "11u"},
            {"start": True, "end": "2u"},
            {"start": "2u", "end": "2u"},
            {"start": None, "end": "2u"},
            {"start": "10u"},
            {"start": "9u", "end": "10.1u"},
        )
        for window in invalid:
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    validate_metric_definitions(
                        {"m": {"analysis": "tran", "expr": "avg(V('out'))",
                                "window": window}}, analyses)

        with self.assertRaisesRegex(RuntimeError, "metric 'm'"):
            compute_metrics(
                {"m": {"analysis": "tran", "expr": "avg(V('out'))",
                        "window": {"start": "1u", "end": "3u"}}},
                t=_us([0.0, 1.0, 2.0]),
                sigs={"out": np.asarray([0.0, 1.0, 2.0])},
            )

    def test_old_strings_and_windowless_dictionaries_remain_compatible(self):
        analyses = {"tran": {"stop": "10u"}}
        validate_metric_definitions({"old": "avg(V('out'))"}, analyses)
        validate_metric_definitions(
            {"new": {"analysis": "tran", "expr": "avg(V('out'))"}},
            analyses)
        t = _us([0, 1, 2])
        result = compute_metrics(
            {"old": "avg(V('out'))",
             "new": {"analysis": "tran", "expr": "avg(V('out'))"}},
            datasets={"tran": (t, {"out": np.asarray([1.0, 2.0, 3.0])})})
        self.assertAlmostEqual(result["old"], 2.0)
        self.assertAlmostEqual(result["new"], 2.0)

    def test_window_changes_measurement_signature(self):
        base = {"m": {"analysis": "tran", "expr": "avg(V('out'))",
                       "window": {"start": "1u", "end": "2u"}}}
        changed = copy.deepcopy(base)
        changed["m"]["window"]["start"] = "1.5u"
        analyses = {"tran": {"stop": "10u"}}
        first = measurement_signature(base, analyses, save=["out"])
        second = measurement_signature(changed, analyses, save=["out"])
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
