import unittest

import numpy as np

from optserver.adc_metrics import spectrum
from optserver.metricexpr import expand_literals, prepare_definitions
from optserver.metric_catalog import metric_catalog
from optserver.metrics import Waves, compute_metrics, measurement_signature


class MetricsIntegrationTests(unittest.TestCase):
    def test_catalog_expressions_preflight_for_the_declared_analysis(self):
        catalog = metric_catalog()
        self.assertEqual(len({entry["name"] for entry in catalog}), len(catalog))
        for entry in catalog:
            with self.subTest(entry=entry["name"]):
                prepare_definitions({entry["name"]: {
                    "expr": entry["expr"], "analysis": entry["analysis"]}},
                    analyses=(entry["analysis"],))

    def test_absolute_settling_band_overrides_relative_default(self):
        t = np.linspace(0, 12, 12001)
        x = 1 - np.exp(-t)
        wave = Waves(t, {"x": x})
        # 12-bit, 1-V full scale: half an LSB is far smaller than the
        # default 0.5% relative band and must not be widened by it.
        actual = wave.settle_time(x, final=1, initial=0, atol=1 / 8192)
        self.assertAlmostEqual(actual, np.log(8192), places=6)

    def test_complex_waveforms_are_not_silently_cast_to_real(self):
        with self.assertRaises(ValueError):
            Waves([0, 1], {"x": [1 + 2j, 3]})
        wave = Waves([0, 1], {"x": [1, 3]})
        with self.assertRaises(ValueError):
            wave.avg([1 + 2j, 3])

    def test_nonuniform_axis_integrates_linear_waveform(self):
        t = np.array([0.0, .2, 1.0, 1.7, 3.0])
        x = 2.0 * t + 1.0
        wave = Waves(t, {"x": x})

        # Piecewise-linear integration is exact for this linear waveform,
        # including the analytic RMS integral over [0, 3].
        self.assertAlmostEqual(wave.avg(x), 4.0, places=13)
        self.assertAlmostEqual(wave.rms(x), np.sqrt(19.0), places=13)
        self.assertAlmostEqual(wave.integ(x), 12.0, places=13)

    def test_window_boundaries_interpolate_and_sampling_does_not_extrapolate(self):
        t = np.array([0.0, 1.0, 3.0])
        x = 2.0 * t + 1.0
        wave = Waves(t, {"x": x})

        self.assertAlmostEqual(wave.avg(x, .5, 2.0), 3.5, places=13)
        self.assertAlmostEqual(wave.at(x, .5), 2.0, places=13)
        np.testing.assert_allclose(wave.sample(x, 1.0, 3, .5), [2.0, 4.0, 6.0])

        with self.assertRaises(ValueError):
            wave.at(x, -0.1)
        with self.assertRaises(ValueError):
            wave.at(x, 3.1)
        with self.assertRaises(ValueError):
            wave.sample(x, 1.0, 2, 2.5)

    @staticmethod
    def _phase_wave():
        fs = 1.0e6
        nsamp = 64
        fund = 5
        dense = 64
        t = np.arange(nsamp * dense + 1, dtype=float) / (fs * dense)
        # A non-coherent component makes phase-search candidates measurably
        # different while keeping the direct-spectrum comparison deterministic.
        x = (np.sin(2.0 * np.pi * 5.25 * fs / nsamp * t)
             + .17 * np.cos(2.0 * np.pi * 9.4 * fs / nsamp * t))
        return Waves(t, {"x": x}), x, fs, nsamp, fund, dense

    def test_fixed_fft_t0_matches_direct_spectrum(self):
        wave, x, fs, nsamp, fund, dense = self._phase_wave()
        t0 = 3.0 / (dense * fs)
        samples = wave.sample(x, fs, nsamp, t0)
        expected = spectrum(samples, fs, fund, harmonics=0)["sndr"]
        actual = wave.sndr(x, fs, fund, nsamp, t0=t0, harmonics=0)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_legacy_phase_search_warns_and_best_worst_select_candidates(self):
        wave, x, fs, nsamp, fund, dense = self._phase_wave()
        t_lo = 2.0 / (dense * fs)
        t_hi = 22.0 / (dense * fs)
        phase_times = np.linspace(t_lo, t_hi, 5)
        candidates = [
            spectrum(wave.sample(x, fs, nsamp, t0), fs, fund, harmonics=0)["sndr"]
            for t0 in phase_times
        ]

        with self.assertWarnsRegex(RuntimeWarning, "legacy FFT phase search"):
            legacy = wave.sndr(x, fs, fund, nsamp, t0_lo=t_lo, t0_hi=t_hi,
                               nphase=5, harmonics=0)
        self.assertAlmostEqual(legacy, max(candidates), places=12)
        self.assertAlmostEqual(
            wave.sndr(x, fs, fund, nsamp, t0_lo=t_lo, t0_hi=t_hi,
                      nphase=5, phase_mode="best", harmonics=0),
            max(candidates), places=12)
        self.assertAlmostEqual(
            wave.sndr(x, fs, fund, nsamp, t0_lo=t_lo, t0_hi=t_hi,
                      nphase=5, phase_mode="worst", harmonics=0),
            min(candidates), places=12)
        self.assertGreater(max(candidates), min(candidates))

    def test_engineering_suffix_does_not_rewrite_signal_names(self):
        expression = "V('5n') + .5n"
        expanded = expand_literals(expression)
        self.assertIn("'5n'", expanded)
        self.assertNotIn("'5e", expanded)

        result = compute_metrics(
            {"offset": "avg(V('5n')) + .5n"},
            t=np.array([0.0, 1.0]),
            sigs={"5n": np.array([2.0, 2.0])},
        )
        self.assertAlmostEqual(result["offset"], 2.0000000005, places=13)

    def test_metric_dependencies_sort_and_preflight_errors(self):
        definitions = {
            "derived": "m['base'] + 2",
            "base": "avg(V('x'))",
        }
        ordered = prepare_definitions(definitions, analyses=("tran",),
                                      default_analysis="tran")
        self.assertEqual(list(ordered), ["base", "derived"])
        result = compute_metrics(
            definitions, t=np.array([0.0, 1.0]), sigs={"x": np.array([2.0, 4.0])}
        )
        self.assertAlmostEqual(result["derived"], 5.0)

        with self.assertRaisesRegex(ValueError, "cyclic metric dependencies"):
            prepare_definitions({"a": "m['b'] + 1", "b": "m['a'] + 1"})
        with self.assertRaisesRegex(ValueError, "unknown metric symbol"):
            prepare_definitions({"bad": "unknown_symbol + 1"})
        with self.assertRaisesRegex(ValueError, "analysis 'dc' is not configured"):
            prepare_definitions(
                {"bad": {"expr": "avg(V('x'))", "analysis": "dc"}},
                analyses=("tran",), default_analysis="tran"
            )

    def test_tran_and_reversed_dc_datasets_use_declared_analysis(self):
        tran_t = np.array([0.0, 1.0, 2.0])
        dc_t = np.array([3.0, 2.0, 1.0, 0.0])
        dc_in = dc_t.copy()
        dc_out = 5.0 - 2.0 * dc_in
        definitions = {
            "tran_mean": {"expr": "avg(V('x'))", "analysis": "tran"},
            "gain": {"expr": "dc_gain(V('out'), V('in'))", "analysis": "dc"},
            "offset": {"expr": "dc_offset(V('out'), V('in'))", "analysis": "dc"},
        }
        result = compute_metrics(
            definitions,
            datasets={
                "tran": (tran_t, {"x": tran_t}),
                "dc": (dc_t, {"in": dc_in, "out": dc_out}),
            },
        )
        self.assertAlmostEqual(result["tran_mean"], 1.0)
        self.assertAlmostEqual(result["gain"], -2.0)
        self.assertAlmostEqual(result["offset"], 5.0)

    def test_non_scalar_nonfinite_and_complex_results_name_the_metric(self):
        kwargs = {"t": np.array([0.0, 1.0]), "sigs": {"x": np.array([1.0, 2.0])}}
        expressions = {
            "array_result": "V('x')",
            "nan_result": "float('nan')",
            "inf_result": "float('inf')",
            "complex_result": "np.fft.fft([1, 0])[0]",
        }
        expected = {
            "array_result": "metric must reduce to one real scalar",
            "nan_result": "metric result is not finite",
            "inf_result": "metric result is not finite",
            "complex_result": "metric must reduce to one real scalar",
        }
        for name, expression in expressions.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError,
                                            "metric %r" % name) as caught:
                    compute_metrics({name: expression}, **kwargs)
                self.assertIn(expected[name], str(caught.exception))

    def test_measurement_signature_tracks_content_not_mapping_order(self):
        first = {"b": "avg(V('x'))", "a": "dc_gain(V('y'), V('x'))"}
        reordered = {"a": "dc_gain(V('y'), V('x'))", "b": "avg(V('x'))"}
        base = measurement_signature(first, ["tran", "dc"], ["y", "x"])
        self.assertEqual(
            base, measurement_signature(reordered, ["tran", "dc"], ["x", "y"])
        )
        self.assertNotEqual(
            base,
            measurement_signature(
                {"b": "avg(V('z'))", "a": "dc_gain(V('y'), V('x'))"},
                ["tran", "dc"], ["x", "y"]
            ),
        )
        self.assertNotEqual(base, measurement_signature(first, ["tran"], ["x", "y"]))
        self.assertNotEqual(base, measurement_signature(first, ["tran", "dc"], ["x"]))


if __name__ == "__main__":
    unittest.main()
