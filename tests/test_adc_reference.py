import unittest

import numpy as np

from optserver.adc_metrics import code_density, decode, spectrum, transition_metrics


class AdcReferenceTests(unittest.TestCase):
    def test_rect_dynamic_metrics_match_known_bin_powers(self):
        n = 128
        index = np.arange(n, dtype=float)
        fundamental = 1.2
        second = .3
        nonharmonic_spur = .1
        nyquist_noise = .2
        samples = (
            fundamental * np.sin(2.0 * np.pi * 5.0 * index / n)
            + second * np.cos(2.0 * np.pi * 10.0 * index / n)
            + nonharmonic_spur * np.sin(2.0 * np.pi * 17.0 * index / n)
            + nyquist_noise * np.cos(np.pi * index)
        )

        result = spectrum(samples, 1.0e6, 5, harmonics=2)
        signal_power = fundamental ** 2 / 2.0
        distortion_power = second ** 2 / 2.0
        noise_power = nonharmonic_spur ** 2 / 2.0 + nyquist_noise ** 2
        self.assertAlmostEqual(result["signal_power"], signal_power, places=12)
        self.assertAlmostEqual(result["distortion_power"], distortion_power, places=12)
        self.assertAlmostEqual(result["noise_power"], noise_power, places=12)
        self.assertAlmostEqual(
            result["sndr"], 10.0 * np.log10(signal_power / (noise_power + distortion_power)),
            places=12,
        )
        self.assertAlmostEqual(
            result["snr"], 10.0 * np.log10(signal_power / noise_power), places=12
        )
        self.assertAlmostEqual(
            result["thd"], 10.0 * np.log10(distortion_power / signal_power), places=12
        )
        # The second harmonic is the largest non-carrier spur in this record.
        self.assertEqual(result["spur_bin"], 10)
        self.assertAlmostEqual(
            result["sfdr"], 10.0 * np.log10(signal_power / distortion_power), places=12
        )
        self.assertEqual(result["harmonic_bins"], [10])

    def test_hann_removes_dc_before_window_and_integrates_harmonic_spur(self):
        n = 128
        index = np.arange(n, dtype=float)
        fundamental = np.sin(2.0 * np.pi * 5.0 * index / n)
        harmonic = .1 * np.cos(2.0 * np.pi * 10.0 * index / n)
        plain = spectrum(fundamental + harmonic, 1.0e6, 5,
                         window="hann", harmonics=2)
        offset = spectrum(fundamental + harmonic + 3.0, 1.0e6, 5,
                          window="hann", harmonics=2)

        # The periodic Hann main lobes are integrated, so the known powers
        # remain A**2/2 even though each tone spans multiple FFT bins.
        self.assertAlmostEqual(plain["signal_power"], .5, places=12)
        self.assertAlmostEqual(plain["distortion_power"], .005, places=12)
        self.assertAlmostEqual(plain["spur_power"], .005, places=12)
        self.assertAlmostEqual(plain["sfdr"], 20.0, places=12)
        self.assertAlmostEqual(offset["signal_power"], plain["signal_power"], places=12)
        self.assertAlmostEqual(offset["distortion_power"], plain["distortion_power"], places=12)
        self.assertAlmostEqual(offset["noise_power"], plain["noise_power"], places=12)
        self.assertAlmostEqual(offset["sfdr"], plain["sfdr"], places=12)

    def test_very_small_real_harmonic_is_not_erased(self):
        n = 128
        index = np.arange(n, dtype=float)
        harmonic_amplitude = 1.0e-8
        samples = (np.sin(2.0 * np.pi * 5.0 * index / n)
                   + harmonic_amplitude * np.cos(2.0 * np.pi * 10.0 * index / n))
        result = spectrum(samples, 1.0e6, 5, harmonics=2)
        expected_power = harmonic_amplitude ** 2 / 2.0
        self.assertGreater(result["distortion_power"], 0.0)
        np.testing.assert_allclose(result["distortion_power"], expected_power,
                                   rtol=1e-8, atol=1e-30)
        self.assertAlmostEqual(result["thd"], -160.0, places=6)

    def test_transition_metrics_use_complete_boundary_array_formulae(self):
        bits = 3
        ideal_step = .25
        ideal_start = -.5
        levels = 1 << bits

        # A pure offset and a pure gain leave the linearity errors at zero;
        # the separate nominal-reference fields still report each error.
        for offset_error, gain_error in ((.3, 0.0), (0.0, .2)):
            with self.subTest(offset_error=offset_error, gain_error=gain_error):
                boundaries = (ideal_start + offset_error * ideal_step
                              + np.arange(levels + 1) * ideal_step * (1.0 + gain_error))
                result = transition_metrics(
                    boundaries, bits, ideal_step=ideal_step, ideal_start=ideal_start
                )
                np.testing.assert_allclose(result["dnl"], 0.0, atol=1e-14)
                np.testing.assert_allclose(result["inl"], 0.0, atol=1e-14)
                self.assertAlmostEqual(result["offset_error"], offset_error)
                self.assertAlmostEqual(result["gain_error"], gain_error)
                self.assertEqual(result["dnl"].shape, (levels,))
                self.assertEqual(result["inl"].shape, (levels + 1,))

        # For a nonlinear record, DNL uses the measured endpoint LSB and INL
        # is measured against the line joining the two endpoints.
        widths = np.array([.2, .25, .3, .25, .25, .2, .3, .3])
        boundaries = np.concatenate(([-.4], -.4 + np.cumsum(widths)))
        result = transition_metrics(
            boundaries, bits, ideal_step=ideal_step, ideal_start=ideal_start
        )
        measured_lsb = (boundaries[-1] - boundaries[0]) / levels
        endpoint_line = boundaries[0] + np.arange(levels + 1) * measured_lsb
        expected_dnl = widths / measured_lsb - 1.0
        expected_inl = (boundaries - endpoint_line) / measured_lsb
        np.testing.assert_allclose(result["widths"], widths)
        np.testing.assert_allclose(result["measured_lsb"], measured_lsb)
        np.testing.assert_allclose(result["dnl"], expected_dnl, atol=1e-14)
        np.testing.assert_allclose(result["inl"], expected_inl, atol=1e-14)
        self.assertAlmostEqual(result["offset_error"],
                               (boundaries[0] - ideal_start) / ideal_step)
        self.assertAlmostEqual(result["gain_error"],
                               (boundaries[-1] - boundaries[0]) /
                               (levels * ideal_step) - 1.0)
        self.assertEqual(result["levels"], levels)
        self.assertTrue(result["endpoint_coverage"])

        with self.assertRaisesRegex(ValueError, r"2\*\*bits \+ 1"):
            transition_metrics(np.arange(8, dtype=float), bits)

    def test_transition_zero_width_is_missing_but_wide_code_is_not(self):
        boundaries = np.array([0., 1., 1., 3., 4., 5., 6., 7., 8.])
        result = transition_metrics(boundaries, 3)
        self.assertEqual(result["missing_codes"], [1])
        self.assertEqual(result["widths"].tolist(), [1., 0., 2., 1., 1., 1., 1., 1.])
        self.assertNotIn(2, result["missing_codes"])

    def test_uniform_ramp_code_density_has_zero_dnl_and_inl(self):
        bits = 4
        samples_per_code = 17
        codes = np.repeat(np.arange(1 << bits), samples_per_code)
        result = code_density(codes, bits)

        self.assertEqual(result["levels"], 1 << bits)
        self.assertEqual(result["ideal_count"], float(samples_per_code))
        np.testing.assert_array_equal(
            result["counts"], np.full(1 << bits, samples_per_code, dtype=np.int64)
        )
        np.testing.assert_allclose(result["dnl"], 0.0)
        np.testing.assert_allclose(result["inl"], 0.0)
        self.assertEqual(result["missing_codes"], [])

    def test_decode_matches_msb_and_lsb_bit_weighting(self):
        bits = np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(decode(bits), np.array([5, 6, 3, 4], dtype=np.uint64))
        np.testing.assert_array_equal(
            decode(bits, msb_first=False), np.array([5, 3, 6, 1], dtype=np.uint64)
        )


if __name__ == "__main__":
    unittest.main()
