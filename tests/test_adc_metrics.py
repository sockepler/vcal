import unittest

import numpy as np

from optserver.adc_metrics import (
    adc_static,
    code_density,
    decode,
    spectrum,
    transition_metrics,
)


class SpectrumTests(unittest.TestCase):
    def test_coherent_harmonic_ratios_and_ideal_limits(self):
        n_samples = 1024
        index = np.arange(n_samples)
        waveform = np.sin(2.0 * np.pi * 5 * index / n_samples)
        waveform += 0.1 * np.sin(2.0 * np.pi * 10 * index / n_samples)

        result = spectrum(waveform, 1.0e6, 5, harmonics=5)

        self.assertAlmostEqual(result["sndr"], 20.0, places=10)
        self.assertGreater(result["snr"], 250.0)
        self.assertAlmostEqual(result["thd"], -20.0, places=10)
        self.assertAlmostEqual(result["sfdr"], 20.0, places=10)
        self.assertEqual(result["harmonic_bins"], [10, 15, 20, 25])
        self.assertAlmostEqual(result["enob"], (20.0 - 1.76) / 6.02, places=10)

    def test_nonharmonic_noise_and_nyquist_power_are_one_sided(self):
        n_samples = 128
        index = np.arange(n_samples)
        waveform = np.sin(2.0 * np.pi * 5 * index / n_samples)
        waveform += 0.1 * np.sin(2.0 * np.pi * 17 * index / n_samples)
        result = spectrum(waveform, 1.0e6, 5, harmonics=0)
        self.assertAlmostEqual(result["snr"], 20.0, places=10)
        self.assertAlmostEqual(result["sndr"], 20.0, places=10)
        self.assertTrue(np.isneginf(result["thd"]))

        nyquist_waveform = np.sin(2.0 * np.pi * 5 * index / n_samples)
        nyquist_waveform += 0.2 * np.cos(np.pi * index)
        nyquist = spectrum(nyquist_waveform, 1.0e6, 5, harmonics=0)
        # A Nyquist cosine is its own conjugate bin and therefore must not be
        # doubled in the one-sided spectrum.  Its mean-square power is 0.2**2.
        self.assertAlmostEqual(nyquist["noise_power"], 0.04, places=12)
        self.assertAlmostEqual(nyquist["snr"], 10.0 * np.log10(0.5 / 0.04), places=10)

    def test_folded_harmonic_and_odd_even_lengths(self):
        n_samples = 128
        index = np.arange(n_samples)
        waveform = np.sin(2.0 * np.pi * 40 * index / n_samples)
        # The second harmonic is at bin 80 and aliases to positive bin 48.
        waveform += 0.1 * np.sin(2.0 * np.pi * 80 * index / n_samples)
        result = spectrum(waveform, 2.0e6, 40, harmonics=2)
        self.assertEqual(result["harmonic_bins"], [48])
        self.assertAlmostEqual(result["thd"], -20.0, places=10)

        for n_samples in (63, 64):
            index = np.arange(n_samples)
            waveform = np.sin(2.0 * np.pi * 5 * index / n_samples)
            result = spectrum(waveform, 1.0e6, 5, harmonics=0)
            self.assertGreater(result["snr"], 250.0)
            self.assertEqual(result["fundamental_bins"], [5])

    def test_periodic_hann_accepts_noncoherent_float_bin(self):
        n_samples = 256
        index = np.arange(n_samples)
        fund = 7.3
        waveform = np.sin(2.0 * np.pi * fund * index / n_samples)
        result = spectrum(waveform, 1.0e6, fund, window="hann", harmonics=0)
        self.assertEqual(result["fundamental_bins"], [6, 7, 8, 9])
        self.assertGreater(result["snr"], 25.0)
        self.assertGreater(result["sfdr"], 25.0)

    def test_validation_and_overlapping_measurement_bands(self):
        with self.assertRaises(ValueError):
            spectrum(np.ones(15), 1.0, 1)
        with self.assertRaises(ValueError):
            spectrum(np.ones(64), 1.0, 1)
        with self.assertRaises(ValueError):
            spectrum(np.sin(2.0 * np.pi * np.arange(64) / 64), 1.0, 0)
        with self.assertRaises(ValueError):
            spectrum(np.sin(2.0 * np.pi * np.arange(64) / 64), 1.0, 32)
        with self.assertRaises(ValueError):
            spectrum(np.sin(2.0 * np.pi * np.arange(64) / 64), 1.0, 1.5)

        index = np.arange(64)
        waveform = np.sin(2.0 * np.pi * 2 * index / 64)
        with self.assertRaises(ValueError):
            spectrum(waveform, 1.0, 2, window="hann", harmonics=2)

        with self.assertRaises(ValueError):
            spectrum(np.zeros(64), 1.0, 1)


class StaticAdcTests(unittest.TestCase):
    def test_uniform_ramp_code_density_is_ideal(self):
        codes = np.repeat(np.arange(8), 20)
        result = code_density(codes, 3)
        np.testing.assert_array_equal(result["counts"], np.full(8, 20))
        np.testing.assert_allclose(result["dnl"], 0.0)
        np.testing.assert_allclose(result["inl"], 0.0)
        self.assertEqual(result["missing_codes"], [])
        self.assertEqual(result["offset_error"], None)
        self.assertEqual(result["gain_error"], None)
        self.assertAlmostEqual(adc_static(codes, 3, method="histogram")["dnl_peak"], 0.0)

    def test_code_density_missing_code_and_explicit_ramp_checks(self):
        codes = np.concatenate(
            [np.repeat(code, 20) for code in range(8) if code != 4]
        )
        result = code_density(codes, 3)
        self.assertEqual(result["missing_codes"], [4])
        self.assertAlmostEqual(result["dnl"][4], -1.0)

        with self.assertRaises(ValueError):
            code_density(np.arange(7), 3)
        with self.assertRaises(ValueError):
            code_density(np.repeat(np.arange(1, 8), 5), 3)
        saturated = np.concatenate(
            [np.repeat(0, 20), np.repeat(7, 20)]
            + [np.repeat(code, 1) for code in range(1, 7)]
        )
        with self.assertRaises(ValueError):
            code_density(saturated, 3)

    def test_transition_boundaries_report_dnl_inl_offset_gain(self):
        ideal = transition_metrics(np.arange(9, dtype=float), 3)
        np.testing.assert_allclose(ideal["dnl"], 0.0)
        np.testing.assert_allclose(ideal["inl"], 0.0)
        self.assertAlmostEqual(ideal["offset_error"], 0.0)
        self.assertAlmostEqual(ideal["gain_error"], 0.0)

        measured = transition_metrics(0.25 + 1.1 * np.arange(9), 3)
        self.assertAlmostEqual(measured["offset_error"], 0.25)
        self.assertAlmostEqual(measured["gain_error"], 0.1)
        self.assertAlmostEqual(measured["dnl_peak"], 0.0)
        self.assertAlmostEqual(measured["inl_peak"], 0.0)

        missing = transition_metrics(np.array([0.0, 1.0, 1.0, 3.0, 4.0]), 2)
        self.assertEqual(missing["missing_codes"], [1])
        with self.assertRaises(ValueError):
            transition_metrics(np.arange(1, 4, dtype=float), 2)
        with self.assertRaises(ValueError):
            adc_static(np.repeat(np.arange(8), 2), 3)


class DecodeTests(unittest.TestCase):
    def test_column_bit_decode_order_and_validation(self):
        samples = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(decode(samples), np.array([1, 2, 5], dtype=np.uint64))
        np.testing.assert_array_equal(
            decode(samples, threshold=0.8, msb_first=False),
            np.array([4, 2, 5], dtype=np.uint64),
        )
        with self.assertRaises(ValueError):
            decode([np.zeros(2), np.zeros(3)])
        with self.assertRaises(ValueError):
            decode(np.array([[0.0, np.nan]]))
        with self.assertRaises(ValueError):
            decode(np.zeros((2, 1)), threshold=np.inf)


if __name__ == "__main__":
    unittest.main()
