"""Small, dependency-free ADC measurement helpers.

The functions in this module deliberately operate on already sampled data.  They
do not choose a sampling phase, run a simulator, or import the rest of the
application.  ``spectrum`` uses FFT *bin numbers* for ``fund``; callers that
have a frequency in hertz can convert it with ``fund = f * n / fs``.

Static measurements are kept explicit.  ``code_density`` is for a complete,
uniform ramp and returns code-density DNL/INL.  ``transition_metrics`` is for
measured transition boundaries.  A transient code vector is never silently
interpreted as an INL measurement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _finite_float(value: Any, name: str) -> float:
    """Return a finite scalar float or raise a useful validation error."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _real_vector(values: Any, name: str) -> np.ndarray:
    """Convert a real, finite one-dimensional input to float64."""

    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _db_ratio(numerator: float, denominator: float) -> float:
    """Return 10 log10(numerator / denominator) without clamping ideal cases."""

    numerator = float(numerator)
    denominator = float(denominator)
    if numerator < 0.0 or denominator < 0.0:
        raise ValueError("power values must be non-negative")
    if denominator == 0.0:
        if numerator == 0.0:
            return float("nan")
        return float("inf")
    if numerator == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(numerator / denominator))


def _window_values(name: str, n: int) -> tuple[str, np.ndarray, float]:
    window = str(name).strip().lower()
    if window in {"rect", "rectangle", "rectangular"}:
        values = np.ones(n, dtype=float)
        return "rect", values, 1.0
    if window in {"hann", "hanning", "periodic_hann", "periodic-hann"}:
        # np.hanning is symmetric.  The periodic form is what an N-point FFT
        # uses for a periodic tone, so construct it directly.
        index = np.arange(n, dtype=float)
        values = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / n)
        return "hann", values, 2.0
    raise ValueError("window must be 'rect' or periodic 'hann'")


def _one_sided_power(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return window-normalized one-sided power, preserving Nyquist power."""

    transform = np.fft.rfft(samples * weights)
    # Normalize to mean-square power.  Dividing by N as well as the window
    # energy makes a unit-amplitude coherent sine report 0.5 total power and
    # keeps the metadata independent of record length.
    power = (np.abs(transform) ** 2) / float(samples.size * np.sum(weights * weights))
    # Interior positive-frequency bins have a conjugate partner.  DC and the
    # even-length Nyquist bin do not and therefore must not be doubled.
    if power.size > 1:
        if samples.size % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0
    return np.asarray(power, dtype=float)


def _band(center: float, half_width: float, n: int) -> np.ndarray:
    """Return integer rFFT bins in a closed +/- ``half_width`` band."""

    last = n // 2
    low = max(0, int(math.ceil(center - half_width)))
    high = min(last, int(math.floor(center + half_width)))
    if high < low:
        # A positive width with a center between integer bins always has a
        # nearest bin.  This branch mainly protects unusual fractional input.
        nearest = int(np.clip(np.rint(center), 0, last))
        return np.asarray([nearest], dtype=int)
    return np.arange(low, high + 1, dtype=int)


def _fold_bin(center: float, n: int) -> float:
    """Fold an FFT-bin frequency into the real rFFT interval [0, N/2]."""

    folded = float(center % n)
    nyquist = n / 2.0
    if folded > nyquist:
        folded = float(n - folded)
    # Avoid a negative zero in metadata and dictionary keys.
    if abs(folded) < 1e-15:
        folded = 0.0
    return folded


def _unique_float(values: Iterable[float]) -> List[float]:
    """Deduplicate aliased floating-point bin centers deterministically."""

    result: List[float] = []
    seen = set()
    for value in values:
        key = round(float(value), 12)
        if key not in seen:
            seen.add(key)
            result.append(float(value))
    return result


def spectrum(
    samples: Sequence[float],
    fs: float,
    fund: float,
    window: str = "rect",
    harmonics: int = 5,
    bin_width: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure ADC dynamic metrics from a real sampled waveform.

    Parameters
    ----------
    samples:
        A finite real vector with at least 16 samples.  A constant vector is
        rejected because it has no measurable AC fundamental.
    fs:
        Positive sampling rate in hertz, used for metadata and bin frequency.
    fund:
        Fundamental FFT-bin index.  Rectangular windows require an integer
        bin.  A fractional bin is accepted for periodic Hann and is measured
        in a contiguous main-lobe band.
    window:
        ``"rect"`` or periodic ``"hann"``.  For Hann the default main-lobe
        half-width is two bins; ``bin_width`` overrides that half-width.
    harmonics:
        Highest harmonic order to classify (orders 2 through this value).
        Aliased centers are folded to the positive rFFT interval and duplicate
        centers are counted once.
    bin_width:
        Hann main-lobe half-width in FFT bins.  Rectangular measurements always
        use exactly one bin, as required for a coherent single-bin tone.

    Returns
    -------
    dict
        Scalar ``sndr``, ``snr``, ``thd`` (dBc), ``sfdr`` (dBc), and ``enob``
        plus power totals, selected bins, and frequency metadata.  A zero
        denominator produces the mathematically meaningful ``inf`` or
        ``-inf`` result rather than a silent numerical clamp.
    """

    values = _real_vector(samples, "samples")
    n = int(values.size)
    if n < 16:
        raise ValueError("samples must contain at least 16 points")
    sample_rate = _finite_float(fs, "fs")
    if sample_rate <= 0.0:
        raise ValueError("fs must be positive")
    fundamental = _finite_float(fund, "fund")
    normalized_window, weights, default_half_width = _window_values(window, n)

    try:
        highest_harmonic = int(harmonics)
    except (TypeError, ValueError) as exc:
        raise ValueError("harmonics must be a non-negative integer") from exc
    if isinstance(harmonics, (float, np.floating)) and not float(harmonics).is_integer():
        raise ValueError("harmonics must be a non-negative integer")
    if highest_harmonic < 0:
        raise ValueError("harmonics must be a non-negative integer")

    nyquist_bin = n / 2.0
    if fundamental <= 0.0 or fundamental >= nyquist_bin:
        raise ValueError("fund must be strictly between DC and Nyquist")
    if normalized_window == "rect":
        if not float(fundamental).is_integer():
            raise ValueError("rect window requires an integer FFT-bin fund")
        fundamental = float(int(round(fundamental)))
        fundamental_bins = np.asarray([int(fundamental)], dtype=int)
    else:
        if bin_width is None:
            half_width = default_half_width
        else:
            half_width = _finite_float(bin_width, "bin_width")
            if half_width <= 0.0:
                raise ValueError("bin_width must be positive")
        # DC is always excluded from the metric.  A near-DC main lobe is
        # clipped to the positive bins rather than silently including offset.
        fundamental_bins = _band(fundamental, half_width, n)
        fundamental_bins = fundamental_bins[fundamental_bins > 0]
        if fundamental_bins.size == 0:
            raise ValueError("fundamental band must contain a positive-frequency bin")

    # Exact constant or zero input has no AC information.  Do this check on
    # the unwindowed samples so a Hann endpoint does not make a valid tone look
    # constant by accident.
    if float(np.ptp(values)) == 0.0:
        raise ValueError("samples must contain a non-constant waveform")
    ac_energy = float(np.sum((values - np.mean(values)) ** 2))
    if ac_energy <= 0.0:
        raise ValueError("samples must contain non-zero AC energy")

    if normalized_window == "rect":
        half_width = 0.0
    # Remove the DC component before applying a window: otherwise Hann
    # spreads a static offset into adjacent bins and calls it noise.
    power = _one_sided_power(values - np.mean(values), weights)

    harmonic_centers = _unique_float(
        _fold_bin(order * fundamental, n) for order in range(2, highest_harmonic + 1)
    )
    dc_harmonic_centers = [center for center in harmonic_centers if center == 0.0]
    # DC is intentionally excluded from ADC dynamic metrics.  Keep a folded
    # DC harmonic visible in metadata, but do not count its offset or window
    # leakage as AC distortion.
    ac_harmonic_centers = [center for center in harmonic_centers if center != 0.0]

    harmonic_bands: List[np.ndarray] = []
    harmonic_band_centers: List[float] = []
    for center in ac_harmonic_centers:
        if normalized_window == "rect":
            center_bin = int(round(center))
            band = np.asarray([center_bin], dtype=int)
        else:
            band = _band(center, half_width, n)
        # DC is an offset reference, even when a very-low-frequency aliased
        # harmonic's integration band happens to touch bin zero.
        band = np.unique(band[(band > 0) & (band <= n // 2)])
        if band.size == 0:
            continue
        harmonic_bands.append(band)
        harmonic_band_centers.append(center)

    def _as_set(array: np.ndarray) -> set[int]:
        return {int(item) for item in array.tolist()}

    fundamental_set = _as_set(fundamental_bins)
    occupied_harmonics: set[int] = set()
    harmonic_bins_list: List[int] = []
    for center, band in zip(harmonic_band_centers, harmonic_bands):
        band_set = _as_set(band)
        if fundamental_set.intersection(band_set):
            raise ValueError("fundamental and harmonic measurement bands overlap")
        if occupied_harmonics.intersection(band_set):
            raise ValueError("harmonic measurement bands overlap")
        occupied_harmonics.update(band_set)
        harmonic_bins_list.extend(sorted(band_set))
    harmonic_bins = np.asarray(sorted(occupied_harmonics), dtype=int)

    all_bins = np.arange(power.size, dtype=int)
    fundamental_mask = np.zeros(power.size, dtype=bool)
    fundamental_mask[fundamental_bins] = True
    harmonic_mask = np.zeros(power.size, dtype=bool)
    if harmonic_bins.size:
        harmonic_mask[harmonic_bins] = True

    # DC is excluded as offset.  Everything else is either the measured
    # fundamental, a requested harmonic, or broadband/spurious residue.
    signal_power = float(np.sum(power[fundamental_mask]))
    distortion_power = float(np.sum(power[harmonic_mask]))
    dc_mask = all_bins == 0
    noise_mask = ~(dc_mask | fundamental_mask | harmonic_mask)
    noise_power = float(np.sum(power[noise_mask]))
    if signal_power <= 0.0:
        raise ValueError("fundamental measurement band has zero energy")

    spur_mask = ~(dc_mask | fundamental_mask)
    spur_candidates = all_bins[spur_mask]
    if spur_candidates.size:
        if normalized_window == "rect":
            spur_powers = power[spur_candidates]
        else:
            # Compare integrated main lobes with the integrated carrier.
            # Keep candidate bands outside the carrier and DC bands. A
            # cumulative sum avoids an O(N * width) search for wide windows.
            residual = np.where(spur_mask, power, 0.0)
            cumulative = np.concatenate(([0.0], np.cumsum(residual)))
            radius = int(math.floor(half_width))
            lo = np.maximum(spur_candidates - radius, 1)
            hi = np.minimum(spur_candidates + radius + 1, len(power))
            spur_powers = cumulative[hi] - cumulative[lo]
        spur_offset = int(np.argmax(spur_powers))
        spur_bin: Optional[int] = int(spur_candidates[spur_offset])
        spur_power = float(spur_powers[spur_offset])
    else:
        spur_bin = None
        spur_power = 0.0

    sndr = _db_ratio(signal_power, noise_power + distortion_power)
    snr = _db_ratio(signal_power, noise_power)
    thd = _db_ratio(distortion_power, signal_power)
    sfdr = _db_ratio(signal_power, spur_power)
    enob = float((sndr - 1.76) / 6.02)

    return {
        "sndr": float(sndr),
        "snr": float(snr),
        "thd": float(thd),
        "sfdr": float(sfdr),
        "enob": float(enob),
        "n": n,
        "fs": sample_rate,
        "fund": float(fundamental),
        "fundamental_frequency": float(fundamental * sample_rate / n),
        "window": normalized_window,
        "harmonics": highest_harmonic,
        "bin_width": None if normalized_window == "rect" else float(half_width),
        "one_sided": True,
        "nyquist_bin": float(nyquist_bin),
        "dc_excluded": True,
        "fundamental_bins": [int(item) for item in fundamental_bins.tolist()],
        "harmonic_bins": [int(item) for item in harmonic_bins_list],
        "harmonic_centers": [float(item) for item in harmonic_centers],
        "dc_harmonic_centers": [float(item) for item in dc_harmonic_centers],
        "signal_power": signal_power,
        "noise_power": noise_power,
        "distortion_power": distortion_power,
        "spur_bin": spur_bin,
        "spur_power": spur_power,
    }


def _validate_bits(bits: Any) -> int:
    try:
        result = int(bits)
    except (TypeError, ValueError) as exc:
        raise ValueError("bits must be a positive integer") from exc
    if isinstance(bits, (float, np.floating)) and not float(bits).is_integer():
        raise ValueError("bits must be a positive integer")
    if result <= 0 or result > 30:
        raise ValueError("bits must be between 1 and 30")
    return result


def _validate_stimulus(stimulus: str) -> str:
    name = str(stimulus).strip().lower()
    if name not in {"ramp", "uniform_ramp", "uniform-ramp", "code_density", "code-density"}:
        raise ValueError("stimulus must identify a uniform ramp")
    return "ramp"


def code_density(
    codes: Sequence[float],
    bits: int,
    stimulus: str = "ramp",
) -> Dict[str, Any]:
    """Compute static DNL/INL from a complete uniform-ramp code-density run.

    ``codes`` are ADC output codes, not arbitrary transient samples.  At least
    one sample per code is required, both endpoint codes must be present, and
    a strong endpoint count excess is rejected because endpoint clipping biases
    the uniform-ramp estimate.  DNL is ``count / ideal_count - 1``.  INL is the
    cumulative DNL at each code edge (the first code is the zero reference).
    Absolute offset and gain are not identifiable without a calibrated ramp
    range and are returned as ``None``.
    """

    _validate_stimulus(stimulus)
    bit_count = _validate_bits(bits)
    values = _real_vector(codes, "codes")
    levels = 1 << bit_count
    if values.size < levels:
        raise ValueError("uniform ramp code-density run must contain at least one sample per code")
    if not np.all(values == np.floor(values)):
        raise ValueError("codes must be integer-valued")
    integer_codes = values.astype(np.int64)
    if np.any(integer_codes < 0) or np.any(integer_codes >= levels):
        raise ValueError("codes must lie between 0 and 2**bits - 1")

    counts = np.bincount(integer_codes, minlength=levels).astype(np.int64)
    if counts[0] == 0 or counts[-1] == 0:
        raise ValueError("uniform ramp must cover both endpoint codes")

    # A clipped endpoint can dominate a nominally uniform ramp.  If the median
    # interior count is zero there is not enough information to diagnose this;
    # the missing-code result remains explicit instead of inventing an INL.
    endpoint_saturation = False
    if levels > 2:
        interior = counts[1:-1]
        positive = interior[interior > 0]
        if positive.size:
            reference = float(np.median(positive))
            endpoint_saturation = bool(
                counts[0] > 3.0 * reference or counts[-1] > 3.0 * reference
            )
            if endpoint_saturation:
                raise ValueError(
                    "endpoint saturation biases the ramp; use a measured interior range"
                )

    ideal_count = float(values.size) / levels
    dnl = counts.astype(float) / ideal_count - 1.0
    inl = np.concatenate((np.asarray([0.0]), np.cumsum(dnl)))
    missing = np.flatnonzero(counts == 0).astype(int).tolist()
    return {
        "method": "code_density",
        "stimulus": "ramp",
        "bits": bit_count,
        "levels": levels,
        "samples": int(values.size),
        "counts": counts.tolist(),
        "ideal_count": ideal_count,
        "dnl": dnl,
        "inl": inl,
        "dnl_peak": float(np.max(np.abs(dnl))),
        "inl_peak": float(np.max(np.abs(inl))),
        "missing_codes": missing,
        "endpoint_saturation": endpoint_saturation,
        "offset_error": None,
        "gain_error": None,
    }


def transition_metrics(
    transitions: Sequence[float],
    bits: int,
    *,
    ideal_step: float = 1.0,
    ideal_start: float = 0.0,
) -> Dict[str, Any]:
    """Compute static metrics from monotonic transition-level boundaries.

    A complete boundary array has ``2**bits + 1`` entries, including both
    outer endpoints.  Internal-only arrays are rejected because their
    unmeasured outer widths cannot support a full-code DNL/INL result.

    DNL and INL use the measured endpoint LSB, removing offset and gain.
    INL is reported at all L+1 boundaries relative to the endpoint line.
    Repeated boundaries describe zero-width (missing) codes. ``ideal_step``
    and ``ideal_start`` specify the nominal transfer for the separate offset
    (nominal LSB) and fractional gain errors.
    """

    bit_count = _validate_bits(bits)
    values = _real_vector(transitions, "transitions")
    levels = 1 << bit_count
    step = _finite_float(ideal_step, "ideal_step")
    start = _finite_float(ideal_start, "ideal_start")
    if step <= 0.0:
        raise ValueError("ideal_step must be positive")

    if values.size != levels + 1:
        raise ValueError(
            "transitions must contain exactly 2**bits + 1 boundaries including both endpoints"
        )
    full_boundaries = values
    endpoint_coverage = True
    if np.any(np.diff(full_boundaries) < 0.0) or full_boundaries[-1] <= full_boundaries[0]:
        raise ValueError("transition levels must be nondecreasing with a positive span")

    widths = np.diff(full_boundaries)
    measured_lsb = float((full_boundaries[-1] - full_boundaries[0]) / levels)
    dnl = widths / measured_lsb - 1.0
    endpoint_line = full_boundaries[0] + np.arange(levels + 1) * measured_lsb
    inl = (full_boundaries - endpoint_line) / measured_lsb
    missing_codes = np.flatnonzero(widths == 0).astype(int).tolist()

    if endpoint_coverage:
        offset_error: Optional[float] = float((full_boundaries[0] - start) / step)
        gain_error: Optional[float] = float(
            (full_boundaries[-1] - full_boundaries[0]) / (levels * step) - 1.0
        )
    return {
        "method": "transition",
        "bits": bit_count,
        "levels": levels,
        "transitions": full_boundaries,
        "endpoint_coverage": endpoint_coverage,
        "widths": widths,
        "measured_lsb": measured_lsb,
        "dnl": dnl,
        "inl": inl,
        "dnl_peak": float(np.max(np.abs(dnl))),
        "inl_peak": float(np.max(np.abs(inl))),
        "missing_codes": sorted(set(missing_codes)),
        "offset_error": offset_error,
        "gain_error": gain_error,
        "ideal_step": step,
        "ideal_start": start,
    }


def adc_static(
    codes: Sequence[float],
    bits: int,
    stimulus: str = "ramp",
    method: str = "endpoint",
) -> Dict[str, Any]:
    """Dispatch an explicit static ADC measurement.

    ``method='histogram'`` or ``'code_density'`` treats ``codes`` as samples
    from a uniform ramp and calls :func:`code_density`.  The default
    ``method='endpoint'`` treats ``codes`` as transition boundaries and calls
    :func:`transition_metrics`.  A raw sample vector passed to the endpoint
    method raises a clear length error instead of being mistaken for INL.
    """

    method_name = str(method).strip().lower()
    if method_name in {"histogram", "code_density", "code-density"}:
        return code_density(codes, bits, stimulus=stimulus)
    if method_name in {"endpoint", "transition", "transitions"}:
        normalized_stimulus = _validate_stimulus(stimulus)
        result = transition_metrics(codes, bits)
        result["stimulus"] = normalized_stimulus
        return result
    raise ValueError(
        "method must be 'endpoint', 'transition', 'histogram', or 'code_density'"
    )


def decode(
    bits: Sequence[Sequence[float]],
    threshold: float = 0.5,
    msb_first: bool = True,
) -> np.ndarray:
    """Decode an array whose columns are sampled bit waveforms.

    ``bits`` has shape ``(samples, bit_count)``.  For ``msb_first=True`` the
    first column is the most-significant bit; for ``False`` it is the least
    significant bit.  Each column must have the same finite length (ragged
    input is rejected before conversion to a numeric matrix).
    """

    try:
        raw = np.asarray(bits)
    except (TypeError, ValueError) as exc:
        raise ValueError("bits must be a rectangular two-dimensional array") from exc
    if raw.dtype == object and raw.ndim != 2:
        raise ValueError("bits must be a rectangular two-dimensional array")
    if raw.ndim != 2:
        raise ValueError("bits must be a rectangular two-dimensional array")
    if np.iscomplexobj(raw):
        raise ValueError("bits must be real")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bits must contain numeric values") from exc
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("bits must contain at least one sample and one bit")
    if not np.all(np.isfinite(values)):
        raise ValueError("bits must contain only finite values")
    if values.shape[1] > 63:
        raise ValueError("bits may contain at most 63 columns")
    level = _finite_float(threshold, "threshold")
    if not isinstance(msb_first, (bool, np.bool_)):
        raise ValueError("msb_first must be boolean")

    bit_count = values.shape[1]
    if msb_first:
        shifts = np.arange(bit_count - 1, -1, -1, dtype=np.uint64)
    else:
        shifts = np.arange(0, bit_count, dtype=np.uint64)
    active = values >= level
    # Cast before shifting so each column is an unsigned integer mask.
    return np.sum(active.astype(np.uint64) << shifts, axis=1, dtype=np.uint64)


# Short aliases make the two static measurement forms discoverable without
# changing their explicit semantics.
histogram = code_density
transition_levels = transition_metrics


__all__ = [
    "spectrum",
    "adc_static",
    "code_density",
    "histogram",
    "transition_metrics",
    "transition_levels",
    "decode",
]
