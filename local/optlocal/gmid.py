"""Standalone gm/Id lookup-table adapter.

The adapter deliberately has no dependency on the original VLUT/gmid-tool
package.  It accepts the compact ``.npz`` files emitted by that tool and a
small flat CSV format suitable for synthetic examples or other local tools.
All voltages and geometry values use SI units (V and m); currents are A,
conductances S, and capacitances F.

The source LUTs use magnitude conventions for both transistor polarities.
For a PMOS, ``vgs``, ``vds``, ``vsb``, ``ids`` and the returned operating-point
values therefore remain positive magnitudes and ``polarity()`` reports ``p``.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


_AXES = ("L", "VSB", "VDS", "VGS")
_REQUIRED = _AXES + ("W", "polarity", "ids", "gm")
_META = set(_AXES) | {"W", "polarity", "temp", "temperature", "vmax"}
_EXTENSIVE = {
    "ids",
    "gm",
    "gds",
    "gmbs",
    "cgg",
    "cgs",
    "cgd",
    "cgb",
    "cdd",
    "css",
}


def _error(message: str) -> ValueError:
    """Return a consistently plain-English validation error."""

    return ValueError(message)


def _as_scalar(value, name: str):
    """Extract a scalar from an NPZ value without permitting object arrays."""

    arr = np.asarray(value)
    if arr.shape != ():
        raise _error("%s must be a scalar" % name)
    return arr.item()


def _numeric_scalar(value, name: str, *, positive: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise _error("%s must be a finite number" % name) from None
    if not math.isfinite(out):
        raise _error("%s must be a finite number" % name)
    if positive and out <= 0.0:
        raise _error("%s must be positive" % name)
    return out


def _axis(values, name: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise _error("%s must be a one-dimensional numeric array" % name) from None
    if out.ndim != 1 or out.size == 0:
        raise _error("%s must be a non-empty one-dimensional array" % name)
    if not np.isfinite(out).all():
        raise _error("%s must contain only finite values" % name)
    if out.size > 1 and not np.all(np.diff(out) > 0.0):
        raise _error("%s must be strictly increasing" % name)
    return out


def _polarity(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", "strict")
    value = str(value).strip().lower()
    if value not in ("n", "p"):
        raise _error("polarity must be 'n' or 'p'")
    return value


def _temperature(value, name: str = "temperature") -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _numeric_scalar(value, name)


def _row_float(value, name: str, *, allow_blank: bool = True) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_blank:
            return float("nan")
        raise _error("%s is required" % name)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise _error("%s must be numeric" % name) from None


def _validate_data(name: str, values: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise _error("%s must be numeric" % name) from None
    if out.shape != shape:
        raise _error("%s shape %s does not match %s" % (name, out.shape, shape))
    # NaN is an intentional representation of an unavailable operating point
    # and is handled by the inverse lookup.  Infinite values are malformed.
    if np.isinf(out).any():
        raise _error("%s must not contain infinite values" % name)
    return out


def _clean_header(header: Sequence[str]) -> list:
    out = [str(item).strip() for item in header]
    if any(not item for item in out):
        raise _error("CSV header contains an empty field name")
    if len(set(out)) != len(out):
        raise _error("CSV header contains duplicate field names")
    return out


def _active_csv_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield line


class GmIdTable:
    """Validated gm/Id table with strict interpolation and sizing helpers."""

    def __init__(
        self,
        axes: Mapping[str, np.ndarray],
        data: Mapping[str, np.ndarray],
        width: float,
        polarity: str,
        temperature: Optional[float],
        source: str,
    ):
        self._axes = {name: _axis(axes[name], name) for name in _AXES}
        shape = tuple(self._axes[name].size for name in _AXES)
        self._data = {
            name: _validate_data(name, values, shape)
            for name, values in data.items()
        }
        for required in ("ids", "gm"):
            if required not in self._data:
                raise _error("LUT is missing required field %s" % required)
        self._width = _numeric_scalar(width, "W", positive=True)
        self._polarity = _polarity(polarity)
        self._temperature = _temperature(temperature)
        self._source = str(source)

    @classmethod
    def load(cls, path) -> "GmIdTable":
        """Load a compatible ``.npz`` or flat ``.csv`` table."""

        source = os.fspath(path)
        suffix = Path(source).suffix.lower()
        if suffix == ".npz":
            return cls._load_npz(source)
        if suffix == ".csv":
            return cls._load_csv(source)
        raise _error("unsupported LUT format: expected .npz or .csv")

    @classmethod
    def _load_npz(cls, path: str) -> "GmIdTable":
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise _error("cannot load NPZ LUT %s: %s" % (path, exc)) from exc
        try:
            keys = set(archive.files)
            missing = [name for name in _REQUIRED if name not in keys]
            if missing:
                raise _error("NPZ LUT is missing required field(s): %s"
                             % ", ".join(missing))
            axes = {name: archive[name] for name in _AXES}
            shape = tuple(np.asarray(axes[name]).size for name in _AXES)
            expected = tuple(shape)
            data = {}
            for name in sorted(keys - _META):
                values = archive[name]
                if np.asarray(values).ndim != 4:
                    raise _error("NPZ data field %s must be four-dimensional" % name)
                data[name] = _validate_data(name, values, expected)
            width = _as_scalar(archive["W"], "W")
            polarity = _as_scalar(archive["polarity"], "polarity")
            temp = _as_scalar(archive["temp"], "temp") if "temp" in keys else None
            if "temperature" in keys:
                alt = _as_scalar(archive["temperature"], "temperature")
                if temp is not None and _temperature(temp) != _temperature(alt):
                    raise _error("temp and temperature disagree")
                temp = alt
        finally:
            archive.close()
        return cls(axes, data, width, polarity, temp, path)

    @classmethod
    def _load_csv(cls, path: str) -> "GmIdTable":
        rows = list(csv.reader(_active_csv_lines(path)))
        if not rows:
            raise _error("CSV LUT is empty")
        header = _clean_header(rows[0])
        missing = [name for name in _REQUIRED if name not in header]
        if missing:
            raise _error("CSV LUT is missing required field(s): %s"
                         % ", ".join(missing))
        values = []
        for line_no, row in enumerate(rows[1:], 2):
            if len(row) != len(header):
                raise _error("CSV row %d has %d fields; expected %d"
                             % (line_no, len(row), len(header)))
            item = dict(zip(header, row))
            coords = tuple(_row_float(item[name], name, allow_blank=False)
                           for name in _AXES)
            width = _row_float(item["W"], "W", allow_blank=False)
            polarity = str(item["polarity"]).strip().lower()
            if not polarity:
                raise _error("polarity is required on CSV row %d" % line_no)
            temp_values = []
            for name in ("temp", "temperature"):
                if name in item and item[name].strip():
                    temp_values.append(_row_float(item[name], name,
                                                  allow_blank=False))
            if len(temp_values) == 2 and temp_values[0] != temp_values[1]:
                raise _error("CSV row %d has disagreeing temp fields" % line_no)
            temp = temp_values[0] if temp_values else None
            row_data = {
                name: _row_float(item[name], name)
                for name in header
                if name not in _META
            }
            values.append((coords, width, polarity, temp, row_data))
        if not values:
            raise _error("CSV LUT has no data rows")

        # The metadata is deliberately checked before constructing arrays so a
        # malformed table cannot silently collapse into a valid-looking grid.
        first_width = _numeric_scalar(values[0][1], "W", positive=True)
        first_polarity = _polarity(values[0][2])
        first_temp = _temperature(values[0][3])
        for _, width, polarity, temp, _ in values:
            if not math.isfinite(width) or width != first_width:
                raise _error("CSV W must be finite, positive, and constant")
            if _polarity(polarity) != first_polarity:
                raise _error("CSV polarity must be constant")
            if _temperature(temp) != first_temp:
                raise _error("CSV temperature must be constant")

        axis_values = {
            name: np.asarray(sorted({row[0][i] for row in values}), dtype=np.float64)
            for i, name in enumerate(_AXES)
        }
        for name in _AXES:
            _axis(axis_values[name], name)
        shape = tuple(axis_values[name].size for name in _AXES)
        expected_count = int(np.prod(shape, dtype=np.int64))
        if len(values) != expected_count:
            raise _error("CSV LUT is missing grid points")

        index = {
            name: {float(value): i for i, value in enumerate(axis_values[name])}
            for name in _AXES
        }
        data_names = sorted({name for _, _, _, _, row in values for name in row})
        data = {name: np.full(shape, np.nan, dtype=np.float64)
                for name in data_names}
        seen = set()
        for coords, _, _, _, row_data in values:
            key = tuple(float(value) for value in coords)
            if key in seen:
                raise _error("CSV LUT contains duplicate grid point")
            seen.add(key)
            loc = tuple(index[name][key[i]] for i, name in enumerate(_AXES))
            for name in data_names:
                # A blank optional value is a valid NaN hole.  A missing column
                # is impossible because all rows share one validated header.
                data[name][loc] = row_data.get(name, float("nan"))
        if len(seen) != expected_count:
            raise _error("CSV LUT is missing grid points")
        return cls(axis_values, data, first_width, first_polarity,
                   first_temp, path)

    @property
    def axes(self) -> Dict[str, np.ndarray]:
        """Return copies of the four SI-unit axes."""

        return {name: values.copy() for name, values in self._axes.items()}

    def width(self) -> float:
        return float(self._width)

    def polarity(self) -> str:
        return self._polarity

    def temperature(self) -> Optional[float]:
        return None if self._temperature is None else float(self._temperature)

    def source(self) -> str:
        return self._source

    def _check_coordinate(self, name: str, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise _error("%s must be finite" % name) from None
        if not math.isfinite(value):
            raise _error("%s must be finite" % name)
        grid = self._axes[name]
        if value < grid[0] or value > grid[-1]:
            raise _error("%s=%g is outside LUT range [%g, %g]"
                         % (name, value, grid[0], grid[-1]))
        return value

    @staticmethod
    def _interp_axis(values: np.ndarray, grid: np.ndarray, value: float,
                     axis: int) -> np.ndarray:
        if value == grid[0]:
            return np.take(values, 0, axis=axis)
        if value == grid[-1]:
            return np.take(values, -1, axis=axis)
        hi = int(np.searchsorted(grid, value, side="left"))
        if hi < grid.size and value == grid[hi]:
            # Do not form low * 1 + high * 0 at an exact internal node:
            # 0 * NaN would contaminate a valid node next to a NaN slice.
            return np.take(values, hi, axis=axis)
        lo = hi - 1
        fraction = (value - grid[lo]) / (grid[hi] - grid[lo])
        low = np.take(values, lo, axis=axis)
        high = np.take(values, hi, axis=axis)
        return low * (1.0 - fraction) + high * fraction

    def _slice(self, length: float, vds: float, vsb: float):
        length = self._check_coordinate("L", length)
        vsb = self._check_coordinate("VSB", vsb)
        vds = self._check_coordinate("VDS", vds)
        out = {}
        for name, values in self._data.items():
            result = self._interp_axis(values, self._axes["L"], length, 0)
            result = self._interp_axis(result, self._axes["VSB"], vsb, 0)
            result = self._interp_axis(result, self._axes["VDS"], vds, 0)
            out[name] = np.asarray(result, dtype=np.float64)
        return out

    @staticmethod
    def _ratio(gm: np.ndarray, ids: np.ndarray) -> np.ndarray:
        out = np.full(np.shape(gm), np.nan, dtype=np.float64)
        valid = np.isfinite(gm) & np.isfinite(ids) & (ids != 0.0)
        np.divide(gm, ids, out=out, where=valid)
        return out

    def curve(self, length: float, vds: float, vsb: float = 0.0):
        """Return the VGS curve at one ``L/VDS/VSB`` point."""

        sliced = self._slice(length, vds, vsb)
        ids = sliced["ids"]
        gm = sliced["gm"]
        result = {
            "vgs": self._axes["VGS"].copy(),
            "gmid": self._ratio(gm, ids),
            "idw": ids / self._width,
            "ids": ids.copy(),
            "gm": gm.copy(),
        }
        if "gds" in sliced:
            result["gain"] = self._ratio(gm, sliced["gds"])
        if "cgg" in sliced:
            cgg = sliced["cgg"]
            result["ft"] = self._safe_divide(gm, 2.0 * math.pi * cgg)
        return result

    @staticmethod
    def _safe_divide(numerator, denominator):
        out = np.full(np.shape(numerator), np.nan, dtype=np.float64)
        valid = (np.isfinite(numerator) & np.isfinite(denominator)
                 & (denominator != 0.0))
        np.divide(numerator, denominator, out=out, where=valid)
        return out

    def _inverse_vgs(self, sliced: Mapping[str, np.ndarray], target: float):
        """Solve gm/ids on adjacent valid VGS samples only.

        Solving the ratio equation from linearly interpolated ``gm`` and
        ``ids`` avoids the bias caused by first interpolating a precomputed
        ratio.  A NaN between two valid samples breaks adjacency and therefore
        cannot be crossed.
        """

        gms = np.asarray(sliced["gm"], dtype=np.float64)
        ids = np.asarray(sliced["ids"], dtype=np.float64)
        vgs = self._axes["VGS"]
        ratios = self._ratio(gms, ids)
        candidates = []
        endpoint_ratios = []
        flat_tolerance = 1e-12 * max(1.0, abs(target))
        for i in range(vgs.size - 1):
            g0, g1 = gms[i], gms[i + 1]
            id0, id1 = ids[i], ids[i + 1]
            r0, r1 = ratios[i], ratios[i + 1]
            if not (math.isfinite(r0) and math.isfinite(r1)
                    and math.isfinite(g0) and math.isfinite(g1)
                    and math.isfinite(id0) and math.isfinite(id1)):
                continue
            endpoint_ratios.extend((float(r0), float(r1)))
            if target < min(r0, r1) - flat_tolerance \
                    or target > max(r0, r1) + flat_tolerance:
                continue
            denominator = (g1 - g0) - target * (id1 - id0)
            numerator = target * id0 - g0
            denominator_scale = max(abs(g0), abs(g1), abs(target * id0),
                                    abs(target * id1))
            denominator_tolerance = 1e-12 * denominator_scale
            if abs(denominator) <= denominator_tolerance:
                # A ratio that is exactly flat at the target has infinitely
                # many answers in this segment and is not a safe sizing point.
                if abs(r0 - target) <= flat_tolerance \
                        and abs(r1 - target) <= flat_tolerance:
                    raise _error("gm/id target has a non-unique VGS solution")
                continue
            t = numerator / denominator
            if t < -1e-10 or t > 1.0 + 1e-10:
                continue
            t = min(1.0, max(0.0, float(t)))
            gm_at = g0 + t * (g1 - g0)
            ids_at = id0 + t * (id1 - id0)
            ratio_at = gm_at / ids_at if ids_at != 0.0 else float("nan")
            if not math.isfinite(ratio_at) or not math.isclose(
                    ratio_at, target, rel_tol=2e-10,
                    abs_tol=2e-12 * max(1.0, abs(target))):
                raise _error("failed to solve gm/id at an adjacent VGS pair")
            candidates.append((float(vgs[i] + t * (vgs[i + 1] - vgs[i])),
                               i, t, gm_at, ids_at))

        if not candidates:
            if endpoint_ratios:
                lo = min(endpoint_ratios)
                hi = max(endpoint_ratios)
                if target < lo - flat_tolerance or target > hi + flat_tolerance:
                    raise _error("gmid target is outside valid range [%g, %g]"
                                 % (lo, hi))
            raise _error("gmid target has no adjacent valid VGS solution")

        candidates.sort(key=lambda item: item[0])
        unique = []
        vgs_tolerance = 1e-10 * max(1.0, abs(vgs[0]), abs(vgs[-1]))
        for candidate in candidates:
            if not unique or abs(candidate[0] - unique[-1][0]) > vgs_tolerance:
                unique.append(candidate)
        if len(unique) > 1:
            raise _error("gmid target has multiple distinct VGS solutions")
        return unique[0]

    @staticmethod
    def _json_float(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise _error("sizing result contains a non-finite value")
        return value

    def size_for(
        self,
        length: float,
        vds: float,
        vsb: float = 0.0,
        *,
        gmid: float,
        ids: Optional[float] = None,
        gm: Optional[float] = None,
    ) -> dict:
        """Return a width-scaled operating point at a target gm/Id."""

        target_gmid = _numeric_scalar(gmid, "gmid")
        if target_gmid <= 0.0:
            raise _error("gmid must be positive")
        if (ids is None) == (gm is None):
            raise _error("provide exactly one of ids or gm")
        target_ids = (_numeric_scalar(ids, "ids", positive=True)
                      if ids is not None else None)
        target_gm = (_numeric_scalar(gm, "gm", positive=True)
                     if gm is not None else None)

        sliced = self._slice(length, vds, vsb)
        vgs, segment_index, fraction, gm_char, ids_char = self._inverse_vgs(
            sliced, target_gmid)
        if not (math.isfinite(gm_char) and math.isfinite(ids_char)
                and ids_char != 0.0):
            raise _error("gm/id solution has invalid operating-point values")
        scale_target = target_gm if target_gm is not None else target_ids
        scale_base = gm_char if target_gm is not None else ids_char
        if scale_base == 0.0:
            raise _error("cannot scale a zero operating-point value")
        scale = float(scale_target / scale_base)
        width = self._width * scale
        if not math.isfinite(width) or width <= 0.0:
            raise _error("scaled W must be positive and finite")

        raw_at = {}
        # Interpolate every field on the exact gm/ids segment.  Retaining the
        # segment from the inverse solve matters at a valid endpoint followed
        # by a NaN gap: choosing the segment by VGS alone could cross that gap.
        t = fraction
        for name, values in sliced.items():
            a, b = values[segment_index], values[segment_index + 1]
            if t == 0.0:
                raw_at[name] = float(a)
            elif t == 1.0:
                raw_at[name] = float(b)
            else:
                raw_at[name] = float(a + t * (b - a))
        ids_out = ids_char * scale
        gm_out = gm_char * scale
        result = {
            "W": self._json_float(width),
            "L": self._json_float(length),
            "vgs": self._json_float(vgs),
            "gmid": self._json_float(gm_out / ids_out),
            "ids": self._json_float(ids_out),
            "gm": self._json_float(gm_out),
            "idw": self._json_float(ids_out / width),
            "polarity": self._polarity,
        }
        if "gds" in raw_at and math.isfinite(raw_at["gds"]):
            gds_out = raw_at["gds"] * scale
            result["gds"] = self._json_float(gds_out)
            result["gain"] = self._json_float(gm_out / gds_out) \
                if gds_out != 0.0 else None
        if "cgg" in raw_at and math.isfinite(raw_at["cgg"]):
            cgg_out = raw_at["cgg"] * scale
            result["cgg"] = self._json_float(cgg_out)
            result["ft"] = self._json_float(gm_out / (2.0 * math.pi * cgg_out)) \
                if cgg_out != 0.0 else None
        for name, value in raw_at.items():
            if name in ("ids", "gm", "gds", "cgg"):
                continue
            if not math.isfinite(value):
                continue
            if name in _EXTENSIVE:
                result[name] = self._json_float(value * scale)
            elif name not in ("vgs",):
                result[name] = self._json_float(value)
        # Keep the requested target visible while retaining the exact ratio
        # computed from the scaled operating point.  They should be equal to
        # floating-point precision because the inverse solve used gm/ids.
        if target_gm is not None:
            result["gm"] = self._json_float(target_gm)
            result["ids"] = self._json_float(target_gm / target_gmid)
            result["gmid"] = self._json_float(result["gm"] / result["ids"])
            result["idw"] = self._json_float(result["ids"] / result["W"])
        else:
            result["ids"] = self._json_float(target_ids)
            result["gm"] = self._json_float(target_ids * target_gmid)
            result["gmid"] = self._json_float(result["gm"] / result["ids"])
            result["idw"] = self._json_float(result["ids"] / result["W"])
        return result
