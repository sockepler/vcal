"""Read Spectre binary PSF sweep results.

Spectre 23.1 must write rawfmt=psfbin on this project (the psfascii writer
segfaults — see sim/maestro_amp/run_spectre.sh). We convert psfbin ->
psfascii offline with Cadence's `psf` utility, then parse the tran "group"
layout:

    TRACE
    "group" GROUP <n>
    "VOP" "V"
    "VON" "V"
    ...
    VALUE
    "time" 0.0
    "group" <val of trace 1>
    <val of trace 2>
    ...

Returns one sweep axis and named real-scalar signal arrays.  The parser accepts
both the older ``group``/bare-value transient layout and the named ``VALUE``
rows emitted by current Spectre DC and transient analyses.
"""
import math
import os
import re
import subprocess

import numpy as np
from .simulator_env import simulation_environment

PSF_BIN = os.environ.get("OPTSERVER_PSF", "/soft/cadence/IC231/bin/psf")


def psfbin_to_ascii(binpath, outpath):
    # The Cadence utility defaults to six significant digits. That adds
    # quantization error to high-resolution ADC and small-signal measurements.
    # Seventeen significant digits preserve IEEE double round trips.
    r = subprocess.run([PSF_BIN, "-f", "%.17g", "-i", binpath, "-o", outpath],
                       capture_output=True, text=True, timeout=300,
                       env=simulation_environment())
    if r.returncode != 0 or not os.path.exists(outpath):
        raise RuntimeError("psf convert failed: %s" % (r.stderr or r.stdout))


_SECTIONS = {"HEADER", "TYPE", "SWEEP", "TRACE", "VALUE", "END"}
_NAMED_VALUE = re.compile(r'^"((?:[^"\\]|\\.)*)"\s+(.+?)\s*$')


def _named_line(line):
    match = _NAMED_VALUE.match(line.strip())
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _quoted_tokens(line):
    return re.findall(r'"((?:[^"\\]|\\.)*)"', line)


def _scalar(raw, label, path):
    """Parse one real scalar and reject vector/complex payloads explicitly."""
    raw = raw.strip()
    parts = raw.split()
    if not parts or len(parts) != 1 or raw.startswith(("(", "[")):
        raise RuntimeError("complex or non-scalar %s in %s" % (label, path))
    try:
        value = float(parts[0])
    except (TypeError, ValueError):
        raise RuntimeError("complex or non-scalar %s in %s" % (label, path)) \
            from None
    if not math.isfinite(value) and not (math.isnan(value)):
        raise RuntimeError("non-finite %s in %s" % (label, path))
    return value


def _parse_sweep_ascii(asciipath):
    """Parse one-dimensional psfascii data without dropping malformed rows."""
    section = None
    ended = False
    sweep_names = []
    trace_names = []
    value_lines = []
    in_prop = False
    with open(asciipath, errors="ignore") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            marker = line.upper()
            if marker in _SECTIONS:
                section = marker
                in_prop = False
                if marker == "END":
                    ended = True
                continue
            if ended:
                raise RuntimeError("data appears after END in %s" % asciipath)

            upper = line.upper()
            if section == "TYPE":
                if any(token in upper for token in
                       ("COMPLEX", "VECTOR", "MATRIX")):
                    raise RuntimeError("complex or multidimensional PSF in %s"
                                       % asciipath)
            elif section == "SWEEP":
                if in_prop:
                    if line == ")":
                        in_prop = False
                    continue
                tokens = _quoted_tokens(line)
                if len(tokens) >= 2 and tokens[1].lower() == "sweep":
                    sweep_names.append(tokens[0])
                if line.endswith("PROP("):
                    in_prop = True
            elif section == "TRACE":
                if in_prop:
                    if line == ")":
                        in_prop = False
                    continue
                named = _named_line(line)
                if named is None:
                    continue
                name, rest = named
                if re.match(r"^GROUP(?:\s|$)", rest, re.IGNORECASE):
                    # Old psfascii files describe a group container before
                    # listing the scalar traces.  It is not a signal itself.
                    if line.endswith("PROP("):
                        in_prop = True
                    continue
                if name in ("group", "GROUP"):
                    continue
                if name in trace_names:
                    raise RuntimeError("duplicate trace %r in %s" %
                                       (name, asciipath))
                if any(token in rest.upper() for token in
                       ("COMPLEX", "VECTOR", "MATRIX")):
                    raise RuntimeError("complex or multidimensional trace %r in %s"
                                       % (name, asciipath))
                trace_names.append(name)
                if line.endswith("PROP("):
                    in_prop = True
            elif section == "VALUE":
                value_lines.append(line)

    if not ended:
        raise RuntimeError("incomplete PSF (missing END) in %s" % asciipath)
    if len(sweep_names) > 1:
        raise RuntimeError("multiple sweep variables in %s" % asciipath)
    if not trace_names:
        raise RuntimeError("no scalar traces in %s" % asciipath)
    if not value_lines:
        raise RuntimeError("no sweep values in %s" % asciipath)

    # Current Spectre names the sweep in SWEEP; old fixtures sometimes do not
    # include that section, so infer the first named VALUE field that is not a
    # declared trace.  This keeps the reader generic for arbitrary DC source
    # names while still rejecting a row that has no identifiable axis.
    sweep_name = sweep_names[0] if sweep_names else None
    if sweep_name is None:
        for line in value_lines:
            named = _named_line(line)
            if named and named[0] not in trace_names:
                sweep_name = named[0]
                break
    if sweep_name is None:
        raise RuntimeError("no sweep variable in %s" % asciipath)
    if sweep_name in trace_names:
        raise RuntimeError("sweep variable is also a trace in %s" % asciipath)

    rows = []
    current = None
    legacy_index = 0

    def finish_row():
        nonlocal current, legacy_index
        if current is not None:
            rows.append(current)
        current = None
        legacy_index = 0

    for line in value_lines:
        named = _named_line(line)
        if named is None:
            # Bare values belong to the old group layout.  They are accepted
            # only after a named sweep value has started a row.
            if current is None:
                raise RuntimeError("incomplete VALUE row in %s" % asciipath)
            if legacy_index >= len(trace_names):
                raise RuntimeError("too many values in VALUE row in %s" %
                                   asciipath)
            name = trace_names[legacy_index]
            legacy_index += 1
            if name in current:
                raise RuntimeError("duplicate value %r in %s" %
                                   (name, asciipath))
            current[name] = _scalar(line, name, asciipath)
            continue

        name, raw_value = named
        if name == sweep_name:
            finish_row()
            current = {sweep_name: _scalar(raw_value, sweep_name, asciipath)}
            continue
        if current is None:
            raise RuntimeError("VALUE row starts before sweep in %s" % asciipath)
        if name in ("group", "GROUP") and name not in trace_names:
            if legacy_index >= len(trace_names):
                raise RuntimeError("too many group values in %s" % asciipath)
            target = trace_names[legacy_index]
            legacy_index += 1
            if target in current:
                raise RuntimeError("duplicate value %r in %s" %
                                   (target, asciipath))
            current[target] = _scalar(raw_value, target, asciipath)
            continue
        if name not in trace_names:
            raise RuntimeError("unknown trace %r in %s" % (name, asciipath))
        if name in current:
            raise RuntimeError("duplicate value %r in %s" % (name, asciipath))
        current[name] = _scalar(raw_value, name, asciipath)
    finish_row()

    if not rows:
        raise RuntimeError("no sweep points in %s" % asciipath)
    expected = {sweep_name, *trace_names}
    for index, row in enumerate(rows, 1):
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            raise RuntimeError("incomplete VALUE row %d in %s (%s)" %
                               (index, asciipath, "; ".join(detail)))

    axis = np.asarray([row[sweep_name] for row in rows], dtype=float)
    signals = {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in trace_names
    }
    if len(axis) == 0 or any(len(values) != len(axis)
                             for values in signals.values()):
        raise RuntimeError("incomplete sweep data in %s" % asciipath)
    return axis, signals


def read_sweep(asciipath):
    """Read one-dimensional psfascii data as ``(axis, {name: values})``."""
    return _parse_sweep_ascii(asciipath)


def read_tran(asciipath):
    """Compatibility wrapper for the historical transient reader."""
    return read_sweep(asciipath)


def read_tran_psfbin(binpath, workdir):
    return read_sweep_psfbin(binpath, workdir, name="tran")


def read_sweep_psfbin(binpath, workdir, name="sweep", *,
                      ascii_name=None, analysis=None):
    """Convert and read one sweep using a unique named ASCII intermediate.

    ``name`` may be an analysis label (``tran``/``dc``) or an explicit
    filename.  ``ascii_name`` and ``analysis`` are accepted as descriptive
    keyword aliases for callers that want to make the intermediate name
    explicit.
    """
    if analysis is not None:
        name = analysis
    if ascii_name is not None:
        outpath = os.path.join(workdir, os.fspath(ascii_name))
    elif os.path.basename(os.fspath(name)).endswith(".psf"):
        outpath = os.path.join(workdir, os.fspath(name))
    else:
        outpath = os.path.join(workdir, "%s_ascii.psf" % name)
    psfbin_to_ascii(binpath, outpath)
    try:
        return read_sweep(outpath)
    finally:
        # ASCII copies can be large; the parsed arrays live in the result and
        # history.  Preserve the old small-file behavior for diagnostics.
        if os.path.exists(outpath) and os.path.getsize(outpath) > 50e6:
            os.unlink(outpath)
