"""Read Spectre binary PSF transient results.

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

Returns time and named signal arrays.
"""
import os
import re
import subprocess

import numpy as np

PSF_BIN = os.environ.get("OPTSERVER_PSF", "/soft/cadence/IC231/bin/psf")


def psfbin_to_ascii(binpath, outpath):
    r = subprocess.run([PSF_BIN, "-i", binpath, "-o", outpath],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(outpath):
        raise RuntimeError("psf convert failed: %s" % (r.stderr or r.stdout))


def read_tran(asciipath):
    """psfascii tran file -> (time ndarray, {name: ndarray})."""
    names = []          # trace order inside the group
    time = []
    cols = None
    section = None
    in_prop = False     # inside a PROP( ... ) attribute block
    vidx = 0            # which trace the next bare VALUE line belongs to
    with open(asciipath, errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s in ("HEADER", "TYPE", "SWEEP", "TRACE", "VALUE", "END"):
                section = s
                in_prop = False
                continue
            if section == "TRACE":
                if in_prop:
                    if s == ")":
                        in_prop = False
                    continue
                m = re.match(r'^"([^"]+)"', s)
                if m and m.group(1) != "group" and "GROUP" not in s:
                    names.append(m.group(1))
                if s.endswith("PROP("):
                    in_prop = True
            elif section == "VALUE":
                if cols is None:
                    cols = {n: [] for n in names}
                m = re.match(r'^"time"\s+(\S+)', s)
                if m:
                    time.append(float(m.group(1)))
                    vidx = 0
                    continue
                m = re.match(r'^"group"\s+(\S+)', s)
                if m:
                    cols[names[vidx]].append(float(m.group(1)))
                    vidx += 1
                    continue
                # bare value line
                try:
                    v = float(s)
                except ValueError:
                    continue
                if vidx < len(names):
                    cols[names[vidx]].append(v)
                    vidx += 1
    if not time:
        raise RuntimeError("no tran sweep points in %s" % asciipath)
    t = np.asarray(time)
    sigs = {}
    for n, v in (cols or {}).items():
        a = np.asarray(v)
        if len(a) == len(t):
            sigs[n] = a
    return t, sigs


def read_tran_psfbin(binpath, workdir):
    outpath = os.path.join(workdir, "tran_ascii.psf")
    psfbin_to_ascii(binpath, outpath)
    try:
        return read_tran(outpath)
    finally:
        # ascii copy can be big; results live in history.jsonl anyway
        if os.path.exists(outpath) and os.path.getsize(outpath) > 50e6:
            os.unlink(outpath)
