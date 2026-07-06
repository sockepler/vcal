"""Read HSpice ASCII transient output (.tr0 with .option post=2).

Format: header lines (counts + title + date), then signal type codes and
names, then whitespace-packed float columns, terminated by 1e30 sentinel.
Values are fixed-width 11-char fields (ingold=2 gives E-notation).

Signal name mapping for backend-neutral metrics:
    v(vop)   -> 'vop'          (Waves.V('VOP') matches case-insensitively)
    i(v_vdd) -> 'v_vdd:p'      (Waves.I('V_VDD') convention)
"""
import re

import numpy as np


def read_tr0(path):
    with open(path, errors="ignore") as f:
        txt = f.read()

    # --- header: everything up to the last signal name ---
    # names appear after the type-code block: TIME then v(...)/i(...) etc.
    m = re.search(r"\$&%#", txt)
    if not m:
        raise RuntimeError("no $&%# header terminator in %s" % path)
    header = txt[:m.start()]
    data_txt = txt[m.end():]
    # discard padding to the end of the terminator line
    nl = data_txt.find("\n")
    if nl >= 0:
        data_txt = data_txt[nl + 1:]

    # names are space-padded 16-char fields, truncated at 15 chars and
    # possibly missing the closing paren: "v(x1.vcmd      "
    names = ["time"]
    for nm in re.finditer(r"([vi])\(\s*([^)\s]+)\)?", header, re.I):
        kind, sig = nm.group(1).lower(), nm.group(2).lower().rstrip(")")
        names.append(sig if kind == "v" else sig + ":p")

    # --- data: continuous stream of fixed-width 13-char float fields,
    # wrapped at arbitrary line ends ---
    stream = "".join(ln for ln in data_txt.splitlines())
    flat = []
    for k in range(0, len(stream) - 12, 13):
        tok = stream[k:k + 13]
        try:
            flat.append(float(tok))
        except ValueError:
            break
    arr = np.asarray(flat)
    # strip the 1e30 terminator
    end = np.nonzero(arr >= 1e29)[0]
    if len(end):
        arr = arr[:end[0]]
    ncol = len(names)
    if ncol == 0 or len(arr) < ncol:
        raise RuntimeError("tr0 parse failed: %d names, %d values"
                           % (ncol, len(arr)))
    nrow = len(arr) // ncol
    arr = arr[:nrow * ncol].reshape(nrow, ncol)
    t = arr[:, 0]
    sigs = {names[k]: arr[:, k] for k in range(1, ncol)}
    return t, sigs
