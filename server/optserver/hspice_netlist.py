"""HSpice-format netlist parsing and per-device parameter rewriting.

Handles the format produced by this project's HSpice exports (top_amp.sp):
  - `+` leading-continuation lines
  - `.subckt NAME ports` / `.ends`
  - instances: `xM7 net189 Vcmd GND GND n18_ckt mr=2 w=10u l=1u ...`
  - positional-value elements: `C7 net179 X 50f`, `I4 VDD Vbn 200u`,
    `R1 a b 1k` — edited via attr "value".

Same editing semantics as the Spectre editor: rewriting a device inside a
subckt changes every instance of that subckt; MOS w edits rescale the
layout-derived as/ad/ps/pd/nrd/nrs parameters.
"""
import re

_NUM_SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "x": 1e6, "k": 1e3,
               "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
               "a": 1e-18}


def parse_num(tok):
    tok = str(tok).strip().strip("'\"")
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                 r"(meg|MEG|[TtGgKkMmUuNnPpFfAaXx])?$", tok)
    if not m:
        return None
    suf = (m.group(2) or "").lower()
    return float(m.group(1)) * _NUM_SUFFIX.get(suf, 1.0)


def fmt_num(v):
    for s, mul in [("", 1), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9),
                   ("p", 1e-12), ("f", 1e-15)]:
        if abs(v) >= mul * 0.9999 or mul == 1e-15:
            return "%.6g%s" % (v / mul, s)
    return "%g" % v


class Instance:
    def __init__(self, name, master, params, subckt, ntokens):
        self.name = name
        self.master = master
        self.params = params        # named k=v params
        self.subckt = subckt
        self.ntokens = ntokens      # positional tokens (nodes + value)

    def num(self, key, default=None):
        if key == "value":
            v = self.ntokens[-1] if self.ntokens else None
        else:
            v = self.params.get(key)
        if v is None:
            return default
        n = parse_num(v)
        return default if n is None else n


class HspiceNetlist:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.rawlines = f.read().splitlines()
        self._reindex()

    def _reindex(self):
        # join `+` continuations into logical lines
        self.logical = []
        i = 0
        n = len(self.rawlines)
        while i < n:
            text = self.rawlines[i]
            j = i + 1
            while j < n and self.rawlines[j].lstrip().startswith("+"):
                text = text.rstrip() + " " + self.rawlines[j].lstrip()[1:]
                j += 1
            self.logical.append((text, i, j - i))
            i = j
        self._parse()

    def _parse(self):
        self.subckts = {}
        self.top_instances = {}
        cur = None
        for text, li, nl in self.logical:
            s = text.strip()
            if not s or s.startswith("*") or s.startswith("$"):
                continue
            low = s.lower()
            if low.startswith(".subckt"):
                toks = s.split()
                cur = toks[1]
                self.subckts[cur] = dict(ports=toks[2:], instances={})
                continue
            if low.startswith(".ends"):
                cur = None
                continue
            if s.startswith("."):
                continue
            inst = self._parse_instance(s, li, cur)
            if inst is not None:
                target = (self.subckts[cur]["instances"] if cur
                          else self.top_instances)
                target[inst.name] = inst

    def _parse_instance(self, s, li, cur):
        toks = s.split()
        if len(toks) < 3 or not re.match(r"^[A-Za-z]", toks[0]):
            return None
        name = toks[0]
        named, positional = {}, []
        for t in toks[1:]:
            m = re.match(r"^([\w.]+)=(.+)$", t)
            if m:
                named[m.group(1)] = m.group(2)
            else:
                positional.append(t)
        # master: for x-instances, last positional token that is not a
        # number; for primitives (R/C/L/I/V/E/G...) the master is implicit
        master = None
        if name[0] in "xX" and positional:
            master = positional[-1]
        inst = Instance(name, master, named, cur, positional)
        inst._line = li
        return inst

    def instance(self, scope, name):
        if scope in (None, "", "top"):
            pool = self.top_instances
        else:
            sub = self.subckts.get(scope)
            pool = sub["instances"] if sub else {}
        inst = pool.get(name)
        if inst is None:  # tolerate missing/extra x prefix in YAML
            for alt in ("x" + name, name.lstrip("xX"), "X" + name):
                if alt in pool:
                    return pool[alt]
        return inst

    _SCALE_W = ("as", "ad")
    _OFFSET_W = ("ps", "pd")
    _INV_W = ("nrd", "nrs")

    def set_attr(self, inst, attr, value, integer=False):
        text, li, nl = None, inst._line, None
        for t, i, n in self.logical:
            if i == inst._line:
                text, nl = t, n
                break

        def repl_named(key, val, as_int=False):
            nonlocal text
            fv = ("%d" % int(round(val))) if as_int else fmt_num(val)
            pat = re.compile(r"(?<=\s)%s=\S+" % re.escape(key))
            if pat.search(text):
                text = pat.sub("%s=%s" % (key, fv), text, count=1)
            else:
                text = text.rstrip() + " %s=%s" % (key, fv)
            inst.params[key] = fv

        if attr == "value":
            # last positional numeric token (C/I/R element value)
            toks = text.split()
            for k in range(len(toks) - 1, 0, -1):
                if "=" not in toks[k] and parse_num(toks[k]) is not None:
                    toks[k] = fmt_num(value)
                    break
            else:
                raise ValueError("no positional value on %s" % inst.name)
            text = " ".join(toks)
        elif attr == "w":
            w_old = inst.num("w")
            repl_named("w", value)
            if w_old:
                k = value / w_old
                for p in self._SCALE_W:
                    v = inst.num(p)
                    if v is not None:
                        repl_named(p, v * k)
                for p in self._OFFSET_W:
                    v = inst.num(p)
                    if v is not None:
                        repl_named(p, v + (value - w_old))
                for p in self._INV_W:
                    v = inst.num(p)
                    if v is not None:
                        repl_named(p, v / k)
        else:
            repl_named(attr, value,
                       as_int=integer or attr in ("mr", "m", "nf"))

        self.rawlines[li:li + nl] = [text]
        self._reindex()

    def save(self, path=None):
        with open(path or self.path, "w") as f:
            f.write("\n".join(self.rawlines) + "\n")


def extract_subckts(src_path, top_cell):
    """Pull `top_cell` and its transitive subckt deps out of a bigger
    HSpice file. Returns the extracted text."""
    with open(src_path) as f:
        lines = f.read().splitlines()
    blocks = {}   # name -> [lines]
    cur = None
    for ln in lines:
        s = ln.strip().lower()
        if s.startswith(".subckt"):
            cur = ln.split()[1]
            blocks[cur] = []
        if cur is not None:
            blocks[cur].append(ln)
        if s.startswith(".ends"):
            cur = None
    if top_cell not in blocks:
        raise ValueError("%s not found in %s" % (top_cell, src_path))

    need, order, stack = set(), [], [top_cell]
    while stack:
        cell = stack.pop()
        if cell in need or cell not in blocks:
            continue
        need.add(cell)
        body = "\n".join(blocks[cell])
        for m in re.finditer(r"^\s*[xX]\S*\s+(.*)$", body, re.M):
            toks = m.group(1).split()
            positional = [t for t in toks if "=" not in t]
            if positional and positional[-1] in blocks:
                stack.append(positional[-1])
        order.append(cell)

    out = ["* extracted from %s (top=%s)" % (src_path, top_cell), ""]
    for cell in reversed(order):        # deps first
        out.extend(blocks[cell])
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="extract a subckt (+deps) from an HSpice netlist")
    ap.add_argument("src")
    ap.add_argument("cell")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    txt = extract_subckts(a.src, a.cell)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print("wrote %s" % a.out)
