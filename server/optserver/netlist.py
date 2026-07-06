"""ADE-exported Spectre netlist parsing and per-device parameter rewriting.

Adapted from gmid-tool (gmid/netlist.py). Handles the Virtuoso netlister
format: `\\` line continuations, values wrapped in parens `w=(1u)`, subckt
hierarchy, `// Cell name:` headers.

Editing a device rewrites the subckt master line, i.e. all instances of
that subckt change together — same semantics as editing the schematic.
"""
import re

_NUM_SUFFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3,
               "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
               "a": 1e-18}


def parse_num(tok):
    """Spectre numeric literal (possibly '(1u)') -> float, or None."""
    tok = str(tok).strip()
    if tok.startswith("(") and tok.endswith(")"):
        tok = tok[1:-1].strip()
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                 r"([TGMKkmunpfa]?)$", tok)
    if not m:
        return None
    return float(m.group(1)) * (_NUM_SUFFIX.get(m.group(2), 1.0)
                                if m.group(2) else 1.0)


def fmt_num(v):
    """float -> spectre literal with engineering suffix."""
    for s, mul in [("", 1), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9),
                   ("p", 1e-12), ("f", 1e-15)]:
        if abs(v) >= mul * 0.9999 or mul == 1e-15:
            return "%.6g%s" % (v / mul, s)
    return "%g" % v


class Instance:
    def __init__(self, name, nodes, master, params, subckt):
        self.name = name          # e.g. M2
        self.nodes = nodes
        self.master = master      # primitive model or subckt name
        self.params = params      # dict param -> raw string token
        self.subckt = subckt      # subckt name it lives in, or None (top)

    def num(self, key, default=None):
        v = self.params.get(key)
        if v is None:
            return default
        n = parse_num(v)
        return default if n is None else n


class Netlist:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.rawlines = f.read().splitlines()
        self._reindex_lines()
        self._parse()

    def _reindex_lines(self):
        self.logical = []          # (text, first_rawline_idx, n_rawlines)
        i = 0
        while i < len(self.rawlines):
            j = i
            text = self.rawlines[i]
            while text.rstrip().endswith("\\"):
                text = text.rstrip()[:-1] + " "
                j += 1
                text += self.rawlines[j].strip()
            self.logical.append((text, i, j - i + 1))
            i = j + 1

    _KEYWORDS = {"simulator", "global", "include", "ahdl_include",
                 "parameters", "subckt", "ends", "model", "options",
                 "save", "ic", "nodeset", "real", "if", "section",
                 "library", "endlibrary", "statistics", "simulatorOptions",
                 "modelParameter", "element", "outputParameter",
                 "designParamVals", "primitive", "subcktParameter",
                 "saveOptions", "sens", "info", "tran", "dc", "ac"}

    def _parse(self):
        self.subckts = {}          # name -> {ports, instances{name: Instance}}
        self.top_instances = {}
        cur = None
        for text, li, nl in self.logical:
            s = text.strip()
            if s.startswith("//") or not s:
                continue
            m = re.match(r"subckt\s+(\S+)\s+(.*)$", s)
            if m:
                cur = m.group(1)
                self.subckts[cur] = dict(ports=m.group(2).split(),
                                         instances={})
                continue
            if re.match(r"ends\b", s):
                cur = None
                continue
            inst = self._parse_instance(s, li, cur)
            if inst is not None:
                if cur:
                    self.subckts[cur]["instances"][inst.name] = inst
                else:
                    self.top_instances[inst.name] = inst

    def _parse_instance(self, s, li, cur):
        m = re.match(r"([A-Za-z_][\w.]*)\s*\(([^)]*)\)\s*(\S+)\s*(.*)$", s)
        if not m:
            return None
        name, nodes, master, rest = (m.group(1), m.group(2).split(),
                                     m.group(3), m.group(4))
        if name in self._KEYWORDS:
            return None
        params = {}
        for pm in re.finditer(r"([\w.]+)\s*=\s*(\([^)]*\)|\S+)", rest):
            params[pm.group(1)] = pm.group(2)
        inst = Instance(name, nodes, master, params, cur)
        inst._line = li
        return inst

    def instance(self, scope, name):
        """scope: subckt name, or None/'top' for the top-level section."""
        if scope in (None, "", "top"):
            inst = self.top_instances.get(name)
        else:
            sub = self.subckts.get(scope)
            inst = sub["instances"].get(name) if sub else None
        return inst

    # ---------- editing ----------
    _SCALE_W = ("as", "ad")        # scale proportionally with w
    _OFFSET_W = ("ps", "pd")       # add delta-w (2 sides of perimeter)
    _INV_W = ("nrd", "nrs")        # scale with 1/w

    def _logical_for(self, inst):
        for t, i, n in self.logical:
            if i == inst._line:
                return t, n
        raise KeyError("instance line not found: %s" % inst.name)

    def set_attr(self, inst, attr, value, integer=False):
        """Rewrite one attribute on an instance line. For MOS w, the
        layout-derived params (as/ad/ps/pd/nrd/nrs) are rescaled too."""
        text, nl = self._logical_for(inst)

        def repl(key, val, as_int=False):
            nonlocal text
            fv = ("%d" % int(round(val))) if as_int else fmt_num(val)
            pat = re.compile(r"\b(%s)\s*=\s*(\([^)]*\)|\S+)" % re.escape(key))
            if pat.search(text):
                text = pat.sub("%s=%s" % (key, fv), text, count=1)
            else:
                text = text.rstrip() + " %s=%s" % (key, fv)
            inst.params[key] = fv

        if attr == "w":
            w_old = inst.num("w")
            repl("w", value)
            if w_old:
                k = value / w_old
                for p in self._SCALE_W:
                    v = inst.num(p)
                    if v is not None:
                        repl(p, v * k)
                for p in self._OFFSET_W:
                    v = inst.num(p)
                    if v is not None:
                        repl(p, v + (value - w_old))
                for p in self._INV_W:
                    v = inst.num(p)
                    if v is not None:
                        repl(p, v / k)
        else:
            repl(attr, value, as_int=integer or attr in ("mr", "m", "nf"))

        self.rawlines[inst._line:inst._line + nl] = [text]
        self._reindex_lines()
        self._parse()

    def save(self, path=None):
        with open(path or self.path, "w") as f:
            f.write("\n".join(self.rawlines) + "\n")
