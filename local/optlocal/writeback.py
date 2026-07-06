"""Write optimized sizes back to Virtuoso schematics via virtuoso-bridge.

Mapping:
  YAML device "ampGA/M2"  -> lib/<scope-cell>/schematic instance M2
  YAML device "top/M5"    -> lib/<top_cell>/schematic instance M5
  netlist 'x' prefixes (xM2) are stripped for the schematic name.

Property names: attr w/l/mr map 1:1 to instance CDF props. attr "value"
needs an explicit `sch_prop` on the param (e.g. dc for isource, c for a
capacitor) — params without it are reported as skipped, not written.

The YAML may carry:
  virtuoso: {lib: YourLib, top_cell: your_top_cell}
"""
import json
import socket

from optserver.hspice_netlist import fmt_num

BRIDGE_ADDR = ("127.0.0.1", 65036)


def skill_exec(code, timeout=60):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout + 10)
    s.connect(BRIDGE_ADDR)
    s.sendall(json.dumps({"skill": code, "timeout": timeout})
              .encode("utf-8"))
    s.shutdown(socket.SHUT_WR)
    resp = b""
    while True:
        d = s.recv(65536)
        if not d:
            break
        resp += d
    s.close()
    if resp and resp[0:1] == b"\x02":
        return True, resp[1:].decode("utf-8", "replace").strip()
    return False, resp[1:].decode("utf-8", "replace")


def bridge_alive():
    try:
        ok, _ = skill_exec("1+1", timeout=5)
        return ok
    except OSError:
        return False


def plan(cfg, param_specs, values):
    """[(cell, inst, prop, new_value_str, param_name)] + skipped list."""
    vcfg = cfg.get("virtuoso", {})
    top_cell = vcfg.get("top_cell", cfg.get("name"))
    edits, skipped = [], []
    for p in param_specs:
        name = p["name"]
        if name not in values:
            continue
        attr = p["attr"]
        if attr in ("w", "l", "mr", "m", "nf", "wr", "lr"):
            # Many PDKs: the user-visible multiplier param is "m"
            # (netlisted as mr); simM/multi are derived props.
            prop = p.get("sch_prop", "m" if attr == "mr" else attr)
        elif p.get("sch_prop"):
            prop = p["sch_prop"]
        else:
            skipped.append((name, "attr %r needs sch_prop in YAML"
                            % attr))
            continue
        v = values[name]
        vs = ("%d" % int(round(v))) if p.get("integer") else fmt_num(v)
        for devspec in p["devices"]:
            if "/" in devspec:
                scope, dev = devspec.split("/", 1)
            else:
                scope, dev = "top", devspec
            cell = top_cell if scope in ("top", "", None) else scope
            inst = dev.lstrip("xX") if dev[:1] in "xX" and \
                len(dev) > 1 else dev
            edits.append((cell, inst, prop, vs, name))
    return edits, skipped


_READ_TMPL = """let((out cv)
  out = ""
  foreach(spec list(%s)
    cv = dbOpenCellViewByType("%s" car(spec) "schematic" "schematic" "r")
    if(cv then
      foreach(ii cv~>instances
        when(equal(ii~>name cadr(spec))
          out = strcat(out car(spec) "/" cadr(spec) "/" caddr(spec) "="
                sprintf(nil "%%L" (dbGetPropByName(ii caddr(spec))~>value
                                   || "<unset>")) ";")))
      dbClose(cv)
    else out = strcat(out car(spec) ": OPENFAIL;")))
  out
)"""

_WRITE_TMPL = """let((out cv n)
  out = ""
  foreach(cellgrp list(%s)
    cv = dbOpenCellViewByType("%s" car(cellgrp) "schematic" "schematic" "a")
    if(cv then
      n = 0
      foreach(ed cadr(cellgrp)
        foreach(ii cv~>instances
          when(equal(ii~>name car(ed))
            dbReplaceProp(ii cadr(ed) "string" caddr(ed))
            n = n+1)))
      dbSave(cv) dbClose(cv)
      out = strcat(out car(cellgrp) ":" sprintf(nil "%%d" n) " props;")
    else out = strcat(out car(cellgrp) ": OPENFAIL;")))
  out
)"""


def read_current(cfg, edits):
    lib = cfg.get("virtuoso", {}).get("lib")
    specs = " ".join('list("%s" "%s" "%s")' % (c, i, p)
                     for c, i, p, _, _ in edits)
    return skill_exec(_READ_TMPL % (specs, lib), timeout=60)


def apply_edits(cfg, edits):
    lib = cfg.get("virtuoso", {}).get("lib")
    by_cell = {}
    for cell, inst, prop, val, _ in edits:
        by_cell.setdefault(cell, []).append((inst, prop, val))
    grps = " ".join(
        'list("%s" list(%s))'
        % (cell, " ".join('list("%s" "%s" "%s")' % e for e in eds))
        for cell, eds in by_cell.items())
    return skill_exec(_WRITE_TMPL % (grps, lib), timeout=120)
