"""Configuration checks shared by simulation and the offline preflight command."""
import math
import os
import re

from .netlist import parse_num


def number(value, label):
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number" % label)
    parsed = parse_num(value)
    if parsed is None or not math.isfinite(parsed):
        raise ValueError("%s must be a finite number (engineering suffixes allowed)" % label)
    return parsed


def validate_config(cfg):
    """Report independent input errors together, before starting any jobs."""
    if not isinstance(cfg, dict):
        raise ValueError("circuit YAML must contain a mapping")
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(isinstance(cfg.get("name"), str) and bool(cfg.get("name")),
          "name must be a nonempty string")
    simulator = cfg.get("simulator", "spectre")
    check(simulator in ("spectre", "hspice"), "simulator must be spectre or hspice")
    check(isinstance(cfg.get("netlist"), str) and bool(cfg.get("netlist")),
          "netlist must be a file path")
    if simulator == "spectre":
        check(isinstance(cfg.get("stimuli"), str) and bool(cfg.get("stimuli")),
              "stimuli must be a file path for Spectre")
    tran = cfg.get("tran")
    if not isinstance(tran, dict):
        errors.append("tran must contain stop and maxstep")
    else:
        for key in ("stop", "maxstep"):
            try:
                check(number(tran.get(key), "tran." + key) > 0,
                      "tran.%s must be positive" % key)
            except ValueError as exc:
                errors.append(str(exc))
    check(isinstance(cfg.get("save"), list) and bool(cfg.get("save"))
          and all(isinstance(s, str) and s.strip() for s in cfg.get("save", [])),
          "save must be a nonempty list of signals")
    params = cfg.get("params")
    if not isinstance(params, list) or not params:
        errors.append("params must be a nonempty list")
        params = []
    names = set()
    for i, p in enumerate(params):
        if not isinstance(p, dict):
            errors.append("params[%d] must be a mapping" % i)
            continue
        name = p.get("name")
        label = "param %s" % (name or i)
        if not isinstance(name, str) or not name:
            errors.append(label + ": name must be a nonempty string")
        elif name in names:
            errors.append(label + ": duplicate name")
        else:
            names.add(name)
        check(isinstance(p.get("devices"), list) and bool(p.get("devices"))
              and all(isinstance(d, str) and d.strip() for d in p.get("devices", [])),
              label + ": devices must be a nonempty list")
        check(isinstance(p.get("attr"), str) and bool(p.get("attr")),
              label + ": attr must be a nonempty string")
        for flag in ("integer", "log", "enabled"):
            if flag in p:
                check(isinstance(p[flag], bool), label + ": " + flag + " must be true/false")
        if p.get("attr") in ("m", "mr", "nf"):
            p["integer"] = True  # Both netlist writers always round multiplicities.
        try:
            lo, hi = (number(p.get(k), label + "." + k) for k in ("lo", "hi"))
            check(lo < hi, label + ": lo must be smaller than hi")
            check(not p.get("log") or lo > 0, label + ": log bounds must be positive")
            check(not p.get("integer") or math.ceil(lo) <= math.floor(hi),
                  label + ": integer range contains no integer")
        except ValueError as exc:
            errors.append(str(exc))
    metrics = cfg.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics must be a nonempty mapping")
        metrics = {}
    else:
        check(all(isinstance(k, str) and isinstance(v, str) and v.strip()
                  for k, v in metrics.items()), "metrics must map names to expressions")
    obj = cfg.get("objective")
    terms = [obj] if isinstance(obj, dict) else obj
    if not isinstance(terms, list) or not terms:
        errors.append("objective must contain at least one objective term")
        terms = []
    for term in terms:
        if not isinstance(term, dict):
            errors.append("objective term must be a mapping")
            continue
        check(isinstance(term.get("metric"), str) and term["metric"] in metrics,
              "objective references unknown metric: %s" % term.get("metric"))
        goal = term.get("goal", "maximize")
        check(goal in ("maximize", "minimize", "target"), "invalid objective goal: %s" % goal)
        if goal == "target" and "target" not in term:
            errors.append("target objective requires target")
        for key in ("target", "tol", "weight", "scale"):
            if key in term:
                try:
                    v = number(term[key], "objective." + key)
                    term[key] = v
                    check(key not in ("tol", "weight") or v >= 0,
                          "objective.%s must be nonnegative" % key)
                    check(key != "scale" or v > 0, "objective.scale must be positive")
                except ValueError as exc:
                    errors.append(str(exc))
    cons = cfg.get("constraints", [])
    if not isinstance(cons, list):
        errors.append("constraints must be a list")
        cons = []
    for c in cons:
        if not isinstance(c, dict):
            errors.append("constraint must be a mapping")
            continue
        check(isinstance(c.get("metric"), str) and c["metric"] in metrics,
              "constraint references unknown metric: %s" % c.get("metric"))
        bounds = set(c) & {"min", "max", "target", "tol"}
        check(bounds in ({"min"}, {"max"}, {"target", "tol"}),
              "constraint needs exactly min, max, or target + tol")
        for key in bounds:
            try:
                v = number(c[key], "constraint." + key)
                c[key] = v
                check(key != "tol" or v >= 0, "constraint.tol must be nonnegative")
            except ValueError as exc:
                errors.append(str(exc))
    corners = cfg.get("corners") or []
    check(isinstance(cfg.get("fixed", {}), dict), "fixed must map parameter names to values")
    check(not cfg.get("pdk") or isinstance(cfg["pdk"], str), "pdk must be a name")
    if not isinstance(corners, list):
        errors.append("corners must be a list")
        corners = []
    corner_names = set()
    for c in corners:
        if not isinstance(c, dict):
            errors.append("corner must be a mapping")
            continue
        name = c.get("name", "nominal")
        if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name) or name in (".", ".."):
            errors.append("corner name must be a simple name: %r" % name)
        elif name in corner_names:
            errors.append("duplicate corner name: " + name)
        else:
            corner_names.add(name)
    worst = cfg.get("corner_worst", {})
    check(isinstance(worst, dict) and all(k in metrics and v in ("min", "max", "mean")
                                        for k, v in worst.items()),
          "corner_worst requires known metrics and min/max/mean")
    settings = cfg.get(simulator, {}) if isinstance(simulator, str) else {}
    if not isinstance(settings, dict):
        errors.append("simulator settings must be a mapping")
    else:
        for key in ("mt", "timeout_s"):
            if key in settings:
                try:
                    v = number(settings[key], simulator + "." + key)
                    settings[key] = v
                    check(v > 0 and (key != "mt" or v.is_integer()),
                          simulator + "." + key + " must be positive" +
                          (" integer" if key == "mt" else ""))
                except ValueError as exc:
                    errors.append(str(exc))
    if errors:
        raise ValueError("Invalid circuit configuration:\n- " + "\n- ".join(errors))


def resolve_file(path, yaml_path, label):
    """Prefer YAML-relative paths; retain existing cwd/project-relative configs."""
    expanded = os.path.expanduser(os.path.expandvars(os.fspath(path)))
    if "$" in expanded:
        raise ValueError("%s has an unset environment variable: %s" % (label, path))
    if os.path.isabs(expanded):
        candidates = [expanded]
    else:
        project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [os.path.join(os.path.dirname(os.path.abspath(yaml_path)), expanded),
                      os.path.abspath(expanded), os.path.join(project, expanded)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise ValueError("%s file not found: %s (checked: %s)" %
                     (label, path, ", ".join(dict.fromkeys(candidates))))
