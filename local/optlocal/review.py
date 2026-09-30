"""Offline, descriptive feedback for a local optimization run.

Rank associations are observations from the sampled points, not causal
sensitivities. They never change constraints or parameter bounds automatically.
"""
import json
import math

import numpy as np

from .objective import Objective


def _ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def analyze_records(records, objective, constraints=(), params=()):
    """Return JSON-compatible run statistics and actionable advice codes."""
    records = list(records)
    obj = Objective(objective)
    constraints = list(constraints) + obj.implicit_constraints()
    valid, feasible = [], []
    best, best_score, last_improvement = None, None, None
    seen, repeats, errors = set(), 0, {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors["invalid history record"] = errors.get("invalid history record", 0) + 1
            continue
        values = record.get("params") or {}
        key = json.dumps(values, sort_keys=True, default=str)
        if values:
            repeats += key in seen
            seen.add(key)
        metrics = record.get("metrics")
        try:
            if not record.get("ok") or not isinstance(metrics, dict) or not metrics:
                raise ValueError(record.get("error") or "missing successful metrics")
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError("non-finite metrics")
            score = float(obj.scalar(metrics))
            if not math.isfinite(score):
                raise ValueError("non-finite objective")
            violations = []
            for c in constraints:
                value = float(metrics[c["metric"]])
                if "max" in c:
                    violations.append(value - float(c["max"]))
                elif "min" in c:
                    violations.append(float(c["min"]) - value)
                else:
                    violations.append(abs(value - float(c["target"])) - float(c["tol"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            error = str(record.get("error") or exc)[:400]
            errors[error] = errors.get(error, 0) + 1
            continue
        valid.append((index, record, score))
        if all(value <= 0 for value in violations):
            feasible.append((index, record, score))
            if best_score is None or score > best_score + 1e-9 * max(abs(best_score), 1e-12):
                best, best_score, last_improvement = record, score, index

    associations = []
    # Feasible points describe the useful design region. If too few exist,
    # report all valid points explicitly, without treating them as feasible.
    sample = feasible if len(feasible) >= 6 else valid
    for p in params:
        if not p.get("enabled", True):
            continue
        pairs = []
        for _, record, score in sample:
            try:
                value = float((record.get("params") or {})[p["name"]])
                if math.isfinite(value):
                    pairs.append((value, score))
            except (KeyError, TypeError, ValueError):
                continue
        if len(pairs) < 6:
            continue
        x, y = (np.asarray(v) for v in zip(*pairs))
        rx, ry = _ranks(x), _ranks(y)
        if np.ptp(rx) == 0 or np.ptp(ry) == 0:
            continue
        rho = float(np.corrcoef(rx, ry)[0, 1])
        associations.append({"parameter": p["name"], "rho": rho, "n": len(pairs)})
    associations.sort(key=lambda entry: (-abs(entry["rho"]), entry["parameter"]))

    n = len(records)
    n_failed = n - len(valid)
    stalled = n - last_improvement - 1 if last_improvement is not None else None
    active = sum(bool(p.get("enabled", True)) for p in params)
    advice = []
    if n == 0:
        advice.append("run_nominal")
    elif n_failed / n >= 0.25:
        advice.append("inspect_failures")
    if valid and not feasible:
        advice.append("check_constraints")
    if feasible:
        advice.append("reuse_best")
    if stalled is not None and stalled >= 8:
        advice.append("enable_exploration")
    if active > 12:
        advice.append("select_scope")
    if len(sample) < 6:
        advice.append("collect_more")
    return {
        "n": n, "n_ok": len(valid), "n_failed": n_failed,
        "n_feasible": len(feasible), "success_rate": len(valid) / n if n else None,
        "feasible_rate": len(feasible) / n if n else None,
        "repeated_points": repeats, "best_record": best,
        "best_scalar": best_score, "evaluations_since_improvement": stalled,
        "associations": associations,
        "association_population": "feasible" if len(feasible) >= 6 else "all_valid",
        "errors": [{"error": error, "count": count} for error, count in
                   sorted(errors.items(), key=lambda item: (-item[1], item[0]))[:5]],
        "advice": advice,
    }
