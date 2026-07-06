"""Run reporting: convergence plot + best-point export."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_history(path):
    recs = []
    with open(path) as f:
        for ln in f:
            if ln.strip():
                recs.append(json.loads(ln))
    return recs


def convergence_plot(rundir, spec, out=None):
    recs = load_history(os.path.join(rundir, "history.jsonl"))
    obj = spec["objective"]["metric"]
    sign = 1.0 if spec["objective"]["goal"] == "maximize" else -1.0
    cons = spec.get("constraints", [])

    def feasible(m):
        for c in cons:
            if "max" in c and m[c["metric"]] > c["max"]:
                return False
            if "min" in c and m[c["metric"]] < c["min"]:
                return False
        return True

    xs, ys, fs, best = [], [], [], []
    cur = None
    for i, r in enumerate(recs):
        m = r.get("metrics")
        if r.get("ok") and m:
            xs.append(i)
            ys.append(m[obj])
            f = feasible(m)
            fs.append(f)
            if f and (cur is None or sign * m[obj] > sign * cur):
                cur = m[obj]
        best.append(cur)

    fig, ax = plt.subplots(figsize=(9, 5))
    xf = [x for x, f in zip(xs, fs) if f]
    yf = [y for y, f in zip(ys, fs) if f]
    xi = [x for x, f in zip(xs, fs) if not f]
    yi = [y for y, f in zip(ys, fs) if not f]
    ax.scatter(xi, yi, s=14, c="silver", label="infeasible")
    ax.scatter(xf, yf, s=18, c="tab:blue", label="feasible")
    bx = [i for i, b in enumerate(best) if b is not None]
    by = [b for b in best if b is not None]
    ax.plot(bx, by, "r-", lw=2, label="best feasible")
    ax.set_xlabel("trial")
    ax.set_ylabel(obj)
    ax.set_title("%s — %s %s" % (spec["circuit"],
                                 spec["objective"]["goal"], obj))
    ax.legend()
    ax.grid(alpha=0.3)
    out = out or os.path.join(rundir, "convergence.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def best_table(rundir):
    path = os.path.join(rundir, "best.json")
    if not os.path.exists(path):
        return "no best.json yet"
    with open(path) as f:
        b = json.load(f)
    lines = ["best trial: %s   feasible: %s" % (b.get("trial"),
                                                b.get("feasible"))]
    lines.append("-- metrics --")
    for k, v in (b.get("metrics") or {}).items():
        lines.append("  %-12s %.6g" % (k, v))
    lines.append("-- params --")
    for k, v in (b.get("params") or {}).items():
        lines.append("  %-16s %.6g" % (k, v))
    return "\n".join(lines)
