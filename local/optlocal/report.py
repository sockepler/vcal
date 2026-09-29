"""Small, dependency-free helpers used by the local GUI report actions."""

import csv
import json
import os


def _cell(value):
    """Return a stable CSV cell for a scalar or nested value."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_records_csv(records, path):
    """Write evaluation records with both structured and scalar columns.

    ``params`` and ``metrics`` stay as JSON columns so no information is lost;
    common scalar fields are also kept as separate columns for convenient
    spreadsheet filtering.  Extra fields emitted by an evaluator are added in
    deterministic order.
    """
    rows = list(records or [])
    common = ["trial", "ok", "error", "params", "metrics", "sim_s",
              "workdir"]
    extras = sorted({k for row in rows if isinstance(row, dict)
                     for k in row if k not in common})
    params = sorted({k for row in rows if isinstance(row, dict)
                     for k in (row.get("params") or {})})
    metrics = sorted({k for row in rows if isinstance(row, dict)
                      for k in (row.get("metrics") or {})})
    columns = common + extras + ["param." + k for k in params] + ["metric." + k for k in metrics]
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = row if isinstance(row, dict) else {"error": str(row)}
            cells = {key: _cell(row.get(key)) for key in common + extras}
            cells.update({"param." + k: _cell(v) for k, v in (row.get("params") or {}).items()})
            cells.update({"metric." + k: _cell(v) for k, v in (row.get("metrics") or {}).items()})
            writer.writerow(cells)
    return path
