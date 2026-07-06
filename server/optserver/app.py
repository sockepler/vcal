"""HTTP simulation service.

    GET  /api/health            -> {ok, circuit, busy, done}
    GET  /api/spec              -> parameter space + objective (client setup)
    POST /api/evaluate          -> {params:{name:val}} -> blocking, result
    GET  /api/history           -> all evaluations so far (resume support)
    GET  /api/nominal           -> evaluate the unmodified netlist once

Start:  ./run_server.sh circuits/your_circuit.yaml [port]
Concurrency: Flask threads + a semaphore capping parallel spectre jobs;
the BO client fires `batch` HTTP requests in parallel.
"""
import json
import os
import threading

from flask import Flask, jsonify, request

from .circuit import Circuit

app = Flask(__name__)
_state = {}


def init(yaml_path, workroot=None, max_jobs=4):
    ckt = Circuit(yaml_path)
    root = workroot or os.path.join(
        os.path.dirname(os.path.abspath(yaml_path)), "..", "work", ckt.name)
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    _state.update(
        ckt=ckt, root=root, sem=threading.Semaphore(max_jobs),
        lock=threading.Lock(), counter=_count_existing(root),
        histfile=os.path.join(root, "history.jsonl"))
    return app


def _count_existing(root):
    n = 0
    for d in os.listdir(root):
        if d.startswith("trial_"):
            try:
                n = max(n, int(d.split("_")[1]) + 1)
            except ValueError:
                pass
    return n


@app.get("/api/health")
def health():
    return jsonify(ok=True, circuit=_state["ckt"].name,
                   done=_state["counter"])


@app.get("/api/spec")
def spec():
    return jsonify(_state["ckt"].spec())


@app.get("/api/history")
def history():
    out = []
    hf = _state["histfile"]
    if os.path.exists(hf):
        with open(hf) as f:
            out = [json.loads(ln) for ln in f if ln.strip()]
    return jsonify(out)


@app.post("/api/evaluate")
def evaluate():
    body = request.get_json(force=True)
    params = body.get("params", {})
    with _state["lock"]:
        idx = _state["counter"]
        _state["counter"] += 1
    workdir = os.path.join(_state["root"], "trial_%05d" % idx)
    with _state["sem"]:
        res = _state["ckt"].evaluate(params, workdir)
    res["trial"] = idx
    with _state["lock"]:
        with open(_state["histfile"], "a") as f:
            f.write(json.dumps(res) + "\n")
    return jsonify(res)


@app.get("/client.zip")
def client_zip():
    """Serve the Windows client package (built by make_client_zip)."""
    from flask import send_file
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "client_pkg.zip")
    if not os.path.exists(path):
        return jsonify(error="run server/make_client_zip.sh first"), 404
    return send_file(path, as_attachment=True,
                     download_name="optclient.zip")


@app.get("/setup_windows.ps1")
def setup_ps1():
    from flask import send_file
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "circuits", "..",
        "setup_windows.ps1")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return jsonify(error="no setup script"), 404
    return send_file(path, mimetype="text/plain")


@app.get("/bench/<path:name>")
def bench_file(name):
    from flask import send_file
    base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "bench")
    path = os.path.abspath(os.path.join(base, name))
    if not path.startswith(base) or not os.path.exists(path):
        return jsonify(error="not found"), 404
    return send_file(path, mimetype="application/octet-stream")


@app.post("/BENCHRESULT")
def benchresult():
    body = request.get_data(as_text=True)
    print("[benchresult] %s" % body, flush=True)
    return jsonify(ok=True)


@app.get("/BEACON-<msg>")
def beacon(msg):
    """Progress beacons from the Windows setup script (logged)."""
    print("[beacon] %s" % msg, flush=True)
    return jsonify(ok=True)


@app.post("/api/nominal")
def nominal():
    """Evaluate with no parameter changes (baseline)."""
    workdir = os.path.join(_state["root"], "nominal")
    with _state["sem"]:
        res = _state["ckt"].evaluate({}, workdir)
    return jsonify(res)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit_yaml")
    ap.add_argument("--port", type=int, default=8492)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--workroot", default=None)
    a = ap.parse_args()
    init(a.circuit_yaml, a.workroot, a.max_jobs)
    print("optserver: circuit=%s  http://%s:%d  max_jobs=%d"
          % (_state["ckt"].name, a.host, a.port, a.max_jobs))
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
