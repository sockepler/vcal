"""Evaluation backends for the local optimizer.

LocalEvaluator  — runs simulations in-process (no HTTP, no network):
                  wraps optserver.circuit.Circuit with a thread pool.
RemoteEvaluator — optional: same interface against a remote optserver.
"""
import concurrent.futures as cf
import json
import os
import threading


class LocalEvaluator:
    def __init__(self, yaml_path, workroot=None, max_jobs=4):
        from optserver.circuit import Circuit
        self.ckt = Circuit(yaml_path)
        self.max_jobs = max_jobs
        root = workroot or os.path.join(
            os.path.dirname(os.path.abspath(yaml_path)), "..", "work",
            self.ckt.name)
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.histfile = os.path.join(self.root, "history.jsonl")
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_jobs)
        self._counter = self._count_existing()

    def _count_existing(self):
        n = 0
        for d in os.listdir(self.root):
            if d.startswith("trial_"):
                try:
                    n = max(n, int(d.split("_")[1]) + 1)
                except ValueError:
                    pass
        return n

    def spec(self):
        return self.ckt.spec()

    def history(self):
        out = []
        if os.path.exists(self.histfile):
            with open(self.histfile) as f:
                out = [json.loads(ln) for ln in f if ln.strip()]
        return out

    def evaluate(self, params):
        with self._lock:
            idx = self._counter
            self._counter += 1
        workdir = os.path.join(self.root, "trial_%05d" % idx)
        with self._sem:
            res = self.ckt.evaluate(params, workdir)
        res["trial"] = idx
        with self._lock:
            with open(self.histfile, "a") as f:
                f.write(json.dumps(res) + "\n")
        return res

    def evaluate_batch(self, param_list, parallel=None):
        par = min(parallel or self.max_jobs, self.max_jobs)
        with cf.ThreadPoolExecutor(max_workers=par) as ex:
            futs = [ex.submit(self.evaluate, p) for p in param_list]
            out = []
            for f in futs:
                try:
                    out.append(f.result())
                except Exception as e:
                    out.append({"ok": False, "error": str(e),
                                "metrics": None, "params": None})
        return out


class RemoteEvaluator:
    """Same interface over HTTP (reuses the Windows client's SimServer)."""

    def __init__(self, base_url, timeout=1800):
        from optclient.remote import SimServer
        self._srv = SimServer(base_url, timeout=timeout)

    def spec(self):
        return self._srv.spec()

    def history(self):
        return self._srv.history()

    def evaluate(self, params):
        return self._srv.evaluate(params)

    def evaluate_batch(self, param_list, parallel=4):
        return self._srv.evaluate_batch(param_list, parallel=parallel)
