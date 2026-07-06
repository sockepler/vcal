"""HTTP client for the optserver simulation service on linux02."""
import concurrent.futures as cf

import requests


class SimServer:
    def __init__(self, base_url, timeout=1800):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self):
        return requests.get(self.base + "/api/health", timeout=10).json()

    def spec(self):
        return requests.get(self.base + "/api/spec", timeout=30).json()

    def history(self):
        return requests.get(self.base + "/api/history",
                            timeout=120).json()

    def evaluate(self, params):
        r = requests.post(self.base + "/api/evaluate",
                          json={"params": params}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def evaluate_batch(self, param_list, parallel=4):
        """Fire q evaluations concurrently (server caps spectre jobs)."""
        with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(self.evaluate, p) for p in param_list]
            out = []
            for f in futs:
                try:
                    out.append(f.result())
                except Exception as e:
                    out.append({"ok": False, "error": "http: %s" % e,
                                "metrics": None, "params": None})
        return out
