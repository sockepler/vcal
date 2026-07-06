"""Remote GPU compute server: exposes propose_candidates over HTTP so
another machine (e.g. linux02 running the GUI + simulations) can borrow
this machine's GPU for the surrogate-model math.

    python -m optclient serve-gpu --port 8494

stdlib http.server only — no extra dependencies on the GPU box.
API:
    GET  /api/health   -> {ok, device, torch}
    POST /api/propose  -> in : {X, y, C, batch, tr_length, center,
                                use_dkl, dkl_after}
                          out: {candidates, used_dkl, device, secs}
"""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .propose import propose_candidates


def _device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "cpu"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/health"):
            dev, name = _device()
            self._send(200, {"ok": True, "device": dev,
                             "device_name": name,
                             "torch": torch.__version__})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/api/propose"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            dev, name = _device()
            t0 = time.time()
            cand, used_dkl = propose_candidates(
                req["X"], req["y"], req.get("C"),
                req.get("batch", 4), req["tr_length"], req["center"],
                use_dkl=req.get("use_dkl", True),
                dkl_after=req.get("dkl_after", 48), device=dev)
            self._send(200, {"candidates": cand.tolist(),
                             "used_dkl": used_dkl,
                             "device": name,
                             "secs": round(time.time() - t0, 2)})
        except Exception as e:
            import traceback
            self._send(500, {"error": str(e),
                             "trace": traceback.format_exc()[-1500:]})

    def log_message(self, fmt, *args):
        print("[gpuserver] %s" % (fmt % args), flush=True)


def main(port=8494, host="0.0.0.0"):
    dev, name = _device()
    print("GPU compute server on %s:%d  device=%s (%s)"
          % (host, port, dev, name), flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
