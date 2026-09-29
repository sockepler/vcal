"""Candidate proposers: local GPU/CPU, or a remote GPU compute server
(`python -m optclient serve-gpu` on the other machine). The remote
proposer falls back to local computation if the server is unreachable.
"""
import json

import numpy as np
import requests

from optclient.propose import propose_candidates

from .i18n import tr


class LocalProposer:
    def __init__(self, device, log=print):
        self.device = device
        self.name = "local:%s" % device

    def propose(self, X, y, C, batch, tr_length, center,
                use_dkl, dkl_after):
        return propose_candidates(X, y, C, batch, tr_length, center,
                                  use_dkl=use_dkl, dkl_after=dkl_after,
                                  device=self.device)


class RemoteProposer:
    def __init__(self, url, device_fallback="cpu", log=print,
                 timeout=600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._log = log
        self._fallback = LocalProposer(device_fallback)
        self.name = "remote:" + self.url

    def health(self):
        r = requests.get(self.url + "/api/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def propose(self, X, y, C, batch, tr_length, center,
                use_dkl, dkl_after):
        payload = {"X": np.asarray(X, float).tolist(),
                   "y": np.asarray(y, float).ravel().tolist(),
                   "C": (np.asarray(C, float).tolist()
                         if C is not None and np.size(C) else None),
                   "batch": int(batch),
                   "tr_length": float(tr_length),
                   "center": np.asarray(center, float).tolist(),
                   "use_dkl": bool(use_dkl),
                   "dkl_after": int(dkl_after)}
        try:
            r = requests.post(self.url + "/api/propose", json=payload,
                              timeout=self.timeout)
            r.raise_for_status()
            out = r.json()
            if "error" in out:
                raise RuntimeError(out["error"])
            return (np.asarray(out["candidates"], float),
                    bool(out["used_dkl"]))
        except Exception as e:
            self._log(tr("远程GPU失败(%s)，本轮回退本地计算") % e)
            return self._fallback.propose(X, y, C, batch, tr_length,
                                          center, use_dkl, dkl_after)
