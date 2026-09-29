"""Evaluation backends for the local optimizer.

LocalEvaluator  — runs simulations in-process (no HTTP, no network):
                  wraps optserver.circuit.Circuit with a thread pool.
RemoteEvaluator — optional: same interface against a remote optserver.
"""
import concurrent.futures as cf
import contextlib
import json
import math
import numbers
import os
import threading
import time
import warnings

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is available on supported Unix hosts
    fcntl = None


_HISTORY_LOCKS = {}
_HISTORY_LOCKS_GUARD = threading.Lock()


def _history_lock(path):
    """Return the in-process lock shared by evaluators for *path*."""
    key = os.path.abspath(path)
    with _HISTORY_LOCKS_GUARD:
        lock = _HISTORY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _HISTORY_LOCKS[key] = lock
        return lock


def _line_has_ending(raw):
    return raw.endswith(b"\n") or raw.endswith(b"\r")


def _scan_history_bytes(data, path, warn_tail=True):
    """Parse history bytes and return ``(records, incomplete_tail)``.

    ``incomplete_tail`` is a ``(start_offset, raw_bytes, line_number)`` tuple
    only when the final physical line is not valid JSON and has no line
    ending.  Any other malformed line is an error, including a malformed
    final line that did have a line ending.
    """
    records = []
    tail = None
    offset = 0
    lines = data.splitlines(keepends=True)
    for line_no, raw in enumerate(lines, 1):
        start = offset
        offset += len(raw)
        if not raw.strip():
            continue
        try:
            text = raw.decode("utf-8")
            records.append(json.loads(text))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_incomplete_tail = (
                line_no == len(lines) and not _line_has_ending(raw))
            if is_incomplete_tail:
                tail = (start, raw, line_no)
                if warn_tail:
                    warnings.warn(
                        "Ignoring incomplete final history JSON line %d in %s: %s"
                        % (line_no, path, exc),
                        RuntimeWarning,
                        stacklevel=3,
                    )
                break
            raise ValueError(
                "Invalid JSON in history %s at line %d: %s"
                % (path, line_no, exc)
            ) from exc
    return records, tail


def read_history(path):
    """Read a JSONL history file.

    A crash can leave a final JSON object only partly written.  That one tail
    line is ignored with a warning; malformed lines before it are reported
    with their line number so that history corruption is not hidden.
    """
    path = os.path.abspath(os.fspath(path))
    try:
        with open(path, "rb") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = f.read()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError:
        return []
    records, _ = _scan_history_bytes(data, path)
    return records


@contextlib.contextmanager
def _locked_history_file(path):
    """Open *path* and hold an OS-level exclusive history lock."""
    f = open(path, "a+b")
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("%s must be a positive integer" % name)
    value = int(value)
    if value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _copy_params(params):
    try:
        return dict(params)
    except (TypeError, ValueError):
        return params


class LocalEvaluator:
    def __init__(self, yaml_path, workroot=None, max_jobs=4):
        max_jobs = _positive_integer(max_jobs, "max_jobs")
        from optserver.circuit import Circuit
        self.ckt = Circuit(yaml_path)
        root = workroot or os.path.join(
            os.path.dirname(os.path.abspath(yaml_path)), "..", "work",
            self.ckt.name)
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.histfile = os.path.join(self.root, "history.jsonl")
        self._lock = _history_lock(self.histfile)
        self._jobs_cond = threading.Condition()
        self._max_jobs = max_jobs
        self._active_jobs = 0
        self._counter = self._count_existing()

    @property
    def max_jobs(self):
        with self._jobs_cond:
            return self._max_jobs

    @max_jobs.setter
    def max_jobs(self, value):
        value = _positive_integer(value, "max_jobs")
        with self._jobs_cond:
            if self._active_jobs:
                raise RuntimeError(
                    "cannot change max_jobs while evaluations are running")
            self._max_jobs = value
            self._jobs_cond.notify_all()

    def _count_existing(self):
        n = 0
        for d in os.listdir(self.root):
            if d.startswith("trial_"):
                try:
                    n = max(n, int(d.split("_")[1]) + 1)
                except ValueError:
                    pass
        return n

    def _allocate_trial(self):
        """Reserve a unique trial directory using atomic mkdir."""
        with self._lock:
            while True:
                idx = self._counter
                self._counter += 1
                workdir = os.path.join(self.root, "trial_%05d" % idx)
                try:
                    os.mkdir(workdir)
                except FileExistsError:
                    continue
                return idx, workdir

    def _acquire_job(self):
        with self._jobs_cond:
            while self._active_jobs >= self._max_jobs:
                self._jobs_cond.wait()
            self._active_jobs += 1

    def _release_job(self):
        with self._jobs_cond:
            self._active_jobs -= 1
            self._jobs_cond.notify_all()

    def _parameter_specs(self):
        spec = self.ckt.spec()
        entries = spec.get("params", []) if isinstance(spec, dict) else []
        by_name = {}
        for entry in entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError("invalid circuit parameter specification")
            name = entry["name"]
            if name in by_name:
                raise ValueError("duplicate circuit parameter %s" % name)
            by_name[name] = entry
        fixed = spec.get("fixed", {}) if isinstance(spec, dict) else {}
        if fixed is None:
            fixed = {}
        if not isinstance(fixed, dict):
            raise ValueError("invalid fixed circuit parameter specification")
        return by_name, fixed

    @staticmethod
    def _actual_param_value(name, value, spec):
        try:
            actual = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameter %s must be numeric" % name) from exc
        if not math.isfinite(actual):
            raise ValueError("parameter %s must be finite" % name)
        lo = (float(spec["lo"]) if "lo" in spec and
              spec["lo"] is not None else None)
        hi = (float(spec["hi"]) if "hi" in spec and
              spec["hi"] is not None else None)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("parameter %s has invalid bounds" % name)
        if lo is not None:
            actual = max(actual, lo)
        if hi is not None:
            actual = min(actual, hi)
        if spec.get("integer", False):
            if lo is not None:
                actual = max(actual, math.ceil(lo))
            if hi is not None:
                actual = min(actual, math.floor(hi))
            return int(round(actual))
        return actual

    @staticmethod
    def _nominal_param_value(name, value):
        try:
            nominal = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameter %s nominal must be numeric" % name) \
                from exc
        if not math.isfinite(nominal):
            raise ValueError("parameter %s nominal must be finite" % name)
        return nominal

    def _normalize_params(self, params):
        if not isinstance(params, dict):
            try:
                params = dict(params)
            except (TypeError, ValueError) as exc:
                raise ValueError("params must be a mapping") from exc
        specs, fixed = self._parameter_specs()
        unknown = [name for name in params if name not in specs]
        if unknown:
            raise ValueError("unknown parameter(s): %s" % ", ".join(
                str(name) for name in unknown))
        normalized = {}
        for name, value in params.items():
            normalized[name] = self._actual_param_value(
                name, value, specs[name])

        recorded = dict(normalized)
        for name, value in fixed.items():
            if name in recorded or name not in specs:
                continue
            recorded[name] = self._actual_param_value(
                name, value, specs[name])
        for name, param_spec in specs.items():
            if name in recorded:
                continue
            if "nominal" not in param_spec or param_spec["nominal"] is None:
                continue
            recorded[name] = self._nominal_param_value(
                name, param_spec["nominal"])
        return normalized, recorded

    def _repair_history_tail(self, f):
        """Drop a crash tail before the next append and return file bytes."""
        f.seek(0)
        data = f.read()
        records, tail = _scan_history_bytes(
            data, self.histfile, warn_tail=False)
        del records  # Parsing is intentional: malformed middle lines must fail.
        if tail is not None:
            start, raw, line_no = tail
            backup = self.histfile + ".corrupt-tail"
            suffix = 0
            while True:
                candidate = backup if suffix == 0 else \
                    "%s.%d" % (backup, suffix)
                try:
                    with open(candidate, "xb") as bf:
                        bf.write(raw)
                    backup = candidate
                    break
                except FileExistsError:
                    suffix += 1
            f.seek(0)
            f.truncate(start)
            f.flush()
            warnings.warn(
                "Removed incomplete final history JSON line %d from %s; "
                "saved it as %s" % (line_no, self.histfile, backup),
                RuntimeWarning,
                stacklevel=3,
            )
            data = data[:start]
        return data

    def _append_history(self, result):
        with self._lock:
            with _locked_history_file(self.histfile) as f:
                data = self._repair_history_tail(f)
                if data and not data.endswith((b"\n", b"\r")):
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
                f.seek(0, os.SEEK_END)
                line = (json.dumps(result) + "\n").encode("utf-8")
                f.write(line)
                f.flush()

    @staticmethod
    def _failure(params, trial, workdir, error, sim_s):
        return {
            "params": _copy_params(params),
            "ok": False,
            "metrics": None,
            "error": str(error),
            "workdir": workdir,
            "trial": trial,
            "sim_s": round(sim_s, 2),
        }

    def spec(self):
        return self.ckt.spec()

    def history(self):
        with self._lock:
            return read_history(self.histfile)

    def evaluate(self, params):
        call_params, record_params = self._normalize_params(params)
        idx, workdir = self._allocate_trial()
        self._acquire_job()
        started = time.monotonic()
        call_params = _copy_params(call_params)
        try:
            try:
                res = self.ckt.evaluate(call_params, workdir)
                if not isinstance(res, dict):
                    raise TypeError("circuit evaluate returned a non-dict result")
                res = dict(res)
            except Exception as exc:
                res = self._failure(record_params, idx, workdir,
                                    exc, time.monotonic() - started)
            else:
                res["params"] = _copy_params(record_params)
                res["trial"] = idx
                res.setdefault("workdir", workdir)
                res.setdefault("sim_s", round(
                    time.monotonic() - started, 2))
        finally:
            self._release_job()
        self._append_history(res)
        return res

    def evaluate_batch(self, param_list, parallel=None):
        limit = self.max_jobs
        if parallel is None:
            requested = limit
        else:
            requested = _positive_integer(parallel, "parallel")
        par = min(requested, limit)
        params_in = list(param_list)
        with cf.ThreadPoolExecutor(max_workers=par) as ex:
            futs = [ex.submit(self.evaluate, p) for p in params_in]
            out = []
            for p, f in zip(params_in, futs):
                try:
                    out.append(f.result())
                except Exception as e:
                    out.append({"ok": False, "error": str(e),
                                "metrics": None, "params": _copy_params(p),
                                "sim_s": 0.0})
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
