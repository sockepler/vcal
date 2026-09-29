import json
import os
import tempfile
import threading
import time
import unittest
import warnings
from unittest import mock

from optlocal.evaluator import LocalEvaluator, read_history


class FakeCircuit:
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    started = None
    release = None
    fail_values = set()
    fixed = {}
    nominal_gain = 1.25

    def __init__(self, yaml_path):
        self.name = "fake"
        self.calls = []

    def spec(self):
        return {
            "params": [
                {"name": "gain", "lo": 0, "hi": 2,
                 "nominal": type(self).nominal_gain, "integer": False},
                {"name": "count", "lo": 0, "hi": 10,
                 "nominal": 3, "integer": True},
                {"name": "expression", "lo": 0, "hi": 10,
                 "nominal": None, "integer": False},
            ],
            "fixed": dict(type(self).fixed),
        }

    def evaluate(self, params, workdir):
        self.calls.append((dict(params), workdir))
        with self.active_lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active,
                                        type(self).active)
        try:
            if type(self).started is not None:
                type(self).started.set()
            if type(self).release is not None:
                type(self).release.wait(5)
            if params.get("gain", 1.25) in type(self).fail_values:
                raise RuntimeError("simulated failure")
            time.sleep(0.01)
            return {"ok": True,
                    "metrics": {"score": params.get("gain", 1.25)},
                    "params": {"stale": True}, "workdir": workdir,
                    "sim_s": 0.01}
        finally:
            with self.active_lock:
                type(self).active -= 1


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        FakeCircuit.active = 0
        FakeCircuit.max_active = 0
        FakeCircuit.started = None
        FakeCircuit.release = None
        FakeCircuit.fail_values = set()
        FakeCircuit.fixed = {}
        FakeCircuit.nominal_gain = 1.25
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch("optserver.circuit.Circuit", FakeCircuit)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def evaluator(self, max_jobs=2, root=None):
        return LocalEvaluator("unused.yaml", workroot=root or self.tmp.name,
                              max_jobs=max_jobs)

    def test_positive_limits_and_dynamic_limit(self):
        with self.assertRaises(ValueError):
            self.evaluator(max_jobs=0)
        ev = self.evaluator(max_jobs=3)
        with self.assertRaises(ValueError):
            ev.max_jobs = 0
        ev.max_jobs = 1
        with self.assertRaises(ValueError):
            ev.evaluate_batch([], parallel=0)
        with self.assertRaises(ValueError):
            ev.evaluate_batch([], parallel=-1)
        with self.assertRaises(ValueError):
            ev.evaluate_batch([], parallel=1.5)

        results = ev.evaluate_batch([{"gain": i} for i in range(4)],
                                    parallel=4)
        self.assertEqual(len(results), 4)
        self.assertEqual(FakeCircuit.max_active, 1)

    def test_changing_limit_while_running_is_explicit_error(self):
        FakeCircuit.started = threading.Event()
        FakeCircuit.release = threading.Event()
        ev = self.evaluator(max_jobs=1)
        thread = threading.Thread(target=ev.evaluate, args=({"gain": 1},))
        thread.start()
        self.assertTrue(FakeCircuit.started.wait(2))
        with self.assertRaisesRegex(RuntimeError, "evaluations are running"):
            ev.max_jobs = 2
        FakeCircuit.release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_failure_is_normalized_and_written(self):
        FakeCircuit.fail_values = {2.0}
        ev = self.evaluator(max_jobs=1)
        result = ev.evaluate({"gain": 2.0, "count": 4.7})
        self.assertFalse(result["ok"])
        self.assertEqual(result["params"], {"gain": 2.0, "count": 5})
        self.assertEqual(result["trial"], 0)
        self.assertTrue(os.path.isdir(result["workdir"]))
        self.assertIn("simulated failure", result["error"])
        self.assertIn("sim_s", result)
        self.assertEqual(ev.history(), [result])

        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            ev.evaluate({"gain": 1, "unknown": 2})

    def test_call_keeps_missing_values_out_and_record_uses_fixed_then_nominal(self):
        FakeCircuit.fixed = {"count": 7}
        ev = self.evaluator(max_jobs=1)
        result = ev.evaluate({"gain": 1})
        call_params, _ = ev.ckt.calls[-1]
        self.assertEqual(call_params, {"gain": 1.0})
        self.assertEqual(result["params"],
                         {"gain": 1.0, "count": 7})
        self.assertNotIn("expression", result["params"])

    def test_missing_nominal_is_recorded_without_applying_or_clipping(self):
        FakeCircuit.nominal_gain = 3.0
        ev = self.evaluator(max_jobs=1)
        result = ev.evaluate({})
        call_params, _ = ev.ckt.calls[-1]
        self.assertEqual(call_params, {})
        self.assertEqual(result["params"]["gain"], 3.0)

    def test_nonfinite_and_integer_bound_values_are_rejected_or_clamped(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            LocalEvaluator._actual_param_value(
                "x", float("nan"), {"lo": 0, "hi": 2})
        with self.assertRaisesRegex(ValueError, "finite"):
            LocalEvaluator._actual_param_value(
                "x", float("inf"), {"lo": 0, "hi": 2})
        integer_spec = {"lo": 1.2, "hi": 3.8, "integer": True}
        self.assertEqual(
            LocalEvaluator._actual_param_value("x", 3.8, integer_spec), 3)
        self.assertEqual(
            LocalEvaluator._actual_param_value("x", 1.2, integer_spec), 2)

    def test_batch_preserves_order_when_one_input_is_invalid(self):
        ev = self.evaluator(max_jobs=2)
        inputs = [{"gain": 0}, {"gain": 1, "unknown": 5}, {"gain": 2}]
        results = ev.evaluate_batch(inputs, parallel=2)
        self.assertEqual(len(results), len(inputs))
        self.assertEqual(results[0]["params"]["gain"], 0.0)
        self.assertEqual(results[1]["params"], inputs[1])
        self.assertFalse(results[1]["ok"])
        self.assertEqual(results[2]["params"]["gain"], 2.0)

    def test_tail_is_warned_ignored_backed_up_before_append(self):
        ev = self.evaluator(max_jobs=1)
        with open(ev.histfile, "wb") as f:
            f.write(b'{"trial": 7}\n{"trial":')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(ev.history(), [{"trial": 7}])
        self.assertTrue(any("incomplete final history JSON line" in
                            str(w.message) for w in caught))

        ev.evaluate({"gain": 1})
        with open(ev.histfile, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual([row["trial"] for row in rows], [7, 0])
        self.assertTrue(os.path.exists(ev.histfile + ".corrupt-tail"))

    def test_middle_corruption_has_line_number(self):
        ev = self.evaluator()
        with open(ev.histfile, "w") as f:
            f.write('{"trial": 0}\n')
            f.write('{"broken":\n')
            f.write('{"trial": 2}\n')
        with self.assertRaisesRegex(ValueError, r"line 2"):
            ev.history()
        with self.assertRaisesRegex(ValueError, r"line 2"):
            read_history(ev.histfile)

    def test_trial_directory_reservation_is_atomic_across_evaluators(self):
        ev1 = self.evaluator(max_jobs=1)
        ev2 = self.evaluator(max_jobs=1, root=self.tmp.name)
        one = ev1.evaluate({"gain": 1})
        two = ev2.evaluate({"gain": 1})
        self.assertNotEqual(one["trial"], two["trial"])
        self.assertNotEqual(one["workdir"], two["workdir"])
        self.assertEqual(len([name for name in os.listdir(self.tmp.name)
                              if name.startswith("trial_")]), 2)


if __name__ == "__main__":
    unittest.main()
