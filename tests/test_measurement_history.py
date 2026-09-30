import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optclient.optimize import Optimizer
from optlocal.engine import Engine
from optlocal.evaluator import LocalEvaluator


class EngineHistoryEvaluator:
    def __init__(self, signature, records=None):
        self.history_records = list(records or [])
        self.definition = {
            "params": [{"name": "x", "lo": 0.0, "hi": 1.0, "nominal": .5}],
            "metrics": ["score"],
            "objective": {"metric": "score", "goal": "maximize"},
            "constraints": [],
        }
        if signature is not None:
            self.definition["measurement_signature"] = signature

    def spec(self):
        return self.definition

    def history(self):
        return self.history_records


class FailingCircuit:
    def __init__(self, yaml_path):
        self.name = "history-failure"

    def spec(self):
        return {
            "params": [{"name": "x", "lo": 0.0, "hi": 1.0, "nominal": .5}],
            "fixed": {},
            "measurement_signature": "metrics-v2",
        }

    def evaluate(self, params, workdir):
        raise RuntimeError("simulator failed before producing metrics")


class ClientHistoryServer:
    def __init__(self, records):
        self.records = records

    def spec(self):
        return {
            "params": [{"name": "x", "lo": 0.0, "hi": 1.0, "nominal": .5}],
            "objective": {"metric": "score", "goal": "maximize"},
            "constraints": [],
            "measurement_signature": "metrics-v2",
        }

    def history(self):
        return self.records


class MeasurementHistoryTests(unittest.TestCase):
    @staticmethod
    def _record(trial, signature, ok=True, x=.2):
        return {
            "trial": trial,
            "workdir": "trial_%d" % trial,
            "params": {"x": x},
            "ok": ok,
            "metrics": {"score": x} if ok else None,
            "measurement_signature": signature,
        }

    def test_engine_resume_filters_signature_but_keeps_matching_failures(self):
        records = [
            self._record(1, "metrics-v2", ok=True, x=.2),
            self._record(2, "metrics-v2", ok=False, x=.4),
            self._record(3, "metrics-v1", ok=True, x=.6),
            {key: value for key, value in self._record(4, "metrics-v1").items()
             if key != "measurement_signature"},
        ]
        evaluator = EngineHistoryEvaluator("metrics-v2", records)
        with tempfile.TemporaryDirectory() as outdir:
            engine = Engine(evaluator, outdir, batch=1, n_init=1,
                            on_log=lambda _: None)
            self.assertEqual(engine.resume(), 2)
            self.assertEqual([record["trial"] for record in engine.records], [1, 2])
            self.assertEqual(engine.ok, [True, False])
            self.assertEqual(engine.summary()["n_failed"], 1)

    def test_checkpoint_with_old_measurement_signature_is_not_restored(self):
        old_record = self._record(10, "metrics-v1", ok=True, x=.3)
        old_evaluator = EngineHistoryEvaluator("metrics-v1", [old_record])
        with tempfile.TemporaryDirectory() as outdir:
            saved = Engine(old_evaluator, outdir, batch=1, n_init=1,
                           on_log=lambda _: None)
            self.assertTrue(saved._ingest(old_record))
            saved.stagnation_count = 4
            saved.exploration_restarts = 2
            saved.exploration_pending = True
            saved.save_best()

            same = Engine(EngineHistoryEvaluator("metrics-v1", [old_record]),
                          outdir, batch=1, n_init=1, on_log=lambda _: None)
            self.assertEqual(same.resume(), 1)
            self.assertEqual(same.stagnation_count, 4)
            self.assertEqual(same.exploration_restarts, 2)
            self.assertTrue(same.exploration_pending)

            new_record = self._record(11, "metrics-v2", ok=True, x=.7)
            changed = Engine(EngineHistoryEvaluator("metrics-v2", [new_record]),
                             outdir, batch=1, n_init=1, on_log=lambda _: None)
            # The replacement history has the same number of records, so the
            # measurement signature itself, rather than only the count, must
            # isolate the checkpoint.
            self.assertEqual(changed.resume(), 1)
            self.assertEqual(changed.records[0]["trial"], 11)
            self.assertEqual(changed.stagnation_count, 0)
            self.assertEqual(changed.exploration_restarts, 0)
            self.assertFalse(changed.exploration_pending)

    def test_local_evaluator_failure_history_gets_measurement_signature(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("optserver.circuit.Circuit", FailingCircuit):
                evaluator = LocalEvaluator("unused.yaml", workroot=root, max_jobs=1)
                result = evaluator.evaluate({"x": .3})
            self.assertFalse(result["ok"])
            self.assertEqual(result["measurement_signature"], "metrics-v2")
            history = evaluator.history()
            self.assertEqual(len(history), 1)
            self.assertFalse(history[0]["ok"])
            self.assertEqual(history[0]["measurement_signature"], "metrics-v2")

    def test_client_resume_from_server_filters_and_counts_matching_records(self):
        records = [
            self._record(20, "metrics-v2", ok=True, x=.2),
            self._record(21, "metrics-v2", ok=False, x=.4),
            self._record(22, "metrics-v1", ok=True, x=.6),
            {key: value for key, value in self._record(23, "metrics-v1").items()
             if key != "measurement_signature"},
        ]
        with tempfile.TemporaryDirectory() as outdir:
            optimizer = Optimizer(
                ClientHistoryServer(records), outdir, batch=1, n_init=1,
                device="cpu", use_dkl=False
            )
            self.assertEqual(optimizer.resume_from_server(), 2)
            self.assertEqual([record["trial"] for record in optimizer.records], [20, 21])
            self.assertEqual(optimizer.ok, [True, False])
            with open(Path(outdir, "history.jsonl"), encoding="utf-8") as stream:
                written = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual([record["trial"] for record in written], [20, 21])


if __name__ == "__main__":
    unittest.main()
