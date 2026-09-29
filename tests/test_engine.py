import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from optclient.space import Space
from optclient.turbo import TurboState
from optlocal.engine import Engine
from optlocal.objective import Objective


class Evaluator:
    def __init__(self, fail=False):
        self.calls = []
        self.history_records = []
        self.fail = fail
        self.definition = {
            "params": [{"name": "x", "lo": 0., "hi": 1., "nominal": .5}],
            "metrics": ["score", "power"],
            "objective": {"metric": "score", "goal": "maximize"},
            "constraints": [{"metric": "power", "max": 1}],
        }

    def spec(self):
        return self.definition

    def history(self):
        return self.history_records

    def evaluate_batch(self, params, parallel=None):
        self.calls.append(len(params))
        out = []
        for p in params:
            rec = {"params": p, "ok": not self.fail, "trial": len(self.history_records),
                   "metrics": None if self.fail else {"score": p["x"], "power": .2},
                   "error": "simulator failed" if self.fail else None}
            self.history_records.append(rec)
            out.append(rec)
        return out


class Proposer:
    name = "test"

    def __init__(self):
        self.batches = []

    def propose(self, X, y, C, batch, *args):
        self.batches.append(batch)
        return np.array([[.01 * (len(X) + i)] for i in range(batch)]), False


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def engine(self, ev=None, **kwargs):
        eng = Engine(ev or Evaluator(), self.tmp.name, on_log=lambda _: None, **kwargs)
        eng.proposer = Proposer()
        return eng

    def test_budget_below_initialization_and_partial_final_batch(self):
        eng = self.engine(batch=4, n_init=24)
        eng.run(3)
        self.assertEqual(eng.ev.calls, [3])
        eng = self.engine(batch=4, n_init=4)
        eng.run(7)
        self.assertEqual(eng.ev.calls, [4, 3])
        self.assertEqual(eng.proposer.batches, [3])

    def test_nonfinite_metrics_and_failed_constraint_imputation(self):
        eng = self.engine()
        eng._ingest({"params": {"x": .1}, "ok": True,
                     "metrics": {"score": 1., "power": .1}})
        for bad in (float("nan"), float("inf"), None):
            eng._ingest({"params": {"x": .2}, "ok": True,
                         "metrics": {"score": bad, "power": .1}})
        self.assertEqual(eng.ok, [True, False, False, False])
        _, y, c = eng._arrays()
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue((c[1:] > 0).all())
        self.assertEqual(eng.summary()["n_feasible"], 1)

    def test_failed_evaluations_consume_budget_and_clear_stale_best(self):
        Path(self.tmp.name, "best.json").write_text('{"stale": true}')
        eng = self.engine(Evaluator(fail=True), n_init=8)
        eng.run(3)
        self.assertEqual(len(eng.records), 3)
        self.assertFalse(Path(self.tmp.name, "best.json").exists())
        summary = json.loads(Path(self.tmp.name, "summary.json").read_text())
        self.assertEqual(summary["n_failed"], 3)
        with open(Path(self.tmp.name, "history.csv"), encoding="utf-8-sig") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_backend_exception_retains_requested_points(self):
        eng = self.engine(n_init=4)
        with patch.object(eng.ev, "evaluate_batch", side_effect=RuntimeError("offline")):
            eng.run(3)
        self.assertEqual(len(eng.records), 3)
        self.assertTrue(all(r["params"] and not r["ok"] for r in eng.records))

    def test_resume_is_idempotent_and_skips_out_of_bounds(self):
        ev = Evaluator()
        ev.history_records = [
            {"trial": 0, "params": {"x": .3}, "ok": True,
             "metrics": {"score": .3, "power": .2}},
            {"trial": 1, "params": {"x": 2}, "ok": True,
             "metrics": {"score": 2, "power": .2}},
        ]
        eng = self.engine(ev)
        self.assertEqual(eng.resume(), 1)
        self.assertEqual(eng.resume(), 0)
        eng.run(1)
        self.assertEqual(ev.calls, [])
        self.assertEqual(len(eng.records), 1)

    def test_fixed_parameters_filter_history(self):
        ev = Evaluator()
        ev.definition["params"].append({"name": "z", "lo": 0., "hi": 1., "nominal": .4})
        ev.history_records = [{"params": {"x": .2, "z": .6}, "ok": True,
                               "metrics": {"score": .2, "power": .1}}]
        eng = self.engine(ev, spec_params=[ev.definition["params"][0]], fixed={"z": .4})
        self.assertEqual(eng.resume(), 0)

    def test_resume_after_running_does_not_duplicate_own_records(self):
        eng = self.engine(n_init=3)
        eng.run(3)
        self.assertEqual(eng.resume(), 0)
        self.assertEqual(len(eng.records), 3)

    def test_duplicate_discrete_points_are_not_resimulated(self):
        ev = Evaluator()
        ev.definition["params"][0].update(integer=True, nominal=0)
        eng = self.engine(ev, n_init=24)
        eng.run(10)
        self.assertEqual(len(eng.records), 2)
        self.assertEqual({r["params"]["x"] for r in eng.records}, {0, 1})
        self.assertEqual(eng.status, "exhausted")

    def test_stop_before_start_does_not_run_simulations(self):
        eng = self.engine()
        eng.stop_event.set()
        eng.run(10)
        self.assertEqual(eng.ev.calls, [])
        self.assertEqual(eng.status, "stopped")

    def test_checkpoint_restores_sobol_and_trust_region(self):
        ev = Evaluator()
        first = self.engine(ev, n_init=3)
        first.run(3)
        first.turbo.length = .2
        first.turbo.failure = 2
        first.save_best()
        second = self.engine(ev, n_init=3)
        second.resume()
        self.assertEqual(second.turbo.length, .2)
        self.assertEqual(second.turbo.failure, 2)
        np.testing.assert_equal(first.sobol.draw(1).numpy(), second.sobol.draw(1).numpy())

    def test_invalid_input_fails_before_evaluation(self):
        for kwargs in ({"batch": 0}, {"n_init": -1}, {"device": "typo"}, {"spec_params": []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.engine(**kwargs)
        for cfg in (1, {"metric": "score", "scale": 0},
                    {"metric": "score", "goal": "target"},
                    {"metric": "score", "weight": float("nan")}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                Objective(cfg)

    def test_turbo_first_finite_improvement_is_success(self):
        turbo = TurboState(2)
        turbo.update(1.)
        self.assertEqual(turbo.success, 1)
        self.assertEqual(turbo.failure, 0)

    def test_integer_projection_stays_inside_noninteger_bounds(self):
        space = Space([{"name": "x", "lo": 1.2, "hi": 3.8, "integer": True}])
        self.assertEqual(space.to_physical([0]), {"x": 2})
        self.assertEqual(space.to_physical([1]), {"x": 3})


if __name__ == "__main__":
    unittest.main()
