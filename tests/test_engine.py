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


class SeedEvaluator:
    def __init__(self):
        self.calls = []
        self.history_records = []
        self.definition = {
            "params": [
                {"name": "x", "lo": 0., "hi": 1., "nominal": .5},
                {"name": "y", "lo": 0., "hi": 4., "nominal": 2., "integer": True},
                {"name": "disabled", "lo": 0., "hi": 1., "nominal": .2,
                 "enabled": False},
                {"name": "fixed", "lo": 0., "hi": 1., "nominal": .4,
                 "enabled": False},
            ],
            "metrics": ["score", "power"],
            "objective": {"metric": "score", "goal": "maximize"},
            "constraints": [{"metric": "power", "max": 1}],
        }

    def spec(self):
        return self.definition

    def history(self):
        return self.history_records

    def evaluate_batch(self, params, parallel=None):
        self.calls.append([dict(p) for p in params])
        out = []
        for p in params:
            rec = {"params": dict(p), "ok": True,
                   "trial": len(self.history_records),
                   "metrics": {"score": p["x"], "power": .2}}
            self.history_records.append(rec)
            out.append(rec)
        return out


class FlatEvaluator(Evaluator):
    def __init__(self, powers=None):
        super().__init__()
        self.powers = list(powers or [])

    def evaluate_batch(self, params, parallel=None):
        self.calls.append([dict(p) for p in params])
        out = []
        for p in params:
            index = len(self.history_records)
            power = self.powers[index] if index < len(self.powers) else .2
            rec = {"params": dict(p), "ok": True,
                   "trial": index,
                   "metrics": {"score": 0., "power": power}}
            self.history_records.append(rec)
            out.append(rec)
        return out


class SpyProposer:
    name = "spy"

    def __init__(self, value=.1234567):
        self.value = value
        self.calls = []

    def propose(self, X, y, C, batch, *args):
        self.calls.append((len(X), batch))
        return np.full((batch, X.shape[1]), self.value), False


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def engine(self, ev=None, **kwargs):
        eng = Engine(ev or Evaluator(), self.tmp.name, on_log=lambda _: None, **kwargs)
        eng.proposer = Proposer()
        return eng

    def test_initial_seed_is_completed_before_nominal_and_fixed(self):
        ev = SeedEvaluator()
        eng = self.engine(ev, batch=1, n_init=2, fixed={"fixed": .4},
                           initial_points=[{"x": .2}])
        eng.run(2)
        self.assertEqual(len(eng.records), 2)
        seed, nominal = [call[0] for call in ev.calls]
        self.assertEqual(seed["x"], .2)
        self.assertEqual(seed["y"], 2)
        self.assertEqual(seed["fixed"], .4)
        self.assertNotIn("disabled", seed)
        self.assertEqual(nominal["x"], .5)
        self.assertEqual(nominal["y"], 2)
        self.assertEqual(nominal["fixed"], .4)

    def test_invalid_initial_seed_fails_before_any_evaluation(self):
        bad_points = (
            {"unknown": .1},
            {"disabled": .1},
            {"x": True},
            {"x": float("nan")},
            {"x": 2.},
            {"y": 1.5},
            {"fixed": .3},
        )
        for point in bad_points:
            with self.subTest(point=point):
                ev = SeedEvaluator()
                with self.assertRaises(ValueError):
                    self.engine(ev, fixed={"fixed": .4},
                                initial_points=[point])
                self.assertEqual(ev.calls, [])

    def test_initial_seed_uses_spec_params_override_bounds(self):
        spec_params = [{"name": "x", "lo": 2., "hi": 4., "nominal": 3.}]
        bad = Evaluator()
        bad.definition["params"][0].update(lo=0., hi=10., nominal=5.)
        with self.assertRaises(ValueError):
            self.engine(bad, spec_params=spec_params,
                        initial_points=[{"x": 5.}])
        self.assertEqual(bad.calls, [])

        good = Evaluator()
        good.definition["params"][0].update(lo=0., hi=10., nominal=5.)
        eng = self.engine(good, batch=1, n_init=1,
                          spec_params=spec_params,
                          initial_points=[{"x": 3.}])
        eng.run(1)
        self.assertEqual(good.calls, [1])
        self.assertEqual(good.history_records[0]["params"]["x"], 3.)

    def test_duplicate_seeds_and_small_budget_do_not_overrun_budget(self):
        ev = Evaluator()
        eng = self.engine(ev, batch=4, n_init=20,
                          initial_points=[{"x": .2}, {"x": .2}, {"x": .8}])
        eng.run(2)
        self.assertEqual(len(eng.records), 2)
        self.assertEqual(sum(ev.calls), 2)
        self.assertEqual([r["params"]["x"] for r in eng.records], [.2, .8])

    def test_resume_skips_used_seed_and_prioritizes_unused_seed(self):
        ev = Evaluator()
        kwargs = {"batch": 1, "n_init": 1,
                  "initial_points": [{"x": .2}, {"x": .8}]}
        first = self.engine(ev, **kwargs)
        first.run(1)
        second = self.engine(ev, **kwargs)
        self.assertEqual(second.resume(), 1)
        second.run(3)
        points = [r["params"]["x"] for r in ev.history_records]
        self.assertEqual(points, [.2, .8, .5])
        self.assertEqual(points.count(.2), 1)
        self.assertEqual(sum(ev.calls), 3)

    def test_stagnation_uses_global_sobol_on_next_batch_within_budget(self):
        ev = FlatEvaluator()
        eng = self.engine(ev, batch=1, n_init=1, stagnation_rounds=2)
        proposer = SpyProposer()
        eng.proposer = proposer
        eng.run(4)
        summary = eng.summary()
        self.assertEqual(len(eng.records), 4)
        self.assertEqual(sum(len(batch) for batch in ev.calls), 4)
        # The first two optimization batches consult the proposer. Once the
        # threshold is reached, the following batch is generated by Sobol.
        self.assertEqual(len(proposer.calls), 2)
        self.assertNotEqual(eng.records[-1]["params"]["x"], proposer.value)
        self.assertEqual(summary["exploration_restarts"], 1)
        self.assertFalse(summary["exploration_pending"])

    def test_constraint_violation_improvement_resets_stagnation(self):
        ev = FlatEvaluator([3., 3., 2.])
        ev.definition["constraints"] = [{"metric": "power", "max": 0.}]
        eng = self.engine(ev, batch=1, n_init=1, stagnation_rounds=2)
        eng.proposer = SpyProposer()
        eng.run(3)
        summary = eng.summary()
        self.assertEqual(summary["n_feasible"], 0)
        self.assertEqual(summary["stagnation_count"], 0)
        self.assertEqual(summary["exploration_restarts"], 0)
        self.assertFalse(summary["exploration_pending"])

    def test_checkpoint_restores_stagnation_state_and_continues_sobol(self):
        ev = FlatEvaluator()
        kwargs = {"batch": 1, "n_init": 1, "stagnation_rounds": 2}
        first = self.engine(ev, **kwargs)
        first.proposer = SpyProposer()
        first.run(3)
        before = first.summary()
        self.assertEqual(before["stagnation_count"], 2)
        self.assertEqual(before["exploration_restarts"], 0)
        self.assertTrue(before["exploration_pending"])

        resumed = self.engine(ev, **kwargs)
        resumed_proposer = SpyProposer()
        resumed.proposer = resumed_proposer
        self.assertEqual(resumed.resume(), 3)
        restored = resumed.summary()
        self.assertEqual(restored["stagnation_count"], before["stagnation_count"])
        self.assertEqual(restored["exploration_restarts"], before["exploration_restarts"])
        self.assertEqual(restored["exploration_pending"], before["exploration_pending"])

        resumed.run(4)
        self.assertEqual(len(resumed.records), 4)
        self.assertEqual(resumed_proposer.calls, [])
        self.assertFalse(resumed.summary()["exploration_pending"])
        self.assertEqual(resumed.summary()["exploration_restarts"], 1)
        self.assertNotEqual(resumed.records[-1]["params"]["x"], resumed_proposer.value)

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

    def test_resume_skips_changed_inactive_background_but_accepts_nominal(self):
        ev = Evaluator()
        ev.definition["params"].append({"name": "z", "lo": 0., "hi": 1.,
                                         "nominal": .4})
        ev.history_records = [
            {"trial": 0, "params": {"x": .2, "z": .6}, "ok": True,
             "metrics": {"score": .2, "power": .1}},
            {"trial": 1, "params": {"x": .3, "z": .4}, "ok": True,
             "metrics": {"score": .3, "power": .1}},
        ]
        eng = self.engine(ev, spec_params=[ev.definition["params"][0]])
        self.assertEqual(eng.resume(), 1)
        self.assertEqual(len(eng.records), 1)
        self.assertEqual(eng.records[0]["trial"], 1)

    def test_resume_accepts_omitted_inactive_background_without_nominal(self):
        ev = Evaluator()
        ev.definition["params"].append({"name": "z", "lo": 0., "hi": 1.,
                                         "nominal": None})
        ev.history_records = [
            {"trial": 0, "params": {"x": .2}, "ok": True,
             "metrics": {"score": .2, "power": .1}},
            {"trial": 1, "params": {"x": .3, "z": .6}, "ok": True,
             "metrics": {"score": .3, "power": .1}},
        ]
        eng = self.engine(ev, spec_params=[ev.definition["params"][0]])
        self.assertEqual(eng.resume(), 1)
        self.assertEqual(len(eng.records), 1)
        self.assertEqual(eng.records[0]["trial"], 0)

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
