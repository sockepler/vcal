import unittest

from optlocal.review import analyze_records


class ReviewTests(unittest.TestCase):
    def test_feasibility_best_and_failure_statistics(self):
        records = [
            {"ok": True, "params": {"w": 1}, "metrics": {"gain": 10, "power": 1}},
            {"ok": True, "params": {"w": 2}, "metrics": {"gain": 20, "power": 9}},
            {"ok": False, "params": {"w": 2}, "error": "Spectre failed"},
            {"ok": True, "params": {"w": 3}, "metrics": {"gain": float("nan"), "power": 1}},
        ]
        result = analyze_records(records, {"metric": "gain", "goal": "maximize"},
                                 [{"metric": "power", "max": 2}])
        self.assertEqual((result["n_ok"], result["n_feasible"], result["n_failed"]), (2, 1, 2))
        self.assertEqual(result["best_record"], records[0])
        self.assertEqual(result["evaluations_since_improvement"], 3)
        self.assertEqual(result["repeated_points"], 1)
        self.assertIn("inspect_failures", result["advice"])

    def test_rank_association_respects_minimization_and_ties(self):
        records = [{"ok": True, "params": {"w": i // 2, "fixed": 2},
                    "metrics": {"power": i // 2}} for i in range(8)]
        result = analyze_records(records, {"metric": "power", "goal": "minimize"},
                                 params=[{"name": "w"}, {"name": "fixed"}])
        self.assertEqual(len(result["associations"]), 1)
        self.assertAlmostEqual(result["associations"][0]["rho"], -1)
        self.assertEqual(result["best_record"], records[0])

    def test_target_tolerance_and_insufficient_data(self):
        record = {"ok": True, "metrics": {"gain": 15}, "params": {"w": 1}}
        result = analyze_records([record], {"metric": "gain", "goal": "target", "target": 10, "tol": 1})
        self.assertEqual(result["n_feasible"], 0)
        self.assertIsNone(result["best_record"])
        self.assertEqual(result["associations"], [])
        self.assertIn("check_constraints", result["advice"])
        self.assertIn("collect_more", result["advice"])


if __name__ == "__main__":
    unittest.main()
