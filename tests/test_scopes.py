import copy
import math
import unittest

from optlocal.scopes import parameter_scopes, scope_inventory, select_scope


class ScopeHelpersTest(unittest.TestCase):
    def test_parameter_scopes_normalizes_top_and_sorts_unique(self):
        param = {"devices": ["z/M1", "M2", "top/M3", "z/M4", "a/M5"]}
        self.assertEqual(parameter_scopes(param), ("a", "top", "z"))

    def test_inventory_keeps_same_instance_name_in_separate_scopes(self):
        params = [
            {"name": "a", "devices": ["cell_b/M1"], "lo": 0., "hi": 1.},
            {"name": "b", "devices": ["cell_a/M1", "cell_a/M2"],
             "lo": 0., "hi": 1.},
            {"name": "shared", "devices": ["cell_a/M3", "cell_b/M3"],
             "lo": 0., "hi": 1.},
            {"name": "top_only", "devices": ["M1", "top/M2"],
             "lo": 0., "hi": 1.},
        ]
        self.assertEqual(scope_inventory(params), [
            {"scope": "cell_a", "parameters": ["b"],
             "shared": ["shared"], "devices": ["M1", "M2", "M3"]},
            {"scope": "cell_b", "parameters": ["a"],
             "shared": ["shared"], "devices": ["M1", "M3"]},
            {"scope": "top", "parameters": ["top_only"],
             "shared": [], "devices": ["M1", "M2"]},
        ])

    def _params(self):
        return [
            {"name": "a", "lo": 1., "hi": 10., "nominal": 2.,
             "enabled": True, "devices": ["cell_a/M1"]},
            {"name": "b", "lo": 1., "hi": 10., "nominal": 3.,
             "enabled": True, "devices": ["cell_b/M1"]},
            {"name": "shared", "lo": 1., "hi": 10., "nominal": 4.,
             "enabled": True, "devices": ["cell_a/M2", "cell_b/M2"]},
            {"name": "disabled", "lo": 1., "hi": 10., "nominal": 5.,
             "enabled": False, "devices": ["cell_a/M3"]},
            {"name": "raw", "lo": 1., "hi": 10., "nominal": None,
             "enabled": True, "devices": ["cell_b/M3"]},
        ]

    def test_select_all_applies_initial_before_fixed_and_keeps_disabled(self):
        params = self._params()
        active, fixed = select_scope(
            params,
            fixed={"b": 6., "disabled": 7.},
            initial_values={"a": 8., "b": 9., "disabled": 10.},
        )
        self.assertEqual([p["name"] for p in active], ["a", "shared", "raw"])
        self.assertEqual(active[0]["nominal"], 8.)
        self.assertEqual(fixed, {"b": 9., "disabled": 10.})

    def test_select_scope_fixes_other_and_shared_parameters(self):
        active, fixed = select_scope(self._params(), "cell_a")
        self.assertEqual([p["name"] for p in active], ["a"])
        self.assertEqual(fixed, {"b": 3., "shared": 4., "disabled": 5.})

    def test_none_nominal_keeps_original_netlist_value(self):
        active, fixed = select_scope(self._params(), "cell_a")
        self.assertEqual([p["name"] for p in active], ["a"])
        self.assertNotIn("raw", fixed)

    def test_invalid_scope_and_no_active_are_errors(self):
        with self.assertRaisesRegex(ValueError, "unknown scope"):
            select_scope(self._params(), "missing")
        with self.assertRaisesRegex(ValueError, "at least one parameter"):
            select_scope(self._params(), "cell_a", fixed={"a": 2.})

    def test_values_require_finite_bounds_and_integer_constraint(self):
        params = [{"name": "n", "lo": 1., "hi": 4., "nominal": 2.,
                   "integer": True, "devices": ["cell_a/M1"]}]
        for values in ({"n": True}, {"n": math.inf}, {"n": 0.}, {"n": 2.5}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    select_scope(params, fixed=values)
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            select_scope(params, initial_values={"missing": 2.})

    def test_result_does_not_mutate_inputs(self):
        params = self._params()
        fixed = {"b": 6.}
        initial = {"a": 8.}
        before = copy.deepcopy(params)
        active, fixed_values = select_scope(params, fixed=fixed,
                                            initial_values=initial)
        active[0]["devices"].append("cell_a/M9")
        fixed_values["new"] = 1.
        self.assertEqual(params, before)
        self.assertEqual(fixed, {"b": 6.})
        self.assertEqual(initial, {"a": 8.})


if __name__ == "__main__":
    unittest.main()
