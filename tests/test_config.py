import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from optserver.circuit import Circuit
from optserver.validation import validate_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "dut.scs").write_text("simulator lang=spectre\nR1 (in 0) resistor r=1k\nR2 (out 0) resistor r=2k\n")
        (self.root / "stimuli.scs").write_text("simulator lang=spectre\nV1 (in 0) vsource dc=1\n")
        (self.root / "models.scs").write_text("// test fixture model file\n")
        self.cfg = {
            "name": "test", "simulator": "spectre", "netlist": "dut.scs",
            "stimuli": "stimuli.scs", "tran": {"stop": "1u", "maxstep": "1n"},
            "save": ["in"], "params": [{"name": "res", "devices": ["R1", "R2"],
                                        "attr": "r", "lo": "500", "hi": "3k"}],
            "metrics": {"voltage": "avg(V('in'))"},
            "objective": {"metric": "voltage", "goal": "maximize"},
        }

    def circuit(self):
        path = self.root / "circuit.yaml"
        path.write_text(yaml.safe_dump(self.cfg))
        return Circuit(str(path))

    def test_yaml_relative_paths_work_from_other_directory(self):
        ckt = self.circuit()
        self.assertEqual(ckt.cfg["netlist"], str(self.root / "dut.scs"))
        self.assertEqual(ckt.cfg["stimuli"], str(self.root / "stimuli.scs"))
        self.assertEqual(ckt.spec()["params"][0]["nominal"], 1000)

    def test_spectre_model_sections_reach_generated_deck(self):
        self.cfg["libs"] = [{"path": "models.scs", "section": "tt"}]
        ckt = self.circuit()
        ckt.write_tb(self.tmp.name)
        text = (self.root / "tb.scs").read_text()
        self.assertIn('include "%s" section=tt' % (self.root / "models.scs"), text)

    def test_missing_second_device_is_detected_before_simulation(self):
        self.cfg["params"][0]["devices"][1] = "Missing"
        with self.assertRaisesRegex(ValueError, "device Missing not found"):
            self.circuit()

    def test_errors_are_aggregated(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["params"][0].update(lo=0, hi=0, log=True)
        cfg["objective"]["metric"] = "unknown"
        cfg["constraints"] = [{"metric": "voltage"}]
        with self.assertRaises(ValueError) as error:
            validate_config(cfg)
        text = str(error.exception)
        self.assertIn("lo must be smaller", text)
        self.assertIn("log bounds", text)
        self.assertIn("unknown metric", text)
        self.assertIn("constraint needs", text)

    def test_disabled_parameters_preserve_fixed_value(self):
        self.cfg["params"][0]["enabled"] = False
        self.cfg["fixed"] = {"res": 1500}
        ckt = self.circuit()
        self.assertFalse(ckt.spec()["params"][0]["enabled"])
        self.assertEqual(ckt.spec()["fixed"], {"res": 1500})

    def test_fixed_mapping_also_disables_parameter_in_gui_spec(self):
        self.cfg["fixed"] = {"res": 1500}
        self.assertFalse(self.circuit().spec()["params"][0]["enabled"])

    def test_initial_points_parse_engineering_units_and_round_trip_in_spec(self):
        self.cfg["initial_points"] = [{"res": "1.5k"}]
        self.cfg["optimizer"] = {"stagnation_rounds": 2}
        spec = self.circuit().spec()
        self.assertEqual(spec["initial_points"], [{"res": 1500.}])
        self.assertEqual(spec["optimizer"], {"stagnation_rounds": 2})

    def test_new_spec_fields_are_copy_isolated(self):
        self.cfg["initial_points"] = [{"res": "1.5k"}]
        self.cfg["optimizer"] = {"stagnation_rounds": 2}
        ckt = self.circuit()
        first = ckt.spec()
        first["initial_points"][0]["res"] = 2200.
        first["optimizer"]["stagnation_rounds"] = 9
        second = ckt.spec()
        self.assertEqual(second["initial_points"], [{"res": 1500.}])
        self.assertEqual(second["optimizer"], {"stagnation_rounds": 2})

    def test_invalid_initial_points_are_rejected(self):
        cases = (
            ({"res": "1k"}, "initial_points must be a list"),
            ([{"unknown": "1k"}], "unknown parameter"),
            ([{"res": "4k"}], "outside parameter bounds"),
            ([{"res": True}], "must be a finite number"),
        )
        for points, message in cases:
            with self.subTest(points=points):
                self.cfg["initial_points"] = points
                with self.assertRaisesRegex(ValueError, message):
                    self.circuit()

        self.cfg["params"][0]["integer"] = True
        self.cfg["initial_points"] = [{"res": 1000.5}]
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.circuit()

    def test_initial_point_conflicting_with_fixed_is_rejected(self):
        self.cfg["fixed"] = {"res": "1.5k"}
        self.cfg["initial_points"] = [{"res": "2k"}]
        with self.assertRaisesRegex(ValueError, "initial_points conflicts with fixed parameter"):
            self.circuit()

    def test_invalid_stagnation_rounds_are_rejected(self):
        for value in (True, 1.5, "2", -1):
            with self.subTest(value=value):
                self.cfg["optimizer"] = {"stagnation_rounds": value}
                with self.assertRaisesRegex(ValueError,
                                           "optimizer.stagnation_rounds must be a nonnegative integer"):
                    self.circuit()

    def test_legacy_yaml_without_new_fields_remains_compatible(self):
        spec = self.circuit().spec()
        self.assertNotIn("initial_points", spec)
        self.assertNotIn("optimizer", spec)
        self.assertEqual(spec["params"][0]["nominal"], 1000.)

    def test_invalid_fixed_value_and_missing_stimuli_fail_early(self):
        self.cfg["fixed"] = {"res": 10000}
        with self.assertRaisesRegex(ValueError, "outside parameter bounds"):
            self.circuit()
        self.cfg.pop("fixed")
        self.cfg["stimuli"] = "missing.scs"
        with self.assertRaisesRegex(ValueError, "stimuli file not found"):
            self.circuit()

    def test_timeout_names_actual_simulator(self):
        ckt = self.circuit()
        ckt.simulator = "hspice"
        with patch.object(ckt, "apply_params"), patch.object(ckt, "write_tb"), \
             patch.object(ckt, "_run_hspice", side_effect=subprocess.TimeoutExpired("hspice", 1)):
            result = ckt.evaluate({}, str(self.root / "trial"))
        self.assertEqual(result["error"], "hspice timeout")


if __name__ == "__main__":
    unittest.main()
