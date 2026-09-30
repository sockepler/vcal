import copy
import unittest

from optserver.validation import validate_config


class MeasurementConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "name": "measurement_fixture",
            "simulator": "spectre",
            "netlist": "dut.scs",
            "stimuli": "stimuli.scs",
            "tran": {"stop": "1u", "maxstep": "1n"},
            "save": ["in", "out"],
            "params": [{"name": "res", "devices": ["R1"],
                        "attr": "r", "lo": "500", "hi": "3k"}],
            "metrics": {"voltage": "avg(V('out'))"},
            "objective": {"metric": "voltage", "goal": "maximize"},
        }

    def assert_valid(self, config):
        # validation intentionally normalizes a few numeric fields in place;
        # each call receives an isolated copy so subtests cannot leak state.
        validate_config(copy.deepcopy(config))

    def assert_invalid(self, config, message):
        with self.assertRaisesRegex(ValueError, message):
            validate_config(copy.deepcopy(config))

    def test_dc_only_and_mixed_analyses_are_valid(self):
        dc_only = copy.deepcopy(self.base)
        dc_only.pop("tran")
        dc_only["dc"] = {"source": "VIN", "start": "-10m",
                          "stop": "10m", "step": "1m"}
        self.assert_valid(dc_only)

        mixed = copy.deepcopy(self.base)
        mixed["dc"] = {"source": "VIN", "start": "10m",
                        "stop": "-10m", "step": "-1m"}
        mixed["metrics"] = {
            "voltage": "avg(V('out'))",
            "slope": {"analysis": "dc",
                      "expr": "dc_gain(V('out'), V('in'), at=0)"},
        }
        self.assert_valid(mixed)

    def test_hspice_dc_is_rejected(self):
        config = copy.deepcopy(self.base)
        config["simulator"] = "hspice"
        config["dc"] = {"source": "VIN", "start": 0,
                         "stop": 1, "step": 1}
        self.assert_invalid(config, "dc sweeps currently require Spectre")

    def test_dc_direction_and_zero_step_are_rejected(self):
        cases = (
            ({"start": 0, "stop": 1, "step": -0.1},
             "dc step must be nonzero and point from start toward stop"),
            ({"start": 1, "stop": 0, "step": 0.1},
             "dc step must be nonzero and point from start toward stop"),
            ({"start": 0, "stop": 1, "step": 0},
             "dc step must be nonzero and point from start toward stop"),
        )
        for values, message in cases:
            with self.subTest(values=values):
                config = copy.deepcopy(self.base)
                config.pop("tran")
                config["dc"] = {"source": "VIN", **values}
                self.assert_invalid(config, message)

    def test_dc_extra_fields_and_source_format_are_rejected(self):
        extra = copy.deepcopy(self.base)
        extra.pop("tran")
        extra["dc"] = {"source": "VIN", "start": 0, "stop": 1,
                        "step": 0.1, "temperature": 27}
        self.assert_invalid(extra, "dc supports only source, start, stop and step")

        bad_source = copy.deepcopy(self.base)
        bad_source.pop("tran")
        bad_source["dc"] = {"source": "VIN-1", "start": 0,
                             "stop": 1, "step": 0.1}
        self.assert_invalid(bad_source, "dc.source must be a source instance name")

    def test_nonfinite_sweep_and_parameter_bounds_are_rejected(self):
        for field, value in (("start", "nan"), ("stop", "inf"),
                             ("step", "-inf")):
            with self.subTest(field=field):
                config = copy.deepcopy(self.base)
                config.pop("tran")
                config["dc"] = {"source": "VIN", "start": 0,
                                 "stop": 1, "step": 0.1}
                config["dc"][field] = value
                self.assert_invalid(config, r"dc\.%s must be a finite number" % field)

        for field, value in (("lo", "nan"), ("hi", "inf")):
            with self.subTest(field=field):
                config = copy.deepcopy(self.base)
                config["params"][0][field] = value
                self.assert_invalid(config, r"param res\.%s must be a finite number" % field)

    def test_metric_syntax_and_unknown_symbols_are_preflight_errors(self):
        syntax = copy.deepcopy(self.base)
        syntax["metrics"] = {"bad": "avg(V('out')"}
        syntax["objective"] = {"metric": "bad", "goal": "maximize"}
        self.assert_invalid(syntax, "Invalid metric definitions")

        unknown_symbol = copy.deepcopy(self.base)
        unknown_symbol["metrics"] = {"bad": "not_a_metric_symbol + 1"}
        unknown_symbol["objective"] = {"metric": "bad", "goal": "maximize"}
        self.assert_invalid(unknown_symbol, "unknown metric symbol")

    def test_unknown_metric_and_cyclic_dependencies_are_preflight_errors(self):
        unknown_metric = copy.deepcopy(self.base)
        unknown_metric["objective"] = {"metric": "missing", "goal": "maximize"}
        self.assert_invalid(unknown_metric, "objective references unknown metric")

        cyclic = copy.deepcopy(self.base)
        cyclic["metrics"] = {"a": "m['b'] + 1", "b": "m['a'] + 1"}
        cyclic["objective"] = {"metric": "a", "goal": "maximize"}
        self.assert_invalid(cyclic, "cyclic metric dependencies")

    def test_unknown_metric_analysis_is_rejected_before_simulation(self):
        config = copy.deepcopy(self.base)
        config["metrics"] = {
            "bad": {"analysis": "ac", "expr": "avg(V('out'))"},
        }
        config["objective"] = {"metric": "bad", "goal": "maximize"}
        self.assert_invalid(config, "analysis 'ac' is not configured")

    def test_legacy_string_metrics_remain_valid_for_each_default_analysis(self):
        self.assert_valid(self.base)

        dc_only = copy.deepcopy(self.base)
        dc_only.pop("tran")
        dc_only["dc"] = {"source": "VIN", "start": 0,
                          "stop": 1, "step": 0.1}
        self.assert_valid(dc_only)

    def test_metric_window_is_checked_against_configured_stop(self):
        config = copy.deepcopy(self.base)
        config["metrics"]["voltage"] = {
            "expr": "avg(V('out'))", "window": {"start": "100n", "end": "900n"}}
        self.assert_valid(config)
        config["metrics"]["voltage"]["window"]["end"] = "2u"
        self.assert_invalid(config, "within tran.stop")


if __name__ == "__main__":
    unittest.main()
