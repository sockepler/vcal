import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "client"))

import optlocal.__main__ as cli
from optlocal.i18n import configure_language, get_language


class CliTests(unittest.TestCase):
    def setUp(self):
        self._old_settings_file = os.environ.get("VCAL_SETTINGS_FILE")
        self._old_vcal_lang = os.environ.get("VCAL_LANG")
        self._settings_tmp = tempfile.TemporaryDirectory(dir="/tmp")
        os.environ["VCAL_SETTINGS_FILE"] = str(
            Path(self._settings_tmp.name) / "settings.json")
        os.environ.pop("VCAL_LANG", None)
        self._old_language = get_language()
        configure_language("zh")

    def tearDown(self):
        if self._old_vcal_lang is None:
            os.environ.pop("VCAL_LANG", None)
        else:
            os.environ["VCAL_LANG"] = self._old_vcal_lang
        if self._old_settings_file is None:
            os.environ.pop("VCAL_SETTINGS_FILE", None)
        else:
            os.environ["VCAL_SETTINGS_FILE"] = self._old_settings_file
        configure_language(self._old_language)
        self._settings_tmp.cleanup()

    @staticmethod
    def _help(argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with unittest.TestCase().assertRaises(SystemExit) as cm:
                cli.main(argv)
        if cm.exception.code != 0:
            raise AssertionError("help exited with %r" % cm.exception.code)
        return output.getvalue()

    def test_python_help_is_translated_in_all_languages(self):
        outputs = {
            "zh": self._help(["--lang", "zh", "--help"]),
            "ja": self._help(["--lang", "ja", "--help"]),
            "en": self._help(["--lang", "en", "--help"]),
        }
        self.assertIn("启动图形界面", outputs["zh"])
        self.assertIn("GUI を起動", outputs["ja"])
        self.assertIn("Launch graphical interface", outputs["en"])
        for text in outputs.values():
            self.assertIn("--lang LANG", text)
            self.assertIn("scopes", text)
            self.assertIn("review", text)
            self.assertIn("gmid", text)

    def test_lang_is_accepted_before_or_after_subcommand_and_forwards_gui(self):
        calls = []
        fake_gui = types.ModuleType("optlocal.gui")
        fake_gui.main = lambda language=None: calls.append(language)
        with mock.patch.dict(sys.modules, {"optlocal.gui": fake_gui}):
            self.assertEqual(cli.main(["--lang", "ja", "gui"]), 0)
            self.assertEqual(cli.main(["gui", "--lang", "en"]), 0)
            self.assertEqual(cli.main(["gui"]), 0)
        self.assertEqual(calls, ["ja", "en", None])

    def test_invalid_language_reports_value_and_status_two(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--lang", "xx", "gui"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("xx", err.getvalue())

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--lang=", "gui"])
        self.assertEqual(cm.exception.code, 2)

        os.environ["VCAL_LANG"] = "bad-env"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--help"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("bad-env", err.getvalue())

    def test_shell_help_does_not_require_python(self):
        env = os.environ.copy()
        env["OPTLOCAL_PYTHON"] = "/path/that/does/not/exist"
        expected = {"zh": "用法", "ja": "使い方", "en": "Usage"}
        for language, marker in expected.items():
            result = subprocess.run(
                [str(ROOT / "vcal"), "--lang", language, "--help"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(marker, result.stdout)
            self.assertIn("vcal check", result.stdout)

        result = subprocess.run(
            [str(ROOT / "vcal"), "install", "--lang", "en", "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage: vcal install", result.stdout)

    def test_shell_unknown_language_is_status_two_without_python(self):
        env = os.environ.copy()
        env["OPTLOCAL_PYTHON"] = "/path/that/does/not/exist"
        result = subprocess.run(
            [str(ROOT / "vcal"), "--lang", "xx", "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("xx", result.stderr)

    def test_run_rejects_non_positive_counts(self):
        with self.assertRaises(SystemExit) as cm:
            cli.main(["run", "missing.yaml", "--budget", "0"])
        self.assertEqual(cm.exception.code, 2)

        with self.assertRaises(SystemExit) as cm:
            cli.main(["run", "missing.yaml", "--device", "invalid"])
        self.assertEqual(cm.exception.code, 2)

        with self.assertRaises(SystemExit) as cm:
            cli.main(["run", "missing.yaml", "--resume", "--no-resume"])
        self.assertEqual(cm.exception.code, 2)

    def test_check_uses_only_circuit_and_can_emit_json(self):
        fake_pkg = types.ModuleType("optserver")
        fake_pkg.__path__ = []
        fake_circuit_module = types.ModuleType("optserver.circuit")
        constructed = []

        class FakeCircuit:
            def __init__(self, path):
                constructed.append(path)

            def spec(self):
                return {"circuit": "fake", "params": [], "metrics": []}

        fake_circuit_module.Circuit = FakeCircuit
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            cfg = Path(tmp) / "circuit.yaml"
            cfg.write_text("this is handled by the fake Circuit\n")
            fake_work = Path(tmp) / "work"

            original_import = builtins.__import__

            def reject_torch(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("check must not import torch")
                return original_import(name, *args, **kwargs)

            out = io.StringIO()
            with mock.patch.dict(
                sys.modules,
                {"optserver": fake_pkg,
                 "optserver.circuit": fake_circuit_module},
            ), mock.patch("builtins.__import__", reject_torch), \
                    contextlib.redirect_stdout(out):
                code = cli.main(["check", str(cfg), "--json"])

            self.assertEqual(code, 0)
            self.assertEqual(constructed, [str(cfg)])
            self.assertFalse(fake_work.exists())
            self.assertEqual(json.loads(out.getvalue())["circuit"], "fake")

    def test_check_preserves_multiline_validation_errors(self):
        fake_pkg = types.ModuleType("optserver")
        fake_pkg.__path__ = []
        fake_circuit_module = types.ModuleType("optserver.circuit")

        class InvalidCircuit:
            def __init__(self, path):
                raise ValueError(
                    "Invalid circuit configuration:\n"
                    "- missing device name\n"
                    "- metric references unknown node"
                )

        fake_circuit_module.Circuit = InvalidCircuit
        err = io.StringIO()
        with mock.patch.dict(
            sys.modules,
            {"optserver": fake_pkg,
             "optserver.circuit": fake_circuit_module},
        ), contextlib.redirect_stderr(err):
            code = cli.main(["check", "broken.yaml"])

        self.assertEqual(code, 2)
        self.assertIn("missing device name", err.getvalue())
        self.assertIn("metric references unknown node", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_gmid_parses_engineering_values_and_emits_stable_json(self):
        gmid_module = types.ModuleType("optlocal.gmid")
        calls = []

        class FakeTable:
            @classmethod
            def load(cls, path):
                calls.append(("load", path))
                return cls()

            def size_for(self, length, vds, vsb, *, gmid, ids=None, gm=None):
                calls.append(("size_for", length, vds, vsb, gmid, ids, gm))
                return {"z": 2.0, "a": 1.0}

        gmid_module.GmIdTable = FakeTable
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {"optlocal.gmid": gmid_module}), \
                contextlib.redirect_stdout(out):
            code = cli.main([
                "gmid", "table.csv", "--length", "180n", "--vds", "0.9",
                "--vsb", "0", "--gmid", "15", "--id", "20u", "--json",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), {"a": 1.0, "z": 2.0})
        self.assertEqual(calls[0], ("load", "table.csv"))
        self.assertEqual(calls[1][0], "size_for")
        for actual, expected in zip(calls[1][1:],
                                    (180e-9, 0.9, 0.0, 15.0, 20e-6, None)):
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertAlmostEqual(actual, expected)

        with self.assertRaises(SystemExit) as missing:
            cli.main(["gmid", "table.csv", "--length", "1", "--vds", "1",
                      "--gmid", "10"])
        self.assertEqual(missing.exception.code, 2)
        with self.assertRaises(SystemExit) as both:
            cli.main(["gmid", "table.csv", "--length", "1", "--vds", "1",
                      "--gmid", "10", "--id", "1", "--gm", "2"])
        self.assertEqual(both.exception.code, 2)

    def test_scopes_and_review_are_read_only_circuit_operations(self):
        circuit_module = types.ModuleType("optserver.circuit")
        fake_pkg = types.ModuleType("optserver")
        fake_pkg.__path__ = []
        constructed = []
        spec = {
            "params": [{"name": "w", "enabled": True,
                        "devices": ["amp/M1"]}],
            "objective": {"metric": "gain", "goal": "maximize"},
            "constraints": [{"metric": "power", "max": 2}],
        }

        class FakeCircuit:
            def __init__(self, path):
                constructed.append(path)

            def spec(self):
                return spec

        circuit_module.Circuit = FakeCircuit
        scopes_module = types.ModuleType("optlocal.scopes")
        scopes_module.scope_inventory = lambda params: [{
            "scope": "amp", "parameters": [params[0]["name"]],
            "shared": [], "devices": ["M1"],
        }]
        evaluator_module = types.ModuleType("optlocal.evaluator")
        review_module = types.ModuleType("optlocal.review")
        review_calls = []
        evaluator_module.read_history = lambda path: [{"path": path}]
        review_module.analyze_records = lambda records, objective, constraints, params: (
            review_calls.append((records, objective, constraints, params)) or
            {"n": len(records), "ok": True}
        )
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp, \
                mock.patch.dict(sys.modules, {
                    "optserver": fake_pkg,
                    "optserver.circuit": circuit_module,
                    "optlocal.scopes": scopes_module,
                    "optlocal.evaluator": evaluator_module,
                    "optlocal.review": review_module,
                }):
            cfg = str(Path(tmp) / "circuit.yaml")
            history = str(Path(tmp) / "history.jsonl")
            Path(cfg).write_text("ignored")
            Path(history).write_text("ignored")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["scopes", cfg, "--json"]), 0)
            self.assertEqual(json.loads(out.getvalue())[0]["scope"], "amp")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main([
                    "review", history, "--config", cfg, "--json",
                ]), 0)
            self.assertEqual(json.loads(out.getvalue())["n"], 1)
        self.assertEqual(constructed, [cfg, cfg])
        self.assertEqual(review_calls[0][1:],
                         (spec["objective"], spec["constraints"], spec["params"]))

    def test_run_passes_scope_seed_and_stagnation_to_engine(self):
        engine_module = types.ModuleType("optlocal.engine")
        evaluator_module = types.ModuleType("optlocal.evaluator")
        engines = []
        spec = {
            "params": [
                {"name": "a", "lo": 0., "hi": 10., "nominal": 1.,
                 "enabled": True, "devices": ["cell_a/M1"]},
                {"name": "b", "lo": 0., "hi": 10., "nominal": 2.,
                 "enabled": True, "devices": ["cell_b/M1"]},
                {"name": "shared", "lo": 0., "hi": 10., "nominal": 3.,
                 "enabled": True,
                 "devices": ["cell_a/M2", "cell_b/M2"]},
            ],
            "fixed": {}, "objective": {"metric": "gain", "goal": "maximize"},
            "constraints": [], "optimizer": {"stagnation_rounds": 8},
        }

        class FakeEvaluator:
            def __init__(self, path, max_jobs, workroot):
                self.root = tempfile.gettempdir()

            def spec(self):
                return spec

        class FakeEngine:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                engines.append(self)

            def resume(self):
                pass

            def run(self, budget):
                return 0, {"trial": 0, "metrics": {"gain": 1.0}}

            def feasible_mask(self):
                return [True]

        engine_module.Engine = FakeEngine
        evaluator_module.LocalEvaluator = FakeEvaluator
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp, \
                mock.patch.dict(sys.modules, {
                    "optlocal.engine": engine_module,
                    "optlocal.evaluator": evaluator_module,
                }), mock.patch("optlocal.__main__._emit", return_value=None):
            points = Path(tmp) / "points.json"
            points.write_text(json.dumps([{
                "a": 4., "b": 9., "shared": 8.,
            }]))
            code = cli.main([
                "run", str(Path(tmp) / "circuit.yaml"), "--scope", "cell_a",
                "--initial-points", str(points), "--stagnation-rounds", "2",
                "--seed", "17", "--budget", "1", "--batch", "1",
                "--init", "1", "--dkl-after", "1", "--no-resume",
            ])
        self.assertEqual(code, 0)
        kwargs = engines[0].kwargs
        self.assertEqual(kwargs["seed"], 17)
        self.assertEqual(kwargs["stagnation_rounds"], 2)
        self.assertEqual([p["name"] for p in kwargs["spec_params"]], ["a"])
        self.assertEqual(kwargs["fixed"], {"b": 2.0, "shared": 3.0})
        self.assertEqual(kwargs["initial_points"],
                         [{"a": 4.0, "b": 2.0, "shared": 3.0}])

    def test_keyboard_interrupt_saves_best_and_returns_130(self):
        engine_module = types.ModuleType("optlocal.engine")
        evaluator_module = types.ModuleType("optlocal.evaluator")
        engines = []

        class FakeEvaluator:
            def __init__(self, path, max_jobs, workroot):
                self.root = tempfile.gettempdir()

        class FakeEngine:
            def __init__(self, *args, **kwargs):
                self.saved = False
                engines.append(self)

            def resume(self):
                pass

            def run(self, budget):
                raise KeyboardInterrupt

            def save_best(self):
                self.saved = True

        engine_module.Engine = FakeEngine
        evaluator_module.LocalEvaluator = FakeEvaluator
        with mock.patch.dict(
            sys.modules,
            {"optlocal.engine": engine_module,
             "optlocal.evaluator": evaluator_module,
             "yaml": types.SimpleNamespace(safe_load=lambda value: None)},
        ):
            code = cli.main(["run", "missing.yaml", "--no-resume"])

        self.assertEqual(code, 130)
        self.assertEqual(len(engines), 1)
        self.assertTrue(engines[0].saved)


if __name__ == "__main__":
    unittest.main()
