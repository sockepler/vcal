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
