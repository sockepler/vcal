import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optserver.simulator_env import simulation_environment


class SimulationEnvironmentTests(unittest.TestCase):
    def test_default_keeps_launcher_cleared_library_path_and_parent_unchanged(self):
        with patch.dict(os.environ, {
            "VCAL_ENV_TEST_MARKER": "keep-me",
        }, clear=False):
            # The launcher already clears LD_LIBRARY_PATH for Python.  With
            # no explicit simulator override, the helper keeps that absence.
            os.environ.pop("LD_LIBRARY_PATH", None)
            os.environ.pop("VCAL_SIM_LD_LIBRARY_PATH", None)
            before = dict(os.environ)
            env = simulation_environment()

            self.assertNotIn("LD_LIBRARY_PATH", env)
            self.assertNotIn("VCAL_SIM_LD_LIBRARY_PATH", env)
            self.assertEqual(env["VCAL_ENV_TEST_MARKER"], "keep-me")
            self.assertEqual(dict(os.environ), before)

    def test_explicit_nonempty_path_is_restored_only_in_child_mapping(self):
        with patch.dict(os.environ, {
            "LD_LIBRARY_PATH": "python-only",
            "VCAL_SIM_LD_LIBRARY_PATH": "cadence/lib:cadence/lib64",
            "VCAL_ENV_TEST_MARKER": "keep-me",
        }, clear=False):
            before = dict(os.environ)
            env = simulation_environment()

            self.assertEqual(env["LD_LIBRARY_PATH"], "cadence/lib:cadence/lib64")
            self.assertEqual(env["VCAL_SIM_LD_LIBRARY_PATH"],
                             "cadence/lib:cadence/lib64")
            self.assertEqual(env["VCAL_ENV_TEST_MARKER"], "keep-me")
            env["LD_LIBRARY_PATH"] = "changed-child-only"
            self.assertEqual(dict(os.environ), before)

    def test_explicit_empty_path_is_preserved_as_an_empty_child_value(self):
        with patch.dict(os.environ, {
            "LD_LIBRARY_PATH": "python-only",
            "VCAL_SIM_LD_LIBRARY_PATH": "",
        }, clear=False):
            before = dict(os.environ)
            env = simulation_environment()

            self.assertIn("LD_LIBRARY_PATH", env)
            self.assertEqual(env["LD_LIBRARY_PATH"], "")
            self.assertEqual(env["VCAL_SIM_LD_LIBRARY_PATH"], "")
            self.assertEqual(dict(os.environ), before)

    def test_spectre_and_psf_child_processes_receive_simulation_environment(self):
        from optserver import circuit, psfread

        with patch.dict(os.environ, {
            "LD_LIBRARY_PATH": "python-only",
            "VCAL_SIM_LD_LIBRARY_PATH": "cadence/lib",
        }, clear=False):
            fake = type("Completed", (), {"returncode": 0,
                                            "stderr": "", "stdout": ""})()
            with tempfile.TemporaryDirectory() as root:
                Path(root, "tb.log").write_text(
                    "spectre completes with 0 errors\n")
                raw = Path(root, "tb.raw")
                raw.mkdir()
                Path(raw, "tran.tran.tran").write_bytes(b"fixture")
                with patch.object(circuit.subprocess, "run", return_value=fake) as run, \
                        patch.object(psfread, "read_tran_psfbin",
                                     return_value=([], {})):
                    obj = object.__new__(circuit.Circuit)
                    obj.cfg = {"spectre": {"mt": 1, "timeout_s": 1},
                               "tran": {"stop": "1u", "maxstep": "1n"},
                               "metrics": {"m": "avg(V('out'))"}}
                    obj.metrics = obj.cfg["metrics"]
                    obj.simulator = "spectre"
                    result = obj._run_spectre(root)
                self.assertEqual(result, ([], {}))
                child_env = run.call_args.kwargs["env"]
                self.assertEqual(child_env["LD_LIBRARY_PATH"], "cadence/lib")

            with tempfile.TemporaryDirectory() as root:
                Path(root, "tb.lis").write_text("ok\n")
                Path(root, "tb.tr0").write_bytes(b"fixture")
                with patch.object(circuit.subprocess, "run", return_value=fake) as run, \
                        patch.object(circuit.tr0read, "read_tr0",
                                     return_value=([], {})):
                    obj = object.__new__(circuit.Circuit)
                    obj.cfg = {"hspice": {"mt": 1, "timeout_s": 1}}
                    result = obj._run_hspice(root)
                self.assertEqual(result, ([], {}))
                child_env = run.call_args.kwargs["env"]
                self.assertEqual(child_env["LD_LIBRARY_PATH"], "cadence/lib")

            with tempfile.TemporaryDirectory() as root:
                output = Path(root, "tran_ascii.psf")

                def convert(_command, **_kwargs):
                    output.write_text("fixture")
                    return fake

                with patch.object(psfread.subprocess, "run",
                                  side_effect=convert) as run:
                    psfread.psfbin_to_ascii("tran.tran.tran", str(output))
                child_env = run.call_args.kwargs["env"]
                self.assertEqual(child_env["LD_LIBRARY_PATH"], "cadence/lib")

    def test_launcher_clears_python_path_and_preserves_default_sim_path(self):
        self._assert_launcher_environment(
            original_library="cadence/original",
            explicit_sim=None,
            expected_sim="cadence/original")

    def test_launcher_explicit_sim_path_wins_including_empty_override(self):
        self._assert_launcher_environment(
            original_library="cadence/original",
            explicit_sim="cadence/override",
            expected_sim="cadence/override")
        self._assert_launcher_environment(
            original_library="cadence/original",
            explicit_sim="",
            expected_sim="")

    def _assert_launcher_environment(self, *, original_library,
                                     explicit_sim, expected_sim):
        root = Path(__file__).resolve().parents[1]
        launcher = root / "vcal"
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake-python"
            fake.write_text(
                "#!/bin/sh\n"
                "if [ \"${LD_LIBRARY_PATH+x}\" = x ]; then\n"
                "  printf 'PY_LD=%s\\n' \"$LD_LIBRARY_PATH\"\n"
                "else\n"
                "  printf 'PY_LD=<unset>\\n'\n"
                "fi\n"
                "if [ \"${VCAL_SIM_LD_LIBRARY_PATH+x}\" = x ]; then\n"
                "  printf 'SIM_LD=%s\\n' \"$VCAL_SIM_LD_LIBRARY_PATH\"\n"
                "else\n"
                "  printf 'SIM_LD=<unset>\\n'\n"
                "fi\n")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env["OPTLOCAL_PYTHON"] = str(fake)
            env["LD_LIBRARY_PATH"] = original_library
            if explicit_sim is None:
                env.pop("VCAL_SIM_LD_LIBRARY_PATH", None)
            else:
                env["VCAL_SIM_LD_LIBRARY_PATH"] = explicit_sim
            completed = subprocess.run(
                [str(launcher), "check", "unused.yaml"],
                cwd=str(root), env=env, capture_output=True, text=True,
                check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PY_LD=<unset>\n", completed.stdout)
        self.assertIn("SIM_LD=%s\n" % expected_sim, completed.stdout)


if __name__ == "__main__":
    unittest.main()
