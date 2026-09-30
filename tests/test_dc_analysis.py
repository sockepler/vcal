import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from optserver.circuit import Circuit
from optserver import psfread


def _ckt(config):
    """Build the circuit methods under test without loading a real netlist."""
    circuit = object.__new__(Circuit)
    circuit.cfg = config
    circuit.name = config.get("name", "fixture")
    circuit.simulator = config.get("simulator", "spectre")
    circuit.metrics = config["metrics"]
    circuit.fixed = {}
    circuit._pdk = None
    return circuit


def _psf(rows, *, sweep="dc", traces=("in", "out"), trace_suffix="\"V\""):
    lines = [
        "HEADER",
        "\"fixture\"",
        "TYPE",
        "\"sweep\" FLOAT DOUBLE",
        "SWEEP",
        '\"%s\" \"sweep\" PROP(' % sweep,
        "    \"points\" INTEGER 3",
        ")",
        "TRACE",
    ]
    lines.extend('\"%s\" %s' % (name, trace_suffix) for name in traces)
    lines.append("VALUE")
    for values in rows:
        lines.extend('"%s" %s' % item for item in zip((sweep, *traces), values))
    lines.append("END")
    return "\n".join(lines) + "\n"


class PsfSweepTests(unittest.TestCase):
    def test_named_dc_rows_and_descending_axis(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "dc.dc"
            path.write_text(_psf([
                ("1", "2", "3"),
                ("0", "1", "2"),
                ("-1", "0", "1"),
            ]))
            axis, signals = psfread.read_sweep(path)
        np.testing.assert_array_equal(axis, [1., 0., -1.])
        np.testing.assert_array_equal(signals["in"], [2., 1., 0.])
        np.testing.assert_array_equal(signals["out"], [3., 2., 1.])

    def test_incomplete_and_complex_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            incomplete = Path(root) / "incomplete.psf"
            incomplete.write_text(_psf([("0", "1", "2"), ("1", "2")]))
            with self.assertRaisesRegex(RuntimeError, "incomplete VALUE row"):
                psfread.read_sweep(incomplete)

            complex_trace = Path(root) / "complex.psf"
            complex_trace.write_text(_psf([("0", "1", "2")],
                                          trace_suffix="COMPLEX"))
            with self.assertRaisesRegex(RuntimeError, "complex"):
                psfread.read_sweep(complex_trace)

    def test_binary_conversion_uses_distinct_analysis_intermediates(self):
        calls = []

        def convert(_binary, outpath):
            calls.append(outpath)
            Path(outpath).write_text(_psf([("0", "1", "2"),
                                           ("1", "2", "3")]))

        with tempfile.TemporaryDirectory() as root, \
                patch.object(psfread, "psfbin_to_ascii", side_effect=convert):
            psfread.read_sweep_psfbin("tran.tran.tran", root, name="tran")
            psfread.read_sweep_psfbin("dc.dc", root, name="dc")
        self.assertEqual([Path(path).name for path in calls],
                         ["tran_ascii.psf", "dc_ascii.psf"])


class CircuitDcTests(unittest.TestCase):
    def base_config(self, metrics=None, *, analyses="dc"):
        config = {
            "name": "fixture",
            "simulator": "spectre",
            "stimuli": "stimuli.scs",
            "libs": [],
            "save": ["in", "out"],
            "metrics": metrics or {"slope": "dc_gain(V('out'), V('in'), at=0)"},
        }
        if analyses in ("tran", "both"):
            config["tran"] = {"stop": "1u", "maxstep": "1n"}
        if analyses in ("dc", "both"):
            config["dc"] = {"source": "VIN", "start": "-1m",
                             "stop": "1m", "step": "1m"}
        return config

    def test_dc_deck_omits_transient_and_uses_dc_syntax(self):
        circuit = _ckt(self.base_config())
        with tempfile.TemporaryDirectory() as root:
            circuit.write_tb(root)
            text = Path(root, "tb.scs").read_text()
        self.assertNotIn("tran1 tran", text)
        self.assertIn("dc1 dc dev=VIN param=dc start=-1m stop=1m step=1m", text)

    def test_both_requested_analyses_are_in_one_deck(self):
        metrics = {
            "transient": "avg(V('out'))",
            "slope": {"analysis": "dc",
                      "expr": "dc_gain(V('out'), V('in'), at=0)"},
        }
        circuit = _ckt(self.base_config(metrics, analyses="both"))
        with tempfile.TemporaryDirectory() as root:
            circuit.write_tb(root)
            text = Path(root, "tb.scs").read_text()
        self.assertIn("tran1 tran stop=1u maxstep=1n", text)
        self.assertIn("dc1 dc dev=VIN param=dc start=-1m stop=1m step=1m", text)

    def test_spectre_runs_once_and_reads_both_raw_sweeps(self):
        circuit = _ckt(self.base_config({
            "transient": "avg(V('out'))",
            "slope": {"analysis": "dc", "expr": "dc_gain(V('out'), V('in'), at=0)"},
        }, analyses="both"))
        calls = []

        def read_sweep(binary, workdir, name="sweep", **kwargs):
            calls.append((binary, name))
            axis = np.asarray([-1., 0., 1.])
            return axis, {"in": axis, "out": 2 * axis}

        with tempfile.TemporaryDirectory() as root:
            raw = Path(root, "tb.raw")
            raw.mkdir()
            Path(raw, "tran.tran.tran").write_bytes(b"tran")
            Path(raw, "dc.dc").write_bytes(b"dc")
            Path(root, "tb.log").write_text("spectre completes with 0 errors\n")
            completed = SimpleNamespace(returncode=0, stderr="", stdout="")
            with patch("optserver.circuit.subprocess.run", return_value=completed) as run, \
                    patch.object(psfread, "read_sweep_psfbin", side_effect=read_sweep):
                result = circuit._run_spectre(root)
        self.assertEqual(run.call_count, 1)
        self.assertEqual([name for _path, name in calls], ["tran", "dc"])
        self.assertEqual(set(result), {"tran", "dc"})

    def test_transient_only_keeps_legacy_tuple_reader(self):
        circuit = _ckt(self.base_config({"mean": "avg(V('out'))"}, analyses="tran"))
        expected = (np.asarray([0., 1.]), {"out": np.asarray([1., 2.])})
        with tempfile.TemporaryDirectory() as root:
            raw = Path(root, "tb.raw")
            raw.mkdir()
            Path(raw, "tran.tran.tran").write_bytes(b"tran")
            Path(root, "tb.log").write_text("spectre completes with 0 errors\n")
            completed = SimpleNamespace(returncode=0, stderr="", stdout="")
            with patch("optserver.circuit.subprocess.run", return_value=completed), \
                    patch.object(psfread, "read_tran_psfbin", return_value=expected) as reader:
                result = circuit._run_spectre(root)
        self.assertIs(result, expected)
        reader.assert_called_once()

    def test_evaluate_passes_datasets_and_records_dynamic_signature(self):
        metrics = {
            "slope": {"analysis": "dc",
                      "expr": "dc_gain(V('out'), V('in'), at=0)"},
        }
        circuit = _ckt(self.base_config(metrics, analyses="both"))
        circuit.cfg["corners"] = [{"name": "tt"}, {"name": "ss"}]
        dataset = {
            "dc": (np.asarray([-1., 0., 1.]),
                    {"in": np.asarray([-1., 0., 1.]),
                     "out": np.asarray([-2., 0., 2.])}),
        }
        circuit.apply_params = Mock()
        circuit._copy_relative_includes = Mock()
        circuit.write_tb = Mock()
        circuit._run_spectre = Mock(return_value=dataset)
        with tempfile.TemporaryDirectory() as root:
            result = circuit.evaluate({}, root)
        self.assertTrue(result["ok"], result["error"])
        self.assertIn("measurement_signature", result)
        self.assertEqual(result["metrics_by_corner"]["tt"]["slope"], 2.)
        self.assertEqual(circuit._run_spectre.call_count, 2)


if __name__ == "__main__":
    unittest.main()
