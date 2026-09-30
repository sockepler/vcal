import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from optlocal.evaluator import LocalEvaluator
from optlocal.gui import MainWindow
from optlocal.i18n import get_language, set_language


class MetricsGuiRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previous_language = get_language()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dut.scs").write_text(
            "simulator lang=spectre\n"
            "R1 (out in) resistor r=1k\n")
        (self.root / "stimuli.scs").write_text(
            "simulator lang=spectre\n"
            "V1 (in 0) vsource dc=0\n")
        self.cfg_path = self.root / "circuit.yaml"
        self.cfg_path.write_text(yaml.safe_dump({
            "name": "metrics_gui_fixture",
            "simulator": "spectre",
            "netlist": "dut.scs",
            "stimuli": "stimuli.scs",
            "tran": {"stop": "1u", "maxstep": "1n"},
            "save": ["in", "out"],
            "params": [{"name": "res", "devices": ["R1"],
                        "attr": "r", "lo": "500", "hi": "3k"}],
            "metrics": {"gain": "avg(V('out'))"},
            "objective": {"metric": "gain", "goal": "maximize",
                          "weight": 2.5, "scale": 7.0},
            "constraints": [{"metric": "gain", "max": 1.5}],
        }))
        self.win = MainWindow(language="en")
        with patch.object(QFileDialog, "getOpenFileName",
                          return_value=(str(self.cfg_path), "YAML (*.yaml)")):
            self.win.open_cfg()
        self.assertIsInstance(self.win.evaluator, LocalEvaluator)

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        set_language(self.previous_language, persist=False)
        self.tmp.cleanup()

    def _metric_row(self, name):
        table = self.win.metrics_panel.table
        for row in range(table.rowCount()):
            if table.item(row, 0).text() == name:
                return row
        self.fail("metric row not found: %s" % name)

    def _add_dc_templates_and_enable_sweep(self):
        panel = self.win.metrics_panel
        index = panel.templates.findData("dc_gain_v_v")
        self.assertGreaterEqual(index, 0)
        panel.templates.setCurrentIndex(index)
        panel.add_template()
        panel.add_template()
        panel.dc_box.setChecked(True)
        names = [panel.table.item(row, 0).text()
                 for row in range(panel.table.rowCount())]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("dc_gain_v_v", names)
        self.assertIn("dc_gain_v_v_2", names)
        return names

    def test_template_names_are_unique_and_apply_tran_plus_dc(self):
        names = self._add_dc_templates_and_enable_sweep()
        self.win.metrics_panel.apply()

        self.assertEqual(set(self.win.cfg["metrics"]), set(names))
        self.assertIn("tran", self.win.cfg)
        self.assertIn("dc", self.win.cfg)
        self.assertEqual(self.win.cfg["dc"]["source"], "VIN")
        self.assertEqual(self.win.cfg["dc"]["step"], "1m")

        objective = self.win._collect_objective()
        self.assertEqual(objective[0]["metric"], "gain")
        self.assertEqual(objective[0]["weight"], 2.5)
        self.assertEqual(objective[0]["scale"], 7.0)
        self.assertEqual(self.win._collect_constraints(),
                         [{"metric": "gain", "max": 1.5}])
        for row in range(self.win.tbl_obj.rowCount()):
            metrics = self.win.tbl_obj.cellWidget(row, 0)
            self.assertTrue(all(metrics.findText(name) >= 0 for name in names))
        for row in range(self.win.tbl_cons.rowCount()):
            metrics = self.win.tbl_cons.cellWidget(row, 0)
            self.assertTrue(all(metrics.findText(name) >= 0 for name in names))

    def test_save_and_reload_preserve_metrics_and_dc_analysis(self):
        self._add_dc_templates_and_enable_sweep()
        self.win.metrics_panel.apply()
        saved = self.root / "saved.yaml"
        with patch.object(QFileDialog, "getSaveFileName",
                          return_value=(str(saved), "YAML (*.yaml)")):
            self.win.save_cfg()

        raw = yaml.safe_load(saved.read_text())
        self.assertIn("dc", raw)
        self.assertEqual(set(raw["metrics"]), set(self.win.cfg["metrics"]))
        objective = raw["objective"]
        objective = objective[0] if isinstance(objective, list) else objective
        self.assertEqual(objective["weight"], 2.5)
        self.assertEqual(objective["scale"], 7.0)
        self.assertEqual(raw["constraints"], [{"metric": "gain", "max": 1.5}])

        other = MainWindow(language="en")
        self.addCleanup(other.close)
        with patch.object(QFileDialog, "getOpenFileName",
                          return_value=(str(saved), "YAML (*.yaml)")):
            other.open_cfg()
        self.assertIn("dc", other.evaluator.ckt.cfg)
        self.assertEqual(set(other.evaluator.ckt.cfg["metrics"]),
                         set(raw["metrics"]))
        self.assertEqual(other.evaluator.ckt.cfg["dc"], raw["dc"])

    def test_delete_metric_referenced_by_objective_is_atomic(self):
        self._add_dc_templates_and_enable_sweep()
        before = copy.deepcopy(self.win.cfg)
        before_metrics = copy.deepcopy(self.win.evaluator.ckt.metrics)
        row = self._metric_row("gain")
        self.win.metrics_panel.table.setCurrentCell(row, 2)
        self.win.metrics_panel.remove_selected()

        with patch.object(QMessageBox, "warning") as warning:
            self.win.metrics_panel.apply()
        warning.assert_called_once()
        self.assertEqual(self.win.cfg, before)
        self.assertEqual(self.win.evaluator.ckt.metrics, before_metrics)

    def test_invalid_dc_step_is_atomic(self):
        self._add_dc_templates_and_enable_sweep()
        self.win.metrics_panel.apply()
        before = copy.deepcopy(self.win.cfg)
        before_metrics = copy.deepcopy(self.win.evaluator.ckt.metrics)
        self.win.metrics_panel.dc_fields["step"].setText("0")

        with patch.object(QMessageBox, "warning") as warning:
            self.win.metrics_panel.apply()
        warning.assert_called_once()
        self.assertEqual(self.win.cfg, before)
        self.assertEqual(self.win.evaluator.ckt.metrics, before_metrics)

    def test_save_signal_list_is_preserved_or_updated_and_changes_signature(self):
        panel = self.win.metrics_panel
        old_signature = self.win.evaluator.spec()["measurement_signature"]
        panel.save_signals.setText("")
        with patch.object(QMessageBox, "warning") as warning:
            panel.apply()
        # An explicitly empty save list is invalid; rejection must preserve
        # the active measurements and signature atomically.
        warning.assert_called_once()
        self.assertEqual(self.win.cfg["save"], ["in", "out"])
        self.assertEqual(self.win.evaluator.spec()["measurement_signature"],
                         old_signature)

        panel.save_signals.setText("in out bias")
        with patch.object(QMessageBox, "warning") as warning:
            panel.apply()
        self.assertFalse(warning.called, warning.call_args)
        self.assertEqual(self.win.cfg["save"], ["in", "out", "bias"])
        self.assertNotEqual(self.win.evaluator.spec()["measurement_signature"],
                            old_signature)

    def test_target_tolerance_constraint_is_retained_as_two_bounds(self):
        target_constraint = [{"metric": "gain", "target": 1.0, "tol": 0.2}]
        self.win.cfg["constraints"] = copy.deepcopy(target_constraint)
        self.win._fill_constraints(target_constraint)
        self.assertEqual(self.win._collect_constraints(), [
            {"metric": "gain", "min": 0.8},
            {"metric": "gain", "max": 1.2},
        ])

        self.win._apply_metric_settings(self.win.metrics_panel.settings())
        self.assertEqual(self.win.cfg["constraints"], [
            {"metric": "gain", "min": 0.8},
            {"metric": "gain", "max": 1.2},
        ])

    def test_active_task_cannot_apply_metric_settings(self):
        self._add_dc_templates_and_enable_sweep()
        proposal = self.win.metrics_panel.settings()
        before = copy.deepcopy(self.win.cfg)

        class ActiveWorker:
            def isRunning(self):
                return True

        self.win.worker = ActiveWorker()
        self.win._apply_metric_settings(proposal)
        self.assertEqual(self.win.cfg, before)
        self.win.worker = None

    def test_changed_metric_signature_excludes_old_history(self):
        evaluator = self.win.evaluator
        old_signature = evaluator.spec()["measurement_signature"]
        with open(evaluator.histfile, "w") as stream:
            json.dump({"trial": 0, "ok": True,
                       "params": {"res": 1000.0},
                       "metrics": {"gain": 1.0},
                       "measurement_signature": old_signature}, stream)
            stream.write("\n")
        self.assertEqual(len(evaluator.history()), 1)

        self._add_dc_templates_and_enable_sweep()
        self.win.metrics_panel.apply()
        new_signature = evaluator.spec()["measurement_signature"]
        self.assertNotEqual(new_signature, old_signature)
        self.assertEqual(len(evaluator.history()), 1)
        self.assertEqual(self.win._result_records(), [])

    def test_language_switch_preserves_metric_edits(self):
        panel = self.win.metrics_panel
        panel.dc_box.setChecked(True)
        panel.dc_fields["source"].setText("VBIAS")
        panel.dc_fields["step"].setText("2m")
        row = self._metric_row("gain")
        expression = "avg(V('out')) + 2"
        panel.table.item(row, 2).setText(expression)
        before = (panel.dc_fields["source"].text(),
                  panel.dc_fields["step"].text(),
                  panel.table.item(row, 0).text(),
                  panel.table.cellWidget(row, 1).currentText(),
                  panel.table.item(row, 2).text())

        for language in ("zh", "ja", "en"):
            self.win.change_language(language, persist=False)
            after = (panel.dc_fields["source"].text(),
                     panel.dc_fields["step"].text(),
                     panel.table.item(row, 0).text(),
                     panel.table.cellWidget(row, 1).currentText(),
                     panel.table.item(row, 2).text())
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
