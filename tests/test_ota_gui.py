import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import yaml

from optlocal import gui, i18n
from optlocal.gmid import GmIdTable
from optserver.hspice_netlist import parse_num
from test_gui import FakeEvaluator


class OtaGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = FakeEvaluator(self.tmp.name)
        params = [
            {"name": "input_w", "devices": ["ota/M1", "ota/M2"], "attr": "w",
             "lo": 0.1e-6, "hi": 100e-6, "nominal": 5e-6},
            {"name": "input_l", "devices": ["ota/M1", "ota/M2"], "attr": "l",
             "lo": 0.1e-6, "hi": 2e-6, "nominal": 0.36e-6},
            {"name": "bias_w", "devices": ["bias/M1"], "attr": "w",
             "lo": 0.1e-6, "hi": 100e-6, "nominal": 10e-6},
            {"name": "shared_w", "devices": ["ota/M3", "bias/M2"], "attr": "w",
             "lo": 0.1e-6, "hi": 100e-6, "nominal": 8e-6},
        ]
        self.ev._spec.update(params=params, fixed={},
                             objective={"metric": "gain", "goal": "maximize"},
                             initial_points=[{"input_w": 7e-6}, {"input_w": 9e-6}],
                             optimizer={"stagnation_rounds": 4})
        self.ev.ckt.params = params
        self.ev.ckt.cfg.update(copy.deepcopy(self.ev._spec))
        # Circuit YAML stores expressions; the public spec stores names.
        self.ev.ckt.cfg["metrics"] = {name: "1.0" for name in self.ev._spec["metrics"]}
        self.win = gui.MainWindow(language="en")
        with mock.patch.object(QFileDialog, "getOpenFileName",
                               return_value=(str(Path(self.tmp.name) / "ota.yaml"), "")), \
                mock.patch.object(gui, "LocalEvaluator", return_value=self.ev):
            self.win.open_cfg()

    def tearDown(self):
        self.win.nw = None
        self.win.worker = None
        self.win.close()
        self.tmp.cleanup()

    def test_gmid_demo_maps_scaled_width_into_real_engine_seed(self):
        panel = self.win.gmid_panel
        before = copy.deepcopy(self.ev.ckt.cfg)
        demo = Path(__file__).parents[1] / "examples" / "gmid_demo.csv"
        with mock.patch.object(QMessageBox, "warning") as warning:
            panel.load_demo()
            panel.vds.setValue(0.6)
            panel.calculate()
            warning.assert_not_called()
        expected = GmIdTable.load(demo).size_for(0.18e-6, 0.6, gmid=15, ids=20e-6)
        self.assertAlmostEqual(panel.result["W"], expected["W"])
        panel.width_parameter.setCurrentIndex(panel.width_parameter.findData("input_w"))
        panel.length_parameter.setCurrentIndex(panel.length_parameter.findData("input_l"))
        panel.multiplier.setValue(2)
        self.assertTrue(panel.btn_apply.isEnabled())
        panel.apply_result()
        active, fixed = self.win._collect_params()
        snapshot = self.win._snapshot_run_settings(active, fixed,
                                                  self.win._collect_objective(),
                                                  self.win._collect_constraints())
        # The GUI table formatter retains six significant figures.
        self.assertAlmostEqual(snapshot["initial_points"][0]["input_w"], expected["W"] / 2, delta=expected["W"] * 1e-6)
        self.assertAlmostEqual(snapshot["initial_points"][0]["input_l"], 0.18e-6, delta=1e-12)
        self.assertEqual(snapshot["stagnation_rounds"], 4)
        self.assertEqual(snapshot["initial_points"][1], {"input_w": 9e-6})
        with mock.patch("optlocal.engine.Engine") as engine:
            self.win._engine_from_snapshot(snapshot, lambda _: None, lambda *_: None)
            self.assertEqual(engine.call_args.kwargs["initial_points"], snapshot["initial_points"])
            self.assertEqual(engine.call_args.kwargs["stagnation_rounds"], 4)
        self.assertEqual(self.ev.ckt.cfg, before)
        self.assertAlmostEqual(parse_num(self.win.tbl_params.item(0, 8).text()), 5e-6, delta=1e-12)

    def test_sizing_bounds_rejection_is_transactional(self):
        before = [self.win.tbl_params.item(i, 9).text() for i in range(4)]
        with self.assertRaises(ValueError):
            self.win._apply_initial_values({"input_w": 3e-6, "input_l": 20e-6})
        self.assertEqual([self.win.tbl_params.item(i, 9).text() for i in range(4)], before)

    def test_scope_freezes_background_and_shared_values_and_saves_seeds(self):
        self.win.tbl_params.item(2, 9).setText("12u")
        self.win.cmb_scope.setCurrentIndex(self.win.cmb_scope.findData("ota"))
        active, fixed = self.win._collect_params()
        self.assertEqual([p["name"] for p in active], ["input_w", "input_l"])
        self.assertEqual(fixed, {"bias_w": 12e-6, "shared_w": 8e-6})
        self.assertTrue(self.win.tbl_params.isRowHidden(2))
        self.assertFalse(self.win.tbl_params.item(3, 0).flags() & Qt.ItemIsEnabled)
        self.assertIn("all instances", self.win.lbl_scope.text())
        path = Path(self.tmp.name) / "copy.yaml"
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")), \
                mock.patch.object(QMessageBox, "critical") as critical:
            self.win.save_cfg()
            critical.assert_not_called()
        saved = yaml.safe_load(path.read_text())
        self.assertEqual(saved["fixed"], fixed)
        self.assertEqual(saved["optimizer"]["stagnation_rounds"], 4)
        for name, value in {"input_w": 7e-6, "input_l": 0.36e-6, **fixed}.items():
            self.assertAlmostEqual(saved["initial_points"][0][name], value, delta=1e-12)
        self.assertEqual(saved["initial_points"][1], {"input_w": 9e-6, **fixed})
        self.assertEqual([p["enabled"] for p in saved["params"]], [True, True, False, False])
        self.win.cmb_scope.setCurrentIndex(0)
        self.assertEqual(len(self.win._collect_params()[0]), 4)
        self.assertTrue(self.win.tbl_params.item(3, 0).flags() & Qt.ItemIsEnabled)

    def test_nominal_check_receives_edited_initial_values_without_widget_reads(self):
        self.win._apply_initial_values({"input_w": 11e-6})
        with mock.patch.object(gui.NominalWorker, "start"):
            self.win.run_nominal()
        params = dict(self.win.nw.params)
        self.assertEqual(params["input_w"], 11e-6)
        self.win.tbl_params.item(0, 9).setText("13u")
        self.assertEqual(self.win.nw.params, params)

    def test_best_reuse_review_and_language_switch_preserve_work(self):
        feasible = {"trial": 1, "ok": True, "params": {"input_w": 15e-6},
                    "metrics": {"gain": 10, "power": 2}}
        infeasible = {"trial": 2, "ok": True, "params": {"input_w": 20e-6},
                      "metrics": {"gain": 100, "power": 9}}
        self.win._records = [feasible, infeasible]
        self.win._objective_snapshot = [{"metric": "gain", "goal": "maximize"}]
        self.win._constraints_snapshot = [{"metric": "power", "max": 5}]
        self.win._last_summary = {"best_record": infeasible, "best_feasible": False}
        with mock.patch.object(QMessageBox, "information") as info:
            self.win._reuse_best()
            info.assert_not_called()
        self.assertAlmostEqual(parse_num(self.win.tbl_params.item(0, 9).text()), 15e-6, delta=1e-12)
        self.win._refresh_review()
        self.assertEqual(self.win._review["best_record"], feasible)
        self.assertEqual(self.win._review["n_feasible"], 1)
        panel = self.win.gmid_panel
        panel.load_demo()
        panel.calculate()
        self.win.cmb_scope.setCurrentIndex(self.win.cmb_scope.findData("ota"))
        before = json.dumps(panel.result, sort_keys=True)
        for language in ("ja", "zh", "en"):
            self.win.change_language(language)
            self.assertEqual(self.win.cmb_scope.currentData(), "ota")
            self.assertAlmostEqual(parse_num(self.win.tbl_params.item(0, 9).text()), 15e-6, delta=1e-12)
            self.assertEqual(json.dumps(panel.result, sort_keys=True), before)
            self.assertEqual(panel.btn_size.text(), i18n.tr("Calculate device size"))
            self.assertIn(i18n.tr("Suggested next steps:"), self.win.txt_review.toPlainText())
        panel.target.setText("30u")
        self.assertIsNone(panel.result)
        self.assertFalse(panel.btn_apply.isEnabled())


if __name__ == "__main__":
    unittest.main()
