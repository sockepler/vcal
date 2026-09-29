import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QMessageBox

from optlocal.gui import MainWindow, RunWorker
from optlocal.report import write_records_csv


class FakeEvaluator:
    def __init__(self, root):
        self.root = root
        self.max_jobs = 4
        self.ckt = SimpleNamespace(
            cfg={"name": "fake", "params": [], "netlist": "/abs/dut.scs",
                 "stimuli": "/abs/stim.scs"}, params=[])
        self._spec = {
            "circuit": "fake",
            "params": [
                {"name": "x", "lo": 0.0, "hi": 10.0, "nominal": 0.0,
                 "enabled": True, "devices": ["M1"], "attr": "w"},
                {"name": "y", "lo": 0.0, "hi": 10.0, "nominal": 2.0,
                 "enabled": False, "devices": ["M2"], "attr": "w"},
            ],
            "metrics": ["gain", "power"],
            "objective": {"metric": "gain", "goal": "maximize",
                          "weight": 0.0, "scale": 2.0},
            "constraints": [{"metric": "power", "max": 5.0}],
            "fixed": {"y": 2.0},
        }
        self.ckt.cfg["params"] = self._spec["params"]
        self.ckt.cfg["fixed"] = {"y": 2.0}
        self.ckt.params = self.ckt.cfg["params"]

    def spec(self):
        return self._spec

    def history(self):
        return []


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.win = MainWindow(language="zh")
        self.ev = FakeEvaluator(self.tmp.name)
        self.win.evaluator = self.ev
        self.win.cfg = self.ev.ckt.cfg
        self.win.metric_names = ["gain", "power"]
        self.win.cmb_metric.addItems(self.win.metric_names)
        self.win._fixed_values = {"y": 2.0}
        self.win._fill_params(self.ev.spec()["params"])
        self.win._fill_objective(self.ev.spec()["objective"])
        self.win._fill_constraints(self.ev.spec()["constraints"])

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def test_zero_weight_scale_and_fixed_schema(self):
        self.win.tbl_params.item(1, 0).setCheckState(0)
        self.win.tbl_params.item(1, 5).setText("3")
        enabled, fixed = self.win._collect_params()
        self.assertEqual([p["name"] for p in enabled], ["x"])
        self.assertEqual(fixed, {"y": 2.0})
        self.assertEqual(self.win._collect_objective()[0]["weight"], 0.0)
        self.assertEqual(self.win._collect_objective()[0]["scale"], 2.0)

        target = os.path.join(self.tmp.name, "copy.yaml")
        with mock.patch.object(QFileDialog, "getSaveFileName",
                              return_value=(target, "YAML (*.yaml)")):
            self.win.save_cfg()
        import yaml
        with open(target) as stream:
            saved = yaml.safe_load(stream)
        self.assertEqual(len(saved["params"]), 2)
        self.assertFalse(saved["params"][1]["enabled"])
        self.assertEqual(saved["params"][1]["hi"], 3.0)
        self.assertEqual(saved["fixed"], {"y": 2.0})
        self.assertEqual(saved["objective"]["weight"], 0.0)
        self.assertEqual(saved["objective"]["scale"], 2.0)

    def test_metric_switch_points_status_and_best_trend(self):
        self.win._headline = "gain"
        self.win._objective_snapshot = [{"metric": "gain",
                                         "goal": "maximize",
                                         "weight": 1.0}]
        self.win._constraints_snapshot = [{"metric": "power", "max": 5.0}]
        self.win.on_record(({"trial": 1, "params": {"x": 1}, "ok": True,
                             "metrics": {"gain": 1.0, "power": 1.0}},
                            {"n": 1, "n_ok": 1, "n_feasible": 1,
                             "last_ok": True, "last_feasible": True,
                             "best_feasible": True,
                             "best_record": {"trial": 1, "metrics":
                                              {"gain": 1.0, "power": 1.0},
                                              "params": {"x": 1}}}))
        self.win.on_record(({"trial": 2, "params": {"x": 2}, "ok": True,
                             "metrics": {"gain": 2.0, "power": 8.0}},
                            {"n": 2, "n_ok": 2, "n_feasible": 1,
                             "last_ok": True, "last_feasible": False,
                             "best_feasible": True}))
        self.win.on_record(({"trial": 3, "params": {"x": 3}, "ok": False,
                             "metrics": None, "error": "spectre failed"},
                            {"n": 3, "n_ok": 2, "n_feasible": 1,
                             "last_ok": False, "last_feasible": False,
                             "best_feasible": True}))
        self.win.on_record(({"trial": 4, "params": {"x": 4}, "ok": True,
                             "metrics": {"gain": 1.5, "power": 1.5}},
                            {"n": 4, "n_ok": 3, "n_feasible": 2,
                             "last_ok": True, "last_feasible": True,
                             "best_feasible": True}))
        self.assertEqual(len(self.win._records), 4)
        self.assertEqual(self.win.ax.get_ylabel(), "gain")
        self.assertIn("最佳可行值", self.win.ax.get_legend_handles_labels()[1])
        self.assertTrue(any("仿真失败" in label
                            for label in self.win.ax.get_legend_handles_labels()[1]))
        artist = next(artist for artist, points in self.win._pick_points.items()
                      if len(points) >= 2)
        self.win._on_pick(SimpleNamespace(artist=artist,
                                          ind=np.array([0])))
        self.assertIn("trial=", self.win.lbl_point.text())
        self.win._on_pick(SimpleNamespace(artist=artist,
                                          ind=np.array([0, 1])))
        self.assertIn("trial=", self.win.lbl_point.text())
        self.win.cmb_metric.setCurrentText("power")
        self.assertEqual(self.win.ax.get_ylabel(), "power")
        # The failure marker is kept below the axes via a transform and has no
        # metric y value in the regular scatter collections.
        self.assertTrue(any("仿真失败" in label
                            for label in self.win.ax.get_legend_handles_labels()[1]))

    def test_export_plot_writes_png(self):
        path = os.path.join(self.tmp.name, "plot.png")
        with mock.patch.object(QFileDialog, "getSaveFileName",
                              return_value=(path, "PNG (*.png)")):
            self.win.export_plot()
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as stream:
            self.assertEqual(stream.read(8), b"\x89PNG\r\n\x1a\n")

    def test_close_active_task_requests_stop_and_ignores_event(self):
        class FakeWorker:
            def __init__(self):
                self.running = True
                self.stopped = False

            def isRunning(self):
                return self.running

            def request_stop(self):
                self.stopped = True

        class FakeEvent:
            def __init__(self):
                self.ignored = False
                self.accepted = False

            def ignore(self):
                self.ignored = True

            def accept(self):
                self.accepted = True

        worker = FakeWorker()
        self.win.worker = worker
        self.win._set_busy(True)
        event = FakeEvent()
        with mock.patch.object(QMessageBox, "question",
                              return_value=QMessageBox.Yes):
            self.win.closeEvent(event)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)
        self.assertTrue(worker.stopped)
        self.assertFalse(self.win.btn_open.isEnabled())
        self.assertFalse(self.win.btn_save.isEnabled())
        self.assertFalse(self.win.btn_nominal.isEnabled())
        self.assertFalse(self.win.btn_wb.isEnabled())
        worker.running = False
        self.win.worker = None

    def test_records_csv_contains_structured_columns(self):
        path = os.path.join(self.tmp.name, "records.csv")
        write_records_csv([{"trial": 7, "ok": False, "error": "bad",
                            "params": {"x": 1}, "metrics": None}], path)
        with open(path, encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["trial"], "7")
        self.assertEqual(row["ok"], "False")
        self.assertEqual(row["error"], "bad")
        self.assertIn('"x": 1', row["params"])
        self.assertEqual(row["param.x"], "1")

    def test_stop_before_engine_creation_is_retained(self):
        class FakeEngine:
            def __init__(self):
                import threading
                self.stop_event = threading.Event()
                self.records = []

            def run(self, _budget):
                self.ran_stopped = self.stop_event.is_set()

            def summary(self):
                return {"n": 0, "n_feasible": 0}

        made = []
        engine = FakeEngine()

        def make_engine(**_callbacks):
            made.append(True)
            return engine

        worker = RunWorker(make_engine, 4, False)
        worker.request_stop()
        worker.run()
        self.assertEqual(made, [True])
        self.assertTrue(engine.ran_stopped)


if __name__ == "__main__":
    unittest.main()
