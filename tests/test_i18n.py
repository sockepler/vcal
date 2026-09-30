import ast
import copy
import json
import os
from pathlib import Path
import re
import string
import tempfile
import unittest
from unittest import mock

from PyQt5.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

from optlocal import gui, i18n
from test_gui import FakeEvaluator


class LanguageEnvironment:
    def setUp(self):
        self.previous_language = i18n.get_language()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "settings.json"
        self.env = mock.patch.dict(os.environ, {"VCAL_SETTINGS_FILE": str(self.path)})
        self.env.start()
        os.environ.pop("VCAL_LANG", None)
        i18n.set_language("zh")

    def tearDown(self):
        i18n.set_language(self.previous_language)
        self.env.stop()
        self.tmp.cleanup()


class LanguageTests(LanguageEnvironment, unittest.TestCase):
    def test_preference_priority_aliases_and_preservation(self):
        self.assertEqual(i18n.configure_language(), "zh")
        self.path.write_text('{"unrelated": 42}', encoding="utf-8")
        i18n.set_language("ja_JP.UTF-8", persist=True)
        self.assertEqual(json.loads(self.path.read_text()),
                         {"language": "ja", "unrelated": 42})
        self.assertEqual(i18n.configure_language(), "ja")
        os.environ["VCAL_LANG"] = "en-US"
        self.assertEqual(i18n.configure_language(), "en")
        self.assertEqual(i18n.configure_language("zh_CN"), "zh")
        self.assertEqual(json.loads(self.path.read_text())["language"], "ja")
        with self.assertRaises(ValueError):
            i18n.configure_language("unsupported")
        with self.assertRaises(ValueError):
            i18n.configure_language("")

    def test_invalid_saved_settings_fall_back_and_failed_save_is_atomic(self):
        for raw in ('{broken', '[]', '{"language":"invalid"}'):
            self.path.write_text(raw, encoding="utf-8")
            self.assertEqual(i18n.configure_language(), "zh")
        i18n.set_language("ja", persist=True)
        before = self.path.read_bytes()
        with mock.patch.object(i18n.os, "replace", side_effect=OSError("read only")):
            with self.assertRaises(OSError):
                i18n.set_language("en", persist=True)
        self.assertEqual(i18n.get_language(), "ja")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()),
                         ["settings.json"])

    def test_catalogs_cover_display_text_and_preserve_placeholders(self):
        root = Path(i18n.__file__).parent
        catalogs = {lang: json.loads((root / "locales" / (lang + ".json"))
                                    .read_text(encoding="utf-8"))
                    for lang in i18n.LANGUAGES}
        keys = set(catalogs["zh"])
        # Percentages in descriptions (e.g. "10% to 90% of") are display
        # text, not printf substitutions. Only recognize valid conversions.
        percent = re.compile(r"(?<!\d)%(?:\([^)]+\))?[-+ #0]*\d*(?:\.\d+)?[diouxXeEfFgGcrsa%]")
        formatter = string.Formatter()
        for language, catalog in catalogs.items():
            self.assertEqual(set(catalog), keys, language)
            for source, translation in catalog.items():
                self.assertIsInstance(translation, str)
                self.assertTrue(translation, (language, source))
                self.assertEqual(percent.findall(source), percent.findall(translation),
                                 (language, source))
                # Only named format placeholders are interpolated; YAML
                # examples containing braces must remain literal display text.
                if "{language}" in source:
                    fields = lambda text: [p[1] for p in formatter.parse(text) if p[1]]
                    self.assertEqual(fields(source), fields(translation))
        for filename in ("gui.py", "gmid_gui.py", "metrics_gui.py", "__main__.py", "engine.py", "proposer.py"):
            tree = ast.parse((root / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                offset = {"tr": 0, "_tr": 0, "_format": 0, "_set_status": 0,
                          "_text_widget": 1, "_text_property": 2,
                          "_form_row": 1, "_tab": 2, "_text": 1}.get(name)
                if offset is None or len(node.args) <= offset:
                    continue
                arg = node.args[offset]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.assertIn(arg.value, keys, (filename, arg.value))
        for source in gui.GOAL_CN.values():
            self.assertIn(source, keys)
        from optserver.metric_catalog import metric_catalog
        for entry in metric_catalog():
            for field in ("title", "description"):
                self.assertIn(entry[field], keys)
        self.assertEqual(i18n.tr("未登记原始信息", "en"), "未登记原始信息")


class GuiLanguageTests(LanguageEnvironment, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.win = gui.MainWindow(language="zh")
        self.ev = FakeEvaluator(self.tmp.name)
        with mock.patch.object(gui.QFileDialog, "getOpenFileName",
                               return_value=(str(Path(self.tmp.name) / "circuit.yaml"), "")), \
                mock.patch.object(gui, "LocalEvaluator", return_value=self.ev):
            self.win.open_cfg()
        self.win._objective_snapshot = [{"metric": "gain", "goal": "maximize"}]
        self.win._constraints_snapshot = [{"metric": "power", "max": 5}]

    def tearDown(self):
        self.win.worker = None
        self.win.close()
        super().tearDown()

    def test_live_switch_preserves_edits_records_selection_and_zoom(self):
        self.win.tbl_params.item(0, 5).setText("7")
        self.win.tbl_obj.item(0, 4).setText("0.75")
        self.win.edit_remote.setText("http://example.invalid:8494")
        records = [
            {"trial": 1, "ok": True, "params": {"x": 2},
             "metrics": {"gain": 12.0, "power": 3.0}},
            {"trial": 2, "ok": True, "params": {"x": 4},
             "metrics": {"gain": 15.0, "power": 8.0}},
            {"trial": 3, "ok": False, "params": {"x": 6},
             "metrics": None, "error": "spectre diagnostic 123"},
        ]
        for index, record in enumerate(records):
            self.win.on_record((record, {"n": index + 1, "n_ok": min(index + 1, 2),
                                         "n_feasible": 1, "status": "running"}))
        self.win.cmb_metric.setCurrentText("power")
        self.win._show_record_details(records[2], "power", None)
        self.win.ax.set_xlim(0.5, 3.5)
        self.win.ax.set_ylim(2.0, 9.0)
        before = copy.deepcopy(self.win._records)
        before_config = copy.deepcopy(self.ev.ckt.cfg)
        for language in ("en", "ja", "zh"):
            with self.subTest(language=language):
                self.win.change_language(language)
                self.assertEqual(self.win._records, before)
                self.assertEqual(self.ev.ckt.cfg, before_config)
                self.assertEqual(self.win.tbl_params.item(0, 5).text(), "7")
                self.assertEqual(self.win._collect_objective()[0]["weight"], 0.75)
                self.assertEqual(self.win._collect_objective()[0]["goal"], "maximize")
                self.assertEqual(self.win._collect_constraints(), [{"metric": "power", "max": 5.0}])
                self.assertEqual(self.win.cmb_metric.currentText(), "power")
                self.assertEqual(self.win.ax.get_ylabel(), "power")
                self.assertEqual(self.win.ax.get_xlim(), (0.5, 3.5))
                self.assertEqual(self.win.ax.get_ylim(), (2.0, 9.0))
                self.assertEqual(self.win.edit_remote.text(), "http://example.invalid:8494")
                self.assertIn(i18n.tr("running"), self.win.lbl_status.text())
                self.assertIn(i18n.tr("仿真失败"), self.win.lbl_point.text())
                self.assertIn("spectre diagnostic 123", self.win.lbl_point.text())
                self.assertEqual(self.win.tbl_best.item(2, 0).text(), i18n.tr("指标 ") + "power")
                self.assertIn(i18n.tr("约束失败"), self.win.ax.get_legend_handles_labels()[1])
                self.assertEqual(self.win.tbl_params.horizontalHeaderItem(4).text(), i18n.tr("下限"))
                self.assertEqual(self.win.toolbar._actions["home"].toolTip(), i18n.tr("Reset original view"))
                self.assertEqual(sum(len(points) for points in self.win._pick_points.values()), 3)

    def test_combo_saves_preference_and_native_buttons_follow_language(self):
        for language, cancel in (("ja", "キャンセル"), ("en", "Cancel"), ("zh", "取消")):
            self.win.cmb_language.setCurrentIndex(self.win.cmb_language.findData(language))
            self.assertEqual(i18n.get_language(), language)
            self.assertEqual(json.loads(self.path.read_text())["language"], language)
            buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
            self.assertEqual(buttons.button(QDialogButtonBox.Cancel).text().replace("&", ""), cancel)
        self.win.change_language("en", persist=True)
        other = gui.MainWindow()
        try:
            self.assertEqual(other.cmb_language.currentData(), "en")
            self.assertIn("Open", other.btn_open.text())
        finally:
            other.close()

    def test_preference_save_failure_restores_selector(self):
        with mock.patch.object(i18n.os, "replace", side_effect=OSError("read only")), \
                mock.patch.object(QMessageBox, "warning") as warning:
            self.win.cmb_language.setCurrentIndex(self.win.cmb_language.findData("ja"))
        self.assertEqual(i18n.get_language(), "zh")
        self.assertEqual(self.win.cmb_language.currentData(), "zh")
        self.assertIn("打开", self.win.btn_open.text())
        warning.assert_called_once()

    def test_switch_during_run_preserves_busy_controls_and_progress(self):
        self.win._set_busy(True)
        self.win._set_status("状态：准备运行 0/%d", 80)
        self.assertTrue(self.win.cmb_language.isEnabled())
        self.win.cmb_language.setCurrentIndex(self.win.cmb_language.findData("en"))
        self.assertFalse(self.win.tabs.isEnabled())
        self.assertFalse(self.win.btn_start.isEnabled())
        self.assertTrue(self.win.btn_stop.isEnabled())
        self.assertIn("0/80", self.win.lbl_status.text())
        self.assertIn("Status", self.win.lbl_status.text())


if __name__ == "__main__":
    unittest.main()
