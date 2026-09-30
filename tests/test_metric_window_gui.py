import unittest

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from optlocal.i18n import get_language, set_language, tr
from optlocal.metrics_gui import MetricsPanel


class MetricWindowGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previous_language = get_language()
        set_language("en")
        self.panel = MetricsPanel()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        set_language(self.previous_language)

    def _row(self, name):
        for row in range(self.panel.table.rowCount()):
            if self.panel.table.item(row, 0).text() == name:
                return row
        self.fail("metric row not found: %s" % name)

    def test_window_columns_round_trip_and_partial_bounds(self):
        self.panel.set_config({
            "simulator": "spectre",
            "tran": {"stop": "20u", "maxstep": "1n"},
            "save": ["out"],
            "metrics": {
                "settle": {
                    "analysis": "tran",
                    "expr": "avg(V('out'))",
                    "window": {"start": "5u", "end": "10u"},
                },
            },
        })
        self.assertEqual(self.panel.table.columnCount(), 5)
        self.assertEqual(self.panel.table.horizontalHeaderItem(2).text(),
                         "Expression")
        row = self._row("settle")
        self.assertEqual(self.panel.table.item(row, 3).text(), "5u")
        self.assertEqual(self.panel.table.item(row, 4).text(), "10u")
        self.assertEqual(self.panel.settings()["metrics"]["settle"]["window"],
                         {"start": "5u", "end": "10u"})

        self.panel.table.item(row, 4).setText("")
        self.assertEqual(self.panel.settings()["metrics"]["settle"]["window"],
                         {"start": "5u"})
        self.panel.table.item(row, 3).setText("")
        self.assertNotIn("window", self.panel.settings()["metrics"]["settle"])

    def test_dc_rows_clear_and_disable_time_cells(self):
        self.panel.dc_box.setChecked(True)
        self.panel._row("dc_metric", "tran", "avg(V('out'))",
                         {"start": "5u", "end": "10u"})
        row = self._row("dc_metric")
        combo = self.panel.table.cellWidget(row, 1)
        combo.setCurrentText("dc")
        self.assertEqual(self.panel.table.item(row, 3).text(), "")
        self.assertEqual(self.panel.table.item(row, 4).text(), "")
        for column in (3, 4):
            flags = self.panel.table.item(row, column).flags()
            self.assertFalse(flags & Qt.ItemIsEditable)
            self.assertFalse(flags & Qt.ItemIsEnabled)
        definition = self.panel.settings()["metrics"]["dc_metric"]
        self.assertEqual(definition["analysis"], "dc")
        self.assertNotIn("window", definition)

        combo.setCurrentText("tran")
        for column in (3, 4):
            flags = self.panel.table.item(row, column).flags()
            self.assertTrue(flags & Qt.ItemIsEditable)
            self.assertTrue(flags & Qt.ItemIsEnabled)

    def test_catalog_window_is_added(self):
        index = self.panel.templates.findData("settling_s")
        self.assertGreaterEqual(index, 0)
        self.panel.templates.setCurrentIndex(index)
        self.panel.add_template()
        row = self._row("settling_s")
        expected = self.panel.catalog[index]["window"]
        self.assertEqual(
            {"start": self.panel.table.item(row, 3).text(),
             "end": self.panel.table.item(row, 4).text()}, expected)

    def test_language_retranslation_preserves_window_edits(self):
        self.panel._row("edited", "tran", "avg(V('out'))",
                         {"start": "7u", "end": "11u"})
        row = self._row("edited")
        before = (self.panel.table.item(row, 0).text(),
                  self.panel.table.cellWidget(row, 1).currentText(),
                  self.panel.table.item(row, 2).text(),
                  self.panel.table.item(row, 3).text(),
                  self.panel.table.item(row, 4).text())
        for language in ("zh", "ja", "en"):
            set_language(language)
            self.panel.retranslate_ui()
            after = (self.panel.table.item(row, 0).text(),
                     self.panel.table.cellWidget(row, 1).currentText(),
                     self.panel.table.item(row, 2).text(),
                     self.panel.table.item(row, 3).text(),
                     self.panel.table.item(row, 4).text())
            self.assertEqual(after, before)
            self.assertEqual(self.panel.table.horizontalHeaderItem(3).text(),
                             tr("Start time (s)", language))
            self.assertEqual(self.panel.table.horizontalHeaderItem(4).text(),
                             tr("End time (s)", language))
            self.assertEqual(self.panel.table.horizontalHeaderItem(3).toolTip(),
                             tr("Window bounds are absolute simulation time; blank uses the available range; DC uses the sweep range.", language))


if __name__ == "__main__":
    unittest.main()
