"""Generic analysis and scalar measurement editor for the local optimizer."""
import copy

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from optserver.metric_catalog import metric_catalog
from optserver.metricexpr import validate_metric_definitions

from .i18n import tr


class MetricsPanel(QWidget):
    proposed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bindings = []
        self.catalog = metric_catalog()
        self._config = {}
        layout = QVBoxLayout(self)
        note = self._text(QLabel, "Define finite scalar measurements, then select their names in objectives and constraints.")
        note.setWordWrap(True)
        layout.addWidget(note)
        analyses = QHBoxLayout()
        self.tran_box = QGroupBox()
        self.tran_box.setCheckable(True)
        self.tran_box.setChecked(True)
        self.dc_box = QGroupBox()
        self.dc_box.setCheckable(True)
        self.dc_box.setChecked(False)
        tran_form = QFormLayout(self.tran_box)
        self.tran_stop = QLineEdit("10u")
        self.tran_step = QLineEdit("1n")
        tran_form.addRow(self._text(QLabel, "Stop time (s)"), self.tran_stop)
        tran_form.addRow(self._text(QLabel, "Maximum step (s)"), self.tran_step)
        dc_form = QFormLayout(self.dc_box)
        self.dc_fields = {}
        for name, label, default in (("source", "Source instance", "VIN"),
                                     ("start", "Sweep start", "-10m"),
                                     ("stop", "Sweep stop", "10m"),
                                     ("step", "Sweep step", "1m")):
            field = QLineEdit(default)
            self.dc_fields[name] = field
            dc_form.addRow(self._text(QLabel, label), field)
        analyses.addWidget(self.tran_box, 1)
        analyses.addWidget(self.dc_box, 1)
        layout.addLayout(analyses)
        signals = QFormLayout()
        self.save_signals = QLineEdit()
        signals.addRow(self._text(QLabel, "Saved signals (space separated)"), self.save_signals)
        layout.addLayout(signals)
        row = QHBoxLayout()
        self.templates = QComboBox()
        for item in self.catalog:
            self.templates.addItem(tr(item["title"]), item["name"])
        row.addWidget(self.templates, 1)
        self.btn_template = self._text(QPushButton, "Add example")
        self.btn_template.clicked.connect(self.add_template)
        row.addWidget(self.btn_template)
        layout.addLayout(row)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.table = QTableWidget(0, 3)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        for label, handler in (("Add custom metric", self.add_empty),
                               ("Remove selected metric", self.remove_selected),
                               ("Validate expressions", self.validate)):
            button = self._text(QPushButton, label)
            button.clicked.connect(handler)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.btn_apply = self._text(QPushButton, "Apply measurements and analyses")
        self.btn_apply.clicked.connect(self.apply)
        layout.addWidget(self.btn_apply)
        self.templates.currentIndexChanged.connect(self._describe)
        self.retranslate_ui()

    def _text(self, cls, source):
        item = cls(tr(source))
        self._bindings.append((item, source))
        return item

    def retranslate_ui(self):
        for widget, source in self._bindings:
            widget.setText(tr(source))
        self.tran_box.setTitle(tr("Transient analysis"))
        self.dc_box.setTitle(tr("DC source sweep (Spectre)"))
        self.table.setHorizontalHeaderLabels([tr("Metric name"), tr("Analysis"), tr("Expression")])
        for index, item in enumerate(self.catalog):
            self.templates.setItemText(index, tr(item["title"]))
        self._describe()

    def _describe(self, *_):
        item = self.catalog[self.templates.currentIndex()]
        self.description.setText("%s | %s — %s" % (item["analysis"], item["unit"], tr(item["description"])))

    def set_config(self, cfg):
        self._config = copy.deepcopy(cfg)
        self.tran_box.setChecked("tran" in cfg or "dc" not in cfg)
        self.dc_box.setChecked("dc" in cfg)
        self.dc_box.setEnabled(cfg.get("simulator", "spectre") == "spectre")
        self.save_signals.setText(" ".join(cfg.get("save", [])))
        tran = cfg.get("tran", {})
        self.tran_stop.setText(str(tran.get("stop", "10u")))
        self.tran_step.setText(str(tran.get("maxstep", "1n")))
        for name, field in self.dc_fields.items():
            default = {"source": "VIN", "start": "-10m", "stop": "10m", "step": "1m"}[name]
            field.setText(str(cfg.get("dc", {}).get(name, default)))
        self.table.setRowCount(0)
        default = "tran" if self.tran_box.isChecked() else "dc"
        for name, definition in cfg.get("metrics", {}).items():
            if isinstance(definition, str):
                self._row(name, default, definition)
            else:
                self._row(name, definition.get("analysis", default), definition.get("expr", ""))

    def _row(self, name, analysis, expr):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        combo = QComboBox()
        combo.addItems(["tran", "dc"])
        combo.setCurrentText(analysis)
        self.table.setCellWidget(row, 1, combo)
        item = QTableWidgetItem(expr)
        item.setToolTip(expr)
        self.table.setItem(row, 2, item)
        self.table.setCurrentCell(row, 2)
        self.table.scrollToItem(item)

    def add_template(self):
        item = self.catalog[self.templates.currentIndex()]
        used = {self.table.item(row, 0).text() for row in range(self.table.rowCount())}
        name, suffix = item["name"], 2
        while name in used:
            name, suffix = item["name"] + "_" + str(suffix), suffix + 1
        self._row(name, item["analysis"], item["expr"])

    def add_empty(self):
        self._row("", "tran" if self.tran_box.isChecked() else "dc", "")

    def remove_selected(self):
        if self.table.currentRow() >= 0:
            self.table.removeRow(self.table.currentRow())

    def settings(self):
        analyses = {}
        if self.tran_box.isChecked():
            analyses["tran"] = {**self._config.get("tran", {}),
                                "stop": self.tran_stop.text().strip(),
                                "maxstep": self.tran_step.text().strip()}
        if self.dc_box.isChecked():
            analyses["dc"] = {key: field.text().strip() for key, field in self.dc_fields.items()}
        definitions = {}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            expr = self.table.item(row, 2).text().strip()
            if not name or name in definitions:
                raise ValueError(tr("Metric names must be nonempty and unique."))
            definitions[name] = {"analysis": self.table.cellWidget(row, 1).currentText(), "expr": expr}
        validate_metric_definitions(definitions, analyses)
        return {"metrics": definitions, "analyses": analyses,
                "save": self.save_signals.text().split()}

    def validate(self):
        try:
            self.settings()
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid measurements"), str(exc))
            return
        QMessageBox.information(self, tr("Validation passed"),
                                tr("Expression syntax, references and analyses are valid. Signal availability and numerical results are checked during evaluation."))

    def apply(self):
        try:
            proposal = self.settings()
        except Exception as exc:
            QMessageBox.warning(self, tr("Invalid measurements"), str(exc))
            return
        self.proposed.emit(proposal)
