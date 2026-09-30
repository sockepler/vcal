"""Embedded gm/Id lookup and sizing panel; no simulator is started here."""
import math
from pathlib import Path

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from optserver.hspice_netlist import parse_num

from .gmid import GmIdTable
from .i18n import tr


class GmIdPanel(QWidget):
    proposed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = None
        self.result = None
        self._bindings = []
        self._loading = False
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        self.btn_load = self._text(QPushButton, "Open gm/Id LUT…")
        self.btn_load.clicked.connect(self.open_table)
        self.btn_demo = self._text(QPushButton, "Load synthetic demo")
        self.btn_demo.clicked.connect(self.load_demo)
        buttons.addWidget(self.btn_load)
        buttons.addWidget(self.btn_demo)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.lbl_source = QLabel()
        self.lbl_source.setWordWrap(True)
        layout.addWidget(self.lbl_source)
        self.lbl_hint = self._text(QLabel, "LUT units: m, V, A, S, F. PMOS biases use magnitudes. No extrapolation.")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        form = QFormLayout()
        self.length = QComboBox()
        self.vds = QDoubleSpinBox()
        self.vsb = QDoubleSpinBox()
        for field in (self.vds, self.vsb):
            field.setDecimals(4)
            field.setRange(0, 10)
            field.setSingleStep(0.05)
        self.gmid = QDoubleSpinBox()
        self.gmid.setRange(0.01, 1000)
        self.gmid.setDecimals(3)
        self.gmid.setValue(15)
        self.target_kind = QComboBox()
        self.target_kind.addItem("ID (A)", "ids")
        self.target_kind.addItem("gm (S)", "gm")
        self.target = QLineEdit("20u")
        target_row = QWidget()
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_kind)
        target_layout.addWidget(self.target)
        for source, field in (("Channel length L", self.length), ("|VDS| (V)", self.vds),
                              ("|VSB| (V)", self.vsb), ("Target gm/Id (1/V)", self.gmid),
                              ("Sizing target", target_row)):
            form.addRow(self._text(QLabel, source), field)
        layout.addLayout(form)
        self.btn_size = self._text(QPushButton, "Calculate device size")
        self.btn_size.clicked.connect(self.calculate)
        layout.addWidget(self.btn_size)

        self.figure = Figure(figsize=(5, 2.4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.setMinimumSize(250, 170)
        self.canvas.setMaximumHeight(220)
        self.values = QTableWidget(0, 3)
        self.values.setMinimumSize(260, 170)
        self.values.setMaximumHeight(220)
        self.values.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.values.horizontalHeader().setStretchLastSection(True)
        results_row = QHBoxLayout()
        results_row.addWidget(self.canvas, 3)
        results_row.addWidget(self.values, 2)
        layout.addLayout(results_row, 1)

        mapping = QFormLayout()
        self.width_parameter = QComboBox()
        self.length_parameter = QComboBox()
        self.multiplier = QSpinBox()
        self.multiplier.setRange(1, 100000)
        self.multiplier.setValue(1)
        mapping.addRow(self._text(QLabel, "Width parameter"), self.width_parameter)
        mapping.addRow(self._text(QLabel, "Length parameter (optional)"), self.length_parameter)
        mapping.addRow(self._text(QLabel, "Parallel width multiplier (nf × m)"), self.multiplier)
        layout.addLayout(mapping)
        hint = self._text(QLabel, "Targets are per device. For width per finger, set nf × m before applying.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.btn_apply = self._text(QPushButton, "Use as optimization initial values")
        self.btn_apply.clicked.connect(self.apply_result)
        layout.addWidget(self.btn_apply)

        self.length.currentIndexChanged.connect(self._bias_changed)
        self.vds.valueChanged.connect(self._bias_changed)
        self.vsb.valueChanged.connect(self._bias_changed)
        self.gmid.valueChanged.connect(self._invalidate)
        self.target.textChanged.connect(self._invalidate)
        self.target_kind.currentIndexChanged.connect(self._invalidate)
        self.set_parameters([])
        self.retranslate_ui()

    def _text(self, cls, source):
        widget = cls(tr(source))
        self._bindings.append((widget, source))
        return widget

    def retranslate_ui(self):
        for widget, source in self._bindings:
            widget.setText(tr(source))
        self.values.setHorizontalHeaderLabels([tr("Quantity"), tr("值"), tr("Unit")])
        self.length_parameter.setItemText(0, tr("Keep current length"))
        self.width_parameter.setItemText(0, tr("Select a width parameter"))
        if self.table is None:
            self.lbl_source.setText(tr("Load an existing NPZ/CSV lookup table, or try the synthetic demo."))
        else:
            temp = self.table.temperature()
            label = "%s | %sMOS | Wref=%.6g µm" % (
                Path(self.table.source()).name, self.table.polarity().upper(), self.table.width() * 1e6)
            if temp is not None:
                label += " | %.6g °C" % temp
            self.lbl_source.setText(label)
        self.btn_size.setEnabled(self.table is not None)
        self.btn_apply.setEnabled(self.result is not None and bool(self.width_parameter.currentData()))
        self._draw_curve()
        if self.result is not None:
            self._show_result()

    def set_parameters(self, params):
        width = self.width_parameter.currentData()
        length = self.length_parameter.currentData()
        self.width_parameter.clear()
        self.length_parameter.clear()
        self.width_parameter.addItem(tr("Select a width parameter"), None)
        self.length_parameter.addItem(tr("Keep current length"), None)
        for p in params:
            if p.get("attr", "").lower() in ("w", "wr"):
                self.width_parameter.addItem(p["name"], p["name"])
            elif p.get("attr", "").lower() in ("l", "lr"):
                self.length_parameter.addItem(p["name"], p["name"])
        if width is not None and self.width_parameter.findData(width) >= 0:
            self.width_parameter.setCurrentIndex(self.width_parameter.findData(width))
        if length is not None and self.length_parameter.findData(length) >= 0:
            self.length_parameter.setCurrentIndex(self.length_parameter.findData(length))
        try:
            self.width_parameter.currentIndexChanged.disconnect(self._mapping_changed)
        except (TypeError, RuntimeError):
            pass
        self.width_parameter.currentIndexChanged.connect(self._mapping_changed)
        self._mapping_changed()

    def _mapping_changed(self, *_):
        self.btn_apply.setEnabled(self.result is not None and bool(self.width_parameter.currentData()))

    def open_table(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open gm/Id LUT…"), "", "LUT (*.npz *.csv)")
        if path:
            self._load_or_report(path)

    def load_demo(self):
        self._load_or_report(Path(__file__).resolve().parents[2] / "examples" / "gmid_demo.csv")

    def _load_or_report(self, path):
        try:
            self.load_table(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("加载失败"), tr("LUT loading failed: %s") % exc)

    def load_table(self, path):
        table = GmIdTable.load(path)
        self._loading = True
        try:
            self.table = table
            self.result = None
            self.length.clear()
            for length in table.axes["L"]:
                self.length.addItem("%.6g µm" % (length * 1e6), float(length))
            for field, name in ((self.vds, "VDS"), (self.vsb, "VSB")):
                grid = table.axes[name]
                field.setRange(float(grid[0]), float(grid[-1]))
                field.setValue(float(grid[len(grid) // 2]) if name == "VDS" else float(grid[0]))
        finally:
            self._loading = False
        self.values.setRowCount(0)
        self.retranslate_ui()

    def _invalidate(self, *_):
        if self._loading:
            return
        self.result = None
        self.values.setRowCount(0)
        self.btn_apply.setEnabled(False)
        self._draw_curve()

    def _bias_changed(self, *_):
        self._invalidate()

    def _draw_curve(self):
        self.ax.clear()
        self.ax.set_xlabel("gm/Id (1/V)")
        self.ax.set_ylabel("Id/W (A/m)")
        self.ax.set_title(tr("gm/Id sizing curve"))
        if self.table is not None and self.length.currentData() is not None:
            try:
                curve = self.table.curve(self.length.currentData(), self.vds.value(), self.vsb.value())
                valid = (np.isfinite(curve["gmid"]) & np.isfinite(curve["idw"])
                         & (curve["gmid"] > 0) & (curve["idw"] > 0))
                self.ax.semilogy(np.where(valid, curve["gmid"], np.nan),
                                 np.where(valid, curve["idw"], np.nan), color="tab:blue")
                if self.result is not None:
                    self.ax.scatter([self.result["gmid"]], [self.result["idw"]], color="tab:red", zorder=3)
            except ValueError as exc:
                self.ax.text(0.5, 0.5, str(exc), transform=self.ax.transAxes,
                             ha="center", va="center", wrap=True, fontsize=8)
        self.ax.grid(alpha=0.3)
        self.figure.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.27)
        self.canvas.draw_idle()

    def calculate(self):
        try:
            if self.table is None:
                raise ValueError(tr("Load a LUT first."))
            target = parse_num(self.target.text().strip())
            if target is None or not math.isfinite(target) or target <= 0:
                raise ValueError(tr("The sizing target must be a positive finite SI value."))
            self.result = self.table.size_for(
                self.length.currentData(), self.vds.value(), self.vsb.value(),
                gmid=self.gmid.value(), **{self.target_kind.currentData(): target})
        except Exception as exc:
            self.result = None
            self.values.setRowCount(0)
            self.btn_apply.setEnabled(False)
            QMessageBox.warning(self, tr("Lookup failed"), str(exc))
            return
        self._show_result()
        self._mapping_changed()
        self._draw_curve()

    def _show_result(self):
        r = self.result
        rows = [(tr("Total effective width"), r["W"] * 1e6, "µm"),
                ("L", r["L"] * 1e6, "µm"), ("|VGS|", r["vgs"], "V"),
                ("ID", r["ids"] * 1e6, "µA"), ("gm", r["gm"] * 1e6, "µS"),
                ("gm/Id", r["gmid"], "1/V")]
        if r.get("gain") is not None:
            rows.append(("gm/gds", r["gain"], "V/V"))
        if r.get("ft") is not None:
            rows.append(("fT", r["ft"] * 1e-9, "GHz"))
        self.values.setRowCount(len(rows))
        from PyQt5.QtCore import Qt
        for row, (name, value, unit) in enumerate(rows):
            for column, text in enumerate((name, "%.6g" % value, unit)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.values.setItem(row, column, item)

    def apply_result(self):
        if self.result is None or not self.width_parameter.currentData():
            return
        values = {self.width_parameter.currentData(): self.result["W"] / self.multiplier.value()}
        if self.length_parameter.currentData():
            values[self.length_parameter.currentData()] = self.result["L"]
        self.proposed.emit({"params": values, "sizing": dict(self.result),
                            "width_multiplier": self.multiplier.value()})
