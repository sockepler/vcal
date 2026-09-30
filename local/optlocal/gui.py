# -*- coding: utf-8 -*-
"""circuit-opt 本地优化 GUI（PyQt5，全离线：本地仿真 + 本地 BoTorch）。

布局仿 gmid-tool：左侧参数/目标/设置，右侧收敛曲线 + 最优点 + 日志。
"""
import json
import os
import copy
import math
import threading

import yaml
from PyQt5.QtCore import QLibraryInfo, QThread, Qt, QTranslator, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib
from matplotlib.figure import Figure

# Use installed CJK fonts for chart labels, including exported PNG files.
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from optserver.hspice_netlist import fmt_num, parse_num

from .evaluator import LocalEvaluator
from .i18n import (LANGUAGES, configure_language, get_language,
                   set_language, tr)
from .objective import Objective
from .scopes import parameter_scopes, scope_inventory, select_scope

GOALS = ["maximize", "minimize", "target"]
GOAL_CN = {"maximize": "最大化", "minimize": "最小化", "target": "逼近目标值"}


def _num_item(v, editable=True):
    it = QTableWidgetItem("" if v is None else fmt_num(float(v)))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


def _ro_item(text):
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    return it


class RunWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_record = pyqtSignal(object)
    sig_done = pyqtSignal(object)

    def __init__(self, make_engine, budget, resume):
        super().__init__()
        self.make_engine = make_engine
        self.budget = budget
        self.resume = resume
        self.engine = None
        # The GUI can receive a stop click before this thread has constructed
        # Engine.  Keep the request outside Engine so that it cannot be lost.
        self._stop_requested = threading.Event()

    def run(self):
        try:
            self.engine = self.make_engine(
                on_log=lambda s: self.sig_log.emit(s),
                on_record=lambda rec, st: self.sig_record.emit((rec, st)))
            if self._stop_requested.is_set():
                self.engine.stop_event.set()
            if self.resume and not self._stop_requested.is_set():
                self.engine.resume()
            self.engine.run(self.budget)
            self.sig_done.emit(self.engine.summary())
        except Exception as e:
            import traceback
            self.sig_log.emit("%s: %s\n%s" % (tr("error"), e,
                                               traceback.format_exc()))
            self.sig_done.emit(None)

    def request_stop(self):
        self._stop_requested.set()
        if self.engine is not None:
            self.engine.stop_event.set()


class NominalWorker(QThread):
    sig_log = pyqtSignal(str)
    sig_done = pyqtSignal(object)

    def __init__(self, evaluator, params):
        super().__init__()
        self.ev = evaluator
        self.params = dict(params)
        self._stop_requested = threading.Event()

    def run(self):
        self.sig_log.emit(tr("nominal 试跑中…"))
        if self._stop_requested.is_set():
            self.sig_log.emit(tr("nominal 已停止（尚未开始仿真）"))
            self.sig_done.emit(None)
            return
        try:
            r = self.ev.evaluate(self.params)
        except Exception as e:
            r = {"ok": False, "error": str(e), "metrics": None}
        if r.get("ok"):
            self.sig_log.emit(tr("nominal 指标: ") + json.dumps(
                {k: round(v, 6) for k, v in r["metrics"].items()},
                ensure_ascii=False))
        else:
            self.sig_log.emit(tr("nominal 失败: %s") % r.get("error"))
        self.sig_done.emit(r)

    def request_stop(self):
        # LocalEvaluator cannot cancel a running Spectre process safely.  The
        # flag still makes a pre-start request deterministic and lets close()
        # wait for this QThread instead of destroying it.
        self._stop_requested.set()


class MainWindow(QMainWindow):
    def __init__(self, language=None):
        super().__init__()
        configure_language(language)
        self._text_bindings = []
        self._header_bindings = []
        self._tab_bindings = []
        self._status_display = ("状态：空闲", ())
        self._status_summary = None
        self._selected_point = None
        self._detail_view = None
        self.setWindowTitle(tr("circuit-opt 本地调参 (BO/DKL, 离线)"))
        self.resize(1400, 900)
        self.cfg = None
        self.cfg_path = None
        self.evaluator = None
        self.worker = None
        self.nw = None
        self._closing_after_stop = False
        self._records = []
        self._record_states = []
        self._pick_points = {}
        self._fixed_values = {}
        self._seed_extras = []
        self._review = None
        self.metric_names = []
        self.plot_data = []          # (idx, headline, feasible, ok)
        self._build_ui()
        self._install_qt_language()
        self.retranslate_ui()

    def _text_widget(self, cls, source, word_wrap=False):
        widget = cls(tr(source))
        if word_wrap:
            widget.setWordWrap(True)
        self._text_bindings.append((widget, "setText", source))
        return widget

    def _text_property(self, widget, setter, source):
        self._text_bindings.append((widget, setter, source))
        getattr(widget, setter)(tr(source))

    def _headers(self, table, sources):
        self._header_bindings.append((table, sources))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setHorizontalHeaderLabels([tr(source) for source in sources])

    def _tab(self, tabs, widget, source):
        index = tabs.addTab(widget, tr(source))
        self._tab_bindings.append((tabs, index, source))

    def _form_row(self, form, source, widget):
        form.addRow(self._text_widget(QLabel, source), widget)

    def _set_status(self, source, *values):
        self._status_display = (source, values)
        self._status_summary = None
        self.lbl_status.setText(tr(source) % values if values else tr(source))

    def _install_qt_language(self):
        """Translate Qt's own file dialogs and standard message-box buttons."""
        app = QApplication.instance()
        previous = getattr(app, "_vcal_translator", None)
        if previous is not None:
            app.removeTranslator(previous)
        app._vcal_translator = None
        locale = {"zh": "zh_CN", "ja": "ja"}.get(get_language())
        if locale:
            translator = QTranslator(app)
            directory = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
            if translator.load("qtbase_" + locale, directory):
                app.installTranslator(translator)
                app._vcal_translator = translator
            else:
                translator.deleteLater()
        if previous is not None:
            previous.deleteLater()

    def change_language(self, language, persist=False):
        """Update display text without rebuilding widgets or discarding edits."""
        set_language(language, persist=persist)
        self._install_qt_language()
        self.retranslate_ui()

    def _on_language_changed(self, index):
        code = self.cmb_language.itemData(index)
        if not code or code == get_language():
            return
        previous = get_language()
        try:
            self.change_language(code, persist=True)
        except OSError as exc:
            self.cmb_language.blockSignals(True)
            self.cmb_language.setCurrentIndex(self.cmb_language.findData(previous))
            self.cmb_language.blockSignals(False)
            QMessageBox.warning(self, tr("保存失败"),
                                tr("Language preference could not be saved: %s") % exc)
            return
        self.log(tr("Language switched to %s") % LANGUAGES[code])

    def retranslate_ui(self):
        dynamic = (self.lbl_cfg, self.lbl_status, self.lbl_point, self.lbl_scope)
        for widget, setter, source in self._text_bindings:
            if widget in dynamic and setter == "setText":
                continue
            getattr(widget, setter)(tr(source))
        for table, sources in self._header_bindings:
            table.setHorizontalHeaderLabels([tr(source) for source in sources])
            table.resizeColumnsToContents()
        for tabs, index, source in self._tab_bindings:
            tabs.setTabText(index, tr(source))
        self.setWindowTitle(tr("circuit-opt 本地调参 (BO/DKL, 离线)"))
        self.cmb_language.blockSignals(True)
        self.cmb_language.setCurrentIndex(self.cmb_language.findData(get_language()))
        self.cmb_language.blockSignals(False)
        self.cmb_scope.setItemText(0, tr("All parameter scopes"))
        for row in range(self.tbl_obj.rowCount()):
            combo = self.tbl_obj.cellWidget(row, 1)
            for index, goal in enumerate(GOALS):
                combo.setItemText(index, tr(GOAL_CN[goal]))
        for row in range(self.tbl_cons.rowCount()):
            combo = self.tbl_cons.cellWidget(row, 1)
            combo.setItemText(0, tr("≤ (max)"))
            combo.setItemText(1, tr("≥ (min)"))
        if self.evaluator is not None:
            spec = self.evaluator.spec()
            self.lbl_cfg.setText(tr("<b>%s</b>  (%s, %d 参数)") %
                                 (spec["circuit"], (self.cfg or {}).get("simulator", "spectre"),
                                  len(spec["params"])))
        else:
            self.lbl_cfg.setText(tr("<i>未加载</i>"))
        if self._status_summary is not None:
            self._update_status(*self._status_summary)
        else:
            source, values = self._status_display
            self._set_status(source, *values)
        if self._selected_point is not None:
            self._render_point_label(*self._selected_point)
        else:
            self.lbl_point.setText(tr("点击图中点查看 trial、指标和参数"))
        if self._detail_view:
            kind, args = self._detail_view
            {"point": self._show_record_details, "best": self._show_best,
             "empty": self._show_no_best}[kind](*args)
        for title, tooltip, _icon, callback in self.toolbar.toolitems:
            action = self.toolbar._actions.get(callback)
            if action is not None:
                action.setText(tr(title))
                action.setToolTip(tr(tooltip) if tooltip else "")
        matplotlib.rcParams["font.sans-serif"] = (
            ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
            if get_language() == "ja" else
            ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"])
        self.gmid_panel.retranslate_ui()
        self.metrics_panel.retranslate_ui()
        self._update_scope_view()
        self._render_review()
        # Keep a user's zoom/pan while refreshing labels on an existing plot.
        limits = (self.ax.get_xlim(), self.ax.get_ylim()) if self._records else None
        self._redraw()
        if limits is not None:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
            self.canvas.draw_idle()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = QWidget()
        top_layout = QVBoxLayout(top)
        tl = QHBoxLayout()
        actions = QHBoxLayout()
        controls = QHBoxLayout()
        self.btn_open = self._text_widget(QPushButton, "打开电路配置…")
        self.btn_open.clicked.connect(self.open_cfg)
        self.lbl_cfg = self._text_widget(QLabel, "<i>未加载</i>")
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cuda", "cpu"])
        self.spin_jobs = QSpinBox()
        self.spin_jobs.setRange(1, 16)
        self.spin_jobs.setValue(4)
        self.btn_nominal = self._text_widget(QPushButton, "Nominal 试跑")
        self.btn_nominal.clicked.connect(self.run_nominal)
        self.btn_start = self._text_widget(QPushButton, "▶ 开始优化")
        self.btn_start.clicked.connect(self.start_run)
        self.btn_stop = self._text_widget(QPushButton, "■ 停止")
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_stop.setEnabled(False)
        self.btn_wb = self._text_widget(QPushButton, "回写原理图…")
        self.btn_wb.clicked.connect(self.writeback)
        self.btn_save = self._text_widget(QPushButton, "保存配置副本…")
        self.btn_save.clicked.connect(self.save_cfg)
        self.btn_export_csv = self._text_widget(QPushButton, "导出结果 CSV…")
        self.btn_export_csv.clicked.connect(self.export_results_csv)
        self.btn_open_results = self._text_widget(QPushButton, "打开结果目录")
        self.btn_open_results.clicked.connect(self.open_results_dir)
        self.lbl_status = self._text_widget(QLabel, "状态：空闲")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_status.setMinimumWidth(270)
        self._text_property(self.lbl_status, 'setToolTip', "显示当前任务状态、评估进度、成功数和可行数")
        for w in (self.btn_open, self.lbl_cfg):
            tl.addWidget(w)
        tl.addStretch(1)
        controls.addWidget(self._text_widget(QLabel, "计算设备:"))
        controls.addWidget(self.cmb_device)
        controls.addWidget(self._text_widget(QLabel, "并行仿真:"))
        controls.addWidget(self.spin_jobs)
        controls.addWidget(self.lbl_status, 1)
        for w in (self.btn_nominal, self.btn_start, self.btn_stop,
                  self.btn_wb, self.btn_save, self.btn_export_csv,
                  self.btn_open_results):
            actions.addWidget(w)
        actions.addStretch(1)
        tl.addWidget(self._text_widget(QLabel, "Language:"))
        self.cmb_language = QComboBox()
        for code, name in LANGUAGES.items():
            self.cmb_language.addItem(name, code)
        self.cmb_language.setCurrentIndex(self.cmb_language.findData(get_language()))
        self.cmb_language.currentIndexChanged.connect(self._on_language_changed)
        tl.addWidget(self.cmb_language)
        top_layout.addLayout(tl)
        top_layout.addLayout(actions)
        top_layout.addLayout(controls)

        # ---- left tabs ----
        tabs = QTabWidget()
        # params
        parameters_widget = QWidget()
        parameters_layout = QVBoxLayout(parameters_widget)
        scope_row = QHBoxLayout()
        scope_row.addWidget(self._text_widget(QLabel, "Optimization scope:"))
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItem(tr("All parameter scopes"), None)
        self.cmb_scope.currentIndexChanged.connect(self._update_scope_view)
        scope_row.addWidget(self.cmb_scope, 1)
        self.btn_reuse_best = self._text_widget(QPushButton, "Use best as initial values")
        self.btn_reuse_best.clicked.connect(self._reuse_best)
        scope_row.addWidget(self.btn_reuse_best)
        parameters_layout.addLayout(scope_row)
        self.lbl_scope = QLabel()
        self.lbl_scope.setWordWrap(True)
        parameters_layout.addWidget(self.lbl_scope)
        self.tbl_params = QTableWidget(0, 10)
        self._headers(self.tbl_params, ["启用", "参数", "器件", "属性", "下限", "上限", "log",
             "整数", "nominal", "Initial value"])
        self.tbl_params.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        parameters_layout.addWidget(self.tbl_params)
        self._tab(tabs, parameters_widget, "参数空间")
        # objective + constraints
        w2 = QWidget()
        v2 = QVBoxLayout(w2)
        v2.addWidget(self._text_widget(QLabel, "优化目标（多项加权求和；「逼近目标值」= 最小化 |指标−目标|）",
                                     word_wrap=True))
        self.tbl_obj = QTableWidget(0, 6)
        self._headers(self.tbl_obj, ["指标", "方向", "目标值", "容差(可选)", "权重", "尺度"])
        v2.addWidget(self.tbl_obj)
        h2 = QHBoxLayout()
        b_add = self._text_widget(QPushButton, "+ 目标项")
        b_add.clicked.connect(lambda: self.add_obj_row())
        b_del = self._text_widget(QPushButton, "− 删除选中")
        b_del.clicked.connect(
            lambda: self.tbl_obj.removeRow(self.tbl_obj.currentRow()))
        h2.addWidget(b_add)
        h2.addWidget(b_del)
        h2.addStretch(1)
        v2.addLayout(h2)
        v2.addWidget(self._text_widget(QLabel, "硬约束（不满足视为不可行）"))
        self.tbl_cons = QTableWidget(0, 3)
        self._headers(self.tbl_cons, ["指标", "类型", "边界值"])
        v2.addWidget(self.tbl_cons)
        h3 = QHBoxLayout()
        b_add2 = self._text_widget(QPushButton, "+ 约束")
        b_add2.clicked.connect(lambda: self.add_cons_row())
        b_del2 = self._text_widget(QPushButton, "− 删除选中")
        b_del2.clicked.connect(
            lambda: self.tbl_cons.removeRow(self.tbl_cons.currentRow()))
        h3.addWidget(b_add2)
        h3.addWidget(b_del2)
        h3.addStretch(1)
        v2.addLayout(h3)
        self._tab(tabs, w2, "目标与约束")
        # run settings
        w3 = QWidget()
        f3 = QFormLayout(w3)
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(4, 100000)
        self.spin_budget.setValue(200)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 16)
        self.spin_batch.setValue(4)
        self.spin_init = QSpinBox()
        self.spin_init.setRange(2, 4096)
        self.spin_init.setValue(24)
        self.chk_dkl = self._text_widget(QCheckBox, "样本足够后切换深度核 GP (DKL)")
        self.chk_dkl.setChecked(True)
        self.spin_dkl_after = QSpinBox()
        self.spin_dkl_after.setRange(8, 4096)
        self.spin_dkl_after.setValue(48)
        self.chk_resume = self._text_widget(QCheckBox, "从历史续跑 (history.jsonl)")
        self.chk_resume.setChecked(True)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_stagnation = QSpinBox()
        self.spin_stagnation.setRange(0, 10000)
        self.spin_stagnation.setValue(6)
        self._text_property(self.spin_stagnation, 'setToolTip',
                            "After this many batches without progress, explore a new Sobol batch. The total budget stays unchanged.")
        rw = QWidget()
        rh = QHBoxLayout(rw)
        rh.setContentsMargins(0, 0, 0, 0)
        self.edit_remote = QLineEdit()
        self._text_property(self.edit_remote, 'setPlaceholderText', "如 http://your-gpu-host:8494（空=本机计算）")
        btn_rtest = self._text_widget(QPushButton, "测试")
        btn_rtest.clicked.connect(self.test_remote)
        rh.addWidget(self.edit_remote, 1)
        rh.addWidget(btn_rtest)
        self._form_row(f3, "总评估次数:", self.spin_budget)
        self._form_row(f3, "每轮提案数:", self.spin_batch)
        self._form_row(f3, "初始采样数:", self.spin_init)
        f3.addRow(self.chk_dkl)
        self._form_row(f3, "DKL 启用样本数:", self.spin_dkl_after)
        f3.addRow(self.chk_resume)
        self._form_row(f3, "随机种子:", self.spin_seed)
        self._form_row(f3, "Stagnant batches (0 = off):", self.spin_stagnation)
        self._form_row(f3, "远程GPU计算服务:", rw)
        self._tab(tabs, w3, "运行设置")
        from .gmid_gui import GmIdPanel
        self.gmid_panel = GmIdPanel()
        self.gmid_panel.proposed.connect(self._apply_gmid_result)
        gmid_scroll = QScrollArea()
        gmid_scroll.setWidgetResizable(True)
        gmid_scroll.setWidget(self.gmid_panel)
        self._tab(tabs, gmid_scroll, "gm/Id assistant")
        review_widget = QWidget()
        review_layout = QVBoxLayout(review_widget)
        self.btn_review = self._text_widget(QPushButton, "Review current run")
        self.btn_review.clicked.connect(self._refresh_review)
        review_layout.addWidget(self.btn_review)
        self.txt_review = QPlainTextEdit()
        self.txt_review.setReadOnly(True)
        review_layout.addWidget(self.txt_review)
        self._tab(tabs, review_widget, "Iteration review")
        from .metrics_gui import MetricsPanel
        self.metrics_panel = MetricsPanel()
        self.metrics_panel.proposed.connect(self._apply_metric_settings)
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setWidget(self.metrics_panel)
        self._tab(tabs, metrics_scroll, "Circuit metrics")
        self.tabs = tabs

        # ---- right side ----
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.metric_bar = QWidget()
        mb = QHBoxLayout(self.metric_bar)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.addWidget(self._text_widget(QLabel, "图表指标:"))
        self.cmb_metric = QComboBox()
        self.cmb_metric.currentIndexChanged.connect(self._on_metric_changed)
        mb.addWidget(self.cmb_metric, 1)
        self.btn_export_plot = self._text_widget(QPushButton, "导出 PNG…")
        self.btn_export_plot.clicked.connect(self.export_plot)
        mb.addWidget(self.btn_export_plot)
        self.lbl_point = self._text_widget(QLabel, "点击图中点查看 trial、指标和参数")
        self.lbl_point.setWordWrap(True)
        self.lbl_point.setMinimumHeight(34)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.metric_bar)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.lbl_point)
        self.tbl_best = QTableWidget(0, 2)
        self._headers(self.tbl_best, ["项", "值"])
        self.tbl_best.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(5000)
        right = QSplitter(Qt.Vertical)
        right.addWidget(plot_widget)
        right.addWidget(self.tbl_best)
        right.addWidget(self.txt_log)
        right.setSizes([420, 260, 180])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(tabs)
        split.addWidget(right)
        split.setSizes([820, 680] if get_language() == "en" else [680, 800])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(top)
        cl.addWidget(split, 1)
        self.setCentralWidget(central)

    def log(self, s):
        self.txt_log.appendPlainText(s)

    # ---------------- config load/save ----------------
    def open_cfg(self):
        if self._has_active_task():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, tr("选择电路配置 YAML"),
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
                "server", "circuits"),
            "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.evaluator = LocalEvaluator(path,
                                            max_jobs=self.spin_jobs.value())
        except Exception as e:
            QMessageBox.critical(self, tr("加载失败"), str(e))
            return
        # Loading a different circuit invalidates every record/summary from
        # the previous circuit.  Active tasks were rejected above, so finished
        # QThreads can be released safely here.
        self.worker = None
        self.nw = None
        self._last_summary = None
        self._objective_snapshot = []
        self._constraints_snapshot = []
        self.cfg_path = path
        self.cfg = self.evaluator.ckt.cfg
        spec = self.evaluator.spec()
        self._fixed_values = dict(spec.get("fixed") or
                                  self.cfg.get("fixed") or {})
        self.lbl_cfg.setText(tr("<b>%s</b>  (%s, %d 参数)")
                             % (spec["circuit"],
                                self.cfg.get("simulator", "spectre"),
                                len(spec["params"])))
        self.metric_names = spec["metrics"]
        initial_points = spec.get("initial_points", [])
        self._seed_extras = copy.deepcopy(initial_points[1:])
        self._review = None
        self.cmb_metric.blockSignals(True)
        self.cmb_metric.clear()
        self.cmb_metric.addItems(self.metric_names)
        self.cmb_metric.blockSignals(False)
        self._fill_params(spec["params"])
        if initial_points:
            for row, param in enumerate(spec["params"]):
                if param["name"] in initial_points[0]:
                    self.tbl_params.item(row, 9).setText(fmt_num(initial_points[0][param["name"]]))
        self.cmb_scope.blockSignals(True)
        self.cmb_scope.clear()
        self.cmb_scope.addItem(tr("All parameter scopes"), None)
        for group in scope_inventory(spec["params"]):
            self.cmb_scope.addItem(group["scope"], group["scope"])
        self.cmb_scope.blockSignals(False)
        self._update_scope_view()
        self.gmid_panel.set_parameters(spec["params"])
        self.metrics_panel.set_config(self.cfg)
        self.spin_stagnation.setValue(spec.get("optimizer", {}).get("stagnation_rounds", 6))
        self._fill_objective(spec["objective"])
        self._fill_constraints(spec.get("constraints", []))
        self.plot_data = []
        self._records = []
        self._record_states = []
        self._pick_points = {}
        self._selected_point = None
        self.lbl_point.setText(tr("点击图中点查看 trial、指标和参数"))
        self._show_no_best()
        self._set_busy(False)
        self._redraw()
        self._render_review()
        self.log(tr("已加载 %s") % path)

    def _fill_params(self, params):
        t = self.tbl_params
        t.setRowCount(0)
        for p in params:
            r = t.rowCount()
            t.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if p.get("enabled", True)
                              else Qt.Unchecked)
            t.setItem(r, 0, chk)
            t.setItem(r, 1, _ro_item(p["name"]))
            t.setItem(r, 2, _ro_item(",".join(p["devices"])))
            t.setItem(r, 3, _ro_item(p["attr"]))
            t.setItem(r, 4, _num_item(p["lo"]))
            t.setItem(r, 5, _num_item(p["hi"]))
            t.setItem(r, 6, _ro_item("√" if p.get("log") else ""))
            t.setItem(r, 7, _ro_item("√" if p.get("integer") else ""))
            nominal = p.get("nominal")
            if not p.get("enabled", True):
                nominal = self._fixed_values.get(p["name"], nominal)
            t.setItem(r, 8, _num_item(nominal, editable=False))
            t.setItem(r, 9, _num_item(nominal))

    def _obj_row_widgets(self, term=None, raw_scale=None):
        r = self.tbl_obj.rowCount()
        self.tbl_obj.insertRow(r)
        cm = QComboBox()
        cm.addItems(self.metric_names)
        cg = QComboBox()
        for g in GOALS:
            cg.addItem(tr(GOAL_CN[g]), g)
        self.tbl_obj.setCellWidget(r, 0, cm)
        self.tbl_obj.setCellWidget(r, 1, cg)
        self.tbl_obj.setItem(r, 2, _num_item(0.0))
        self.tbl_obj.setItem(r, 3, QTableWidgetItem(""))
        self.tbl_obj.setItem(r, 4, _num_item(1.0))
        # Keep scale explicit in the GUI so editing/saving a multi-term
        # objective does not silently reset its normalization.
        self.tbl_obj.setItem(r, 5, _num_item(1.0))
        if term:
            if term.metric in self.metric_names:
                cm.setCurrentText(term.metric)
            cg.setCurrentIndex(GOALS.index(term.goal))
            self.tbl_obj.item(r, 2).setText(fmt_num(term.target))
            if term.tol is not None:
                self.tbl_obj.item(r, 3).setText(fmt_num(float(term.tol)))
            self.tbl_obj.item(r, 4).setText(fmt_num(term.weight))
            scale = term.scale if raw_scale is None else raw_scale
            self.tbl_obj.item(r, 5).setText(fmt_num(scale))

    def add_obj_row(self):
        self._obj_row_widgets()

    def _fill_objective(self, obj_cfg):
        self.tbl_obj.setRowCount(0)
        raw_terms = [obj_cfg] if isinstance(obj_cfg, dict) else list(obj_cfg)
        for i, term in enumerate(Objective(obj_cfg).terms):
            raw = raw_terms[i] if i < len(raw_terms) else {}
            self._obj_row_widgets(term, raw.get("scale"))

    def add_cons_row(self, cfg=None):
        r = self.tbl_cons.rowCount()
        self.tbl_cons.insertRow(r)
        cm = QComboBox()
        cm.addItems(self.metric_names)
        ct = QComboBox()
        ct.addItem(tr("≤ (max)"), "max")
        ct.addItem(tr("≥ (min)"), "min")
        self.tbl_cons.setCellWidget(r, 0, cm)
        self.tbl_cons.setCellWidget(r, 1, ct)
        self.tbl_cons.setItem(r, 2, _num_item(0.0))
        if cfg:
            if cfg["metric"] in self.metric_names:
                cm.setCurrentText(cfg["metric"])
            ct.setCurrentIndex(0 if "max" in cfg else 1)
            self.tbl_cons.item(r, 2).setText(
                fmt_num(float(cfg.get("max", cfg.get("min", 0)))))

    def _fill_constraints(self, cons):
        self.tbl_cons.setRowCount(0)
        for c in cons:
            if "target" in c and "tol" in c:
                self.add_cons_row({"metric": c["metric"], "min": c["target"] - c["tol"]})
                self.add_cons_row({"metric": c["metric"], "max": c["target"] + c["tol"]})
            else:
                self.add_cons_row(c)

    # ---- collect edited state ----
    @staticmethod
    def _read_num(item, label, default=None):
        text = item.text().strip() if item is not None else ""
        if not text:
            return default
        value = parse_num(text)
        if value is None:
            raise ValueError(tr("%s 不是有效数字: %s") % (label, text))
        return value

    def _collect_params(self):
        spec = self.evaluator.spec()["params"]
        edited, fixed, initial = [], {}, {}
        self._edited_param_bounds = {}
        for r, p in enumerate(spec):
            name = p.get("name", tr("第%d行") % (r + 1))
            lo = self._read_num(self.tbl_params.item(r, 4),
                                tr("%s 下限") % name, p.get("lo"))
            hi = self._read_num(self.tbl_params.item(r, 5),
                                tr("%s 上限") % name, p.get("hi"))
            if lo is None or hi is None:
                raise ValueError(tr("参数 %s 必须填写下限和上限") % name)
            if lo >= hi:
                raise ValueError(tr("参数 %s 下限必须小于上限") % name)
            self._edited_param_bounds[name] = (lo, hi)
            q = dict(p)
            q["lo"] = lo
            q["hi"] = hi
            value = self._read_num(self.tbl_params.item(r, 9), tr("Initial value") + " " + name)
            if value is not None:
                initial[name] = value
            q["enabled"] = self.tbl_params.item(r, 0).checkState() == Qt.Checked
            edited.append(q)
            if not q["enabled"]:
                value = initial.get(name, self._fixed_values.get(name, p.get("nominal")))
                if value is not None:
                    value = float(value)
                    if not lo <= value <= hi:
                        raise ValueError(tr("固定参数 %s=%s 超出边界 [%s, %s]") %
                                         (name, fmt_num(value), fmt_num(lo),
                                          fmt_num(hi)))
                    fixed[p["name"]] = value
        return select_scope(edited, self.cmb_scope.currentData(), fixed=fixed, initial_values=initial)

    def _collect_seed_points(self, enabled, fixed):
        names = {p["name"] for p in enabled}
        first = {p["name"]: p["nominal"] for p in enabled if p.get("nominal") is not None}
        points = [{**fixed, **first}] if first else []
        for point in self._seed_extras:
            points.append({**{name: value for name, value in point.items() if name in names}, **fixed})
        return points

    def _update_scope_view(self, *_):
        if not hasattr(self, "tbl_params"):
            return
        if self.evaluator is None:
            self.lbl_scope.setText(tr("Load a circuit to select its parameter scope."))
            return
        selected = self.cmb_scope.currentData()
        local, shared = 0, 0
        for row, p in enumerate(self.evaluator.spec()["params"]):
            scopes = parameter_scopes(p)
            visible = selected is None or selected in scopes
            self.tbl_params.setRowHidden(row, not visible)
            local += bool(visible and (selected is None or scopes == (selected,)))
            shared += bool(selected is not None and selected in scopes and len(scopes) > 1)
            item = self.tbl_params.item(row, 0)
            if item is not None:
                flags = item.flags() | Qt.ItemIsEnabled
                if selected is not None and scopes != (selected,):
                    flags &= ~Qt.ItemIsEnabled
                item.setFlags(flags)
        if selected is None:
            self.lbl_scope.setText(tr("All scopes: %d parameters. Initial values do not edit the source netlist.") % local)
        else:
            self.lbl_scope.setText(tr("%s: %d local parameters, %d shared parameters held fixed. Other scopes keep their initial/netlist values. The existing testbench is used; all instances of this subcircuit master change together.") %
                                   (selected, local, shared))

    def _apply_initial_values(self, values):
        if self.evaluator is None:
            raise ValueError(tr("先打开电路配置 YAML"))
        rows = {p["name"]: (r, p) for r, p in enumerate(self.evaluator.spec()["params"])}
        updates = []
        for name, value in values.items():
            if name not in rows:
                raise ValueError(tr("Unknown initial-value parameter: %s") % name)
            row, p = rows[name]
            lo = self._read_num(self.tbl_params.item(row, 4), name, p["lo"])
            hi = self._read_num(self.tbl_params.item(row, 5), name, p["hi"])
            if isinstance(value, bool) or not math.isfinite(float(value)) or not lo <= float(value) <= hi:
                raise ValueError(tr("Suggested %s=%g is outside [%g, %g]; edit the bounds or sizing target first.") %
                                 (name, float(value), lo, hi))
            if p.get("integer") and not float(value).is_integer():
                raise ValueError(tr("Initial value for %s must be an integer.") % name)
            updates.append((row, float(value)))
        # Validate the complete proposal before changing any table cell.
        for row, value in updates:
            self.tbl_params.item(row, 9).setText(fmt_num(value))

    def _apply_metric_settings(self, proposal):
        if self._has_active_task():
            return
        from optserver.validation import validate_config
        try:
            if self.evaluator is None or self.cfg is None:
                raise ValueError(tr("先打开电路配置 YAML"))
            candidate = copy.deepcopy(self.cfg)
            for analysis in ("tran", "dc"):
                candidate.pop(analysis, None)
            candidate.update(copy.deepcopy(proposal["analyses"]))
            candidate["metrics"] = copy.deepcopy(proposal["metrics"])
            candidate["save"] = list(proposal.get("save", candidate["save"]))
            # Validate current objective edits too: deleting a referenced
            # measurement must not silently select a different objective.
            candidate["objective"] = self._collect_objective()
            candidate["constraints"] = self._collect_constraints()
            validate_config(candidate)
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, tr("Invalid measurements"), str(exc))
            return
        previous = self.cmb_metric.currentText()
        self.cfg.clear()
        self.cfg.update(candidate)
        self.evaluator.ckt.metrics = self.cfg["metrics"]
        self.metric_names = list(self.cfg["metrics"])
        self.metrics_panel.set_config(self.cfg)
        self.cmb_metric.blockSignals(True)
        self.cmb_metric.clear()
        self.cmb_metric.addItems(self.metric_names)
        if previous in self.metric_names:
            self.cmb_metric.setCurrentText(previous)
        self.cmb_metric.blockSignals(False)
        self._fill_objective(candidate["objective"])
        self._fill_constraints(candidate["constraints"])
        self.worker = self.nw = None
        self._last_summary = None
        self._objective_snapshot = []
        self._constraints_snapshot = []
        self._records = []
        self._record_states = []
        self.plot_data = []
        self._pick_points = {}
        self._selected_point = None
        self._review = None
        self.lbl_point.setText(tr("点击图中点查看 trial、指标和参数"))
        self._show_no_best()
        self._set_status("状态：空闲")
        self._redraw()
        self._render_review()
        self.log(tr("Measurements applied. History with different measurement definitions is excluded. Save the configuration to keep these changes."))

    def _apply_gmid_result(self, proposal):
        if self._has_active_task():
            return
        try:
            self._apply_initial_values(proposal["params"])
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, tr("配置错误"), str(exc))
            return
        self.log(tr("gm/Id initial values applied: %s") % json.dumps(proposal["params"], ensure_ascii=False))
        self.tabs.setCurrentIndex(0)

    def _reuse_best(self):
        if self._has_active_task():
            return
        from .review import analyze_records
        try:
            result = analyze_records(self._result_records(), self._collect_objective(),
                                     self._collect_constraints())
            best = result["best_record"]
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, tr("配置错误"), str(exc))
            return
        if best is None:
            QMessageBox.information(self, tr("提示"), tr("No feasible best point is available yet."))
            return
        try:
            self._apply_initial_values(best.get("params") or {})
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, tr("配置错误"), str(exc))
            return
        self.log(tr("Best feasible point copied to initial values. Raise the total budget to continue, or disable resume for a fresh run."))

    def _refresh_review(self):
        if self.evaluator is None:
            return
        from .review import analyze_records
        try:
            objective = getattr(self, "_objective_snapshot", None) or self._collect_objective()
            constraints = (getattr(self, "_constraints_snapshot", []) if self._records else
                           self._collect_constraints())
            self._review = analyze_records(self._result_records(), objective, constraints,
                                          self.evaluator.spec()["params"])
        except Exception as exc:
            self.txt_review.setPlainText(tr("Review failed: %s") % exc)
            return
        self._render_review()

    def _render_review(self):
        if self._review is None:
            self.txt_review.setPlainText(tr("Review the current run to inspect failures, feasibility, stagnation, and parameter associations."))
            return
        result = self._review
        lines = [tr("Evaluations: %d | successful: %d | feasible: %d | failed: %d") %
                 (result["n"], result["n_ok"], result["n_feasible"], result["n_failed"]),
                 tr("Repeated parameter points: %d") % result["repeated_points"]]
        if result["evaluations_since_improvement"] is not None:
            lines.append(tr("Evaluations since last feasible improvement: %d") % result["evaluations_since_improvement"])
        advice = {
            "run_nominal": "Run the nominal point first to check the testbench and metrics.",
            "inspect_failures": "Inspect failed simulation logs before spending more evaluations.",
            "check_constraints": "No feasible point yet: review bias ranges and constraint definitions.",
            "reuse_best": "Reuse the feasible best point as the initial value for the next design round.",
            "enable_exploration": "Progress has stalled: enable stagnation exploration or review the search bounds.",
            "select_scope": "Many parameters vary at once: consider one subcell scope at a time.",
            "collect_more": "Too few valid samples for parameter-association estimates.",
        }
        lines += ["", tr("Suggested next steps:")]
        lines += ["• " + tr(advice[code]) for code in result["advice"]]
        if result["associations"]:
            lines += ["", tr("Historical rank association with the objective (not causal sensitivity):")]
            lines.append(tr("Population: feasible points") if result["association_population"] == "feasible"
                         else tr("Population: all valid points, including constraint violations"))
            lines += ["%s: ρ=%+.3f (n=%d)" % (entry["parameter"], entry["rho"], entry["n"])
                      for entry in result["associations"][:12]]
        if result["errors"]:
            lines += ["", tr("Most frequent simulation errors:")]
            lines += ["%d × %s" % (entry["count"], entry["error"]) for entry in result["errors"]]
        self.txt_review.setPlainText("\n".join(lines))

    def _collect_objective(self):
        terms = []
        for r in range(self.tbl_obj.rowCount()):
            metric = self.tbl_obj.cellWidget(r, 0).currentText()
            goal = self.tbl_obj.cellWidget(r, 1).currentData()
            tgt = self._read_num(self.tbl_obj.item(r, 2),
                                 tr("目标项第%d行目标值") % (r + 1), 0.0)
            tol_txt = self.tbl_obj.item(r, 3).text().strip()
            tol = self._read_num(self.tbl_obj.item(r, 3),
                                 tr("目标项第%d行容差") % (r + 1))
            w = self._read_num(self.tbl_obj.item(r, 4),
                               tr("目标项第%d行权重") % (r + 1), 1.0)
            scale = self._read_num(self.tbl_obj.item(r, 5),
                                   tr("目标项第%d行尺度") % (r + 1), 1.0)
            if w < 0:
                raise ValueError(tr("目标项第%d行权重不能为负数") % (r + 1))
            if scale <= 0:
                raise ValueError(tr("目标项第%d行尺度必须大于0") % (r + 1))
            t = {"metric": metric, "goal": goal, "weight": w}
            if goal == "target":
                t["target"] = tgt
                if tol_txt:
                    if tol is None:
                        raise ValueError(tr("目标项第%d行容差不是有效数字") %
                                         (r + 1))
                    if tol < 0:
                        raise ValueError(tr("目标项第%d行容差不能为负数") %
                                         (r + 1))
                    t["tol"] = tol
            if scale != 1.0 or self.tbl_obj.item(r, 5).text().strip():
                t["scale"] = scale
            terms.append(t)
        return terms

    def _collect_constraints(self):
        out = []
        for r in range(self.tbl_cons.rowCount()):
            metric = self.tbl_cons.cellWidget(r, 0).currentText()
            kind = self.tbl_cons.cellWidget(r, 1).currentData()
            v = self._read_num(self.tbl_cons.item(r, 2),
                               tr("约束第%d行边界值") % (r + 1))
            if v is None:
                continue
            out.append({"metric": metric, kind: v})
        return out

    def save_cfg(self):
        if not self.cfg or self._has_active_task():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("保存配置副本"), self.cfg_path or "", "YAML (*.yaml)")
        if not path:
            return
        try:
            enabled, fixed = self._collect_params()
            enabled_by_name = {p["name"]: p for p in enabled}
            # Deep-copy nested simulator/PDK mappings.  Circuit has already
            # resolved netlist/stimuli/model paths in cfg, so those absolute
            # paths are the source of truth for a saved GUI copy.
            cfg = copy.deepcopy(self.cfg)
            newp = []
            for p in cfg.get("params", []):
                name = p["name"]
                q = copy.deepcopy(p)
                if name in self._edited_param_bounds:
                    q["lo"], q["hi"] = self._edited_param_bounds[name]
                edited = enabled_by_name.get(name)
                if edited is not None:
                    q["lo"] = edited["lo"]
                    q["hi"] = edited["hi"]
                    q["enabled"] = True
                else:
                    q["enabled"] = False
                newp.append(q)
            cfg["params"] = newp
            # Keep fixed nominal values separate from params so Circuit can
            # merge them into every evaluation while retaining all rows.
            cfg["fixed"] = fixed
            cfg["initial_points"] = self._collect_seed_points(enabled, fixed)
            cfg.setdefault("optimizer", {})["stagnation_rounds"] = self.spin_stagnation.value()
            obj = self._collect_objective()
            cfg["objective"] = obj[0] if len(obj) == 1 else obj
            cfg["constraints"] = self._collect_constraints()
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            self.log(tr("配置保存失败: %s") % e)
            QMessageBox.critical(self, tr("保存失败"), str(e))
            return
        self.log(tr("配置已保存: %s（注意：原文件注释不保留）") % path)

    # ---------------- run ----------------
    def _has_active_task(self):
        return bool((self.worker is not None and self.worker.isRunning()) or
                    (self.nw is not None and self.nw.isRunning()))

    def _set_busy(self, busy):
        """Lock every mutable configuration control while a task is active."""
        if not hasattr(self, "tabs"):
            return
        self.tabs.setEnabled(not busy)
        for button in (self.btn_open, self.btn_save, self.btn_nominal,
                       self.btn_wb, self.btn_start):
            button.setEnabled(not busy)
        for widget in (self.cmb_device, self.spin_jobs):
            widget.setEnabled(not busy)
        self.btn_stop.setEnabled(bool(busy))
        self._set_status("状态：运行中" if busy else "状态：空闲")

    def _snapshot_run_settings(self, enabled, fixed, objective,
                               constraints):
        """Read Qt state once, on the GUI thread, before creating QThread."""
        return {
            "evaluator": self.evaluator,
            "outdir": os.path.join(self.evaluator.root, "gui_run"),
            "objective": copy.deepcopy(objective),
            "constraints": copy.deepcopy(constraints),
            "batch": self.spin_batch.value(),
            "n_init": self.spin_init.value(),
            "device": self.cmb_device.currentText(),
            "use_dkl": self.chk_dkl.isChecked(),
            "dkl_after": self.spin_dkl_after.value(),
            "seed": self.spin_seed.value(),
            "spec_params": copy.deepcopy(enabled),
            "fixed": copy.deepcopy(fixed),
            "remote_gpu": self.edit_remote.text().strip() or None,
            "budget": self.spin_budget.value(),
            "resume": self.chk_resume.isChecked(),
            "max_jobs": self.spin_jobs.value(),
            "initial_points": self._collect_seed_points(enabled, fixed),
            "stagnation_rounds": self.spin_stagnation.value(),
        }

    def _apply_runtime_bounds(self, enabled):
        """Make edited GUI bounds effective to Circuit.apply_params.

        Circuit keeps parameter dictionaries shared with ``ckt.cfg``.  Updating
        those in-memory dictionaries avoids rewriting the source YAML while
        ensuring the netlist patcher uses the same bounds that Space sees.
        """
        ckt = getattr(self.evaluator, "ckt", None)
        if ckt is None:
            return
        by_name = {p["name"]: p for p in enabled}
        edited_bounds = getattr(self, "_edited_param_bounds", {})
        params = getattr(ckt, "params", None) or ckt.cfg.get("params", [])
        for p in params:
            edited = by_name.get(p.get("name"))
            bounds = edited_bounds.get(p.get("name"))
            if edited is None and bounds is None:
                continue
            if bounds is None:
                bounds = (edited["lo"], edited["hi"])
            p["lo"], p["hi"] = bounds

    @staticmethod
    def _engine_from_snapshot(snapshot, on_log, on_record):
        from .engine import Engine
        return Engine(
            snapshot["evaluator"], snapshot["outdir"],
            objective=copy.deepcopy(snapshot["objective"]),
            constraints=copy.deepcopy(snapshot["constraints"]),
            batch=snapshot["batch"], n_init=snapshot["n_init"],
            device=snapshot["device"], use_dkl=snapshot["use_dkl"],
            dkl_after=snapshot["dkl_after"], seed=snapshot["seed"],
            on_log=on_log, on_record=on_record,
            spec_params=copy.deepcopy(snapshot["spec_params"]),
            fixed=copy.deepcopy(snapshot["fixed"]),
            remote_gpu=snapshot["remote_gpu"],
            initial_points=copy.deepcopy(snapshot.get("initial_points", [])),
            stagnation_rounds=snapshot.get("stagnation_rounds", 0))

    def test_remote(self):
        if self._has_active_task():
            return
        url = self.edit_remote.text().strip()
        if not url:
            self.log(tr("远程GPU: 未填写 URL"))
            return
        try:
            from .proposer import RemoteProposer
            h = RemoteProposer(url).health()
            self.log(tr("远程GPU可用: %s (torch %s)")
                     % (h.get("device_name"), h.get("torch")))
        except Exception as e:
            self.log(tr("远程GPU不可达: %s") % e)

    def run_nominal(self):
        if not self.evaluator or self._has_active_task():
            return
        try:
            enabled, fixed = self._collect_params()
            points = self._collect_seed_points(enabled, fixed)
            params = copy.deepcopy(points[0] if points else fixed)
            self._apply_runtime_bounds(enabled)
        except Exception as e:
            self.log(tr("配置错误: %s") % e)
            QMessageBox.critical(self, tr("配置错误"), str(e))
            return
        # All parameter values are copied before the worker is created.  The
        # nominal thread never reads a widget or mutable table item.
        self.nw = NominalWorker(self.evaluator, params)
        self.nw.sig_log.connect(self.log)
        self.nw.finished.connect(self._nominal_finished)
        self._set_busy(True)
        self._set_status("状态：nominal 运行中")
        self.nw.start()

    def start_run(self):
        if not self.evaluator:
            QMessageBox.information(self, tr("提示"), tr("先打开电路配置 YAML"))
            return
        if self._has_active_task():
            return
        try:
            enabled, fixed = self._collect_params()
            objective = self._collect_objective()
            constraints = self._collect_constraints()
            if not enabled:
                raise ValueError(tr("至少启用一个参数"))
            if not objective:
                raise ValueError(tr("至少一个目标项"))
        except Exception as e:
            self.log(tr("配置错误: %s") % e)
            QMessageBox.critical(self, tr("配置错误"), str(e))
            return
        self._apply_runtime_bounds(enabled)
        # This is intentionally before RunWorker construction: every value
        # read from Qt is frozen in one GUI-thread snapshot.
        snapshot = self._snapshot_run_settings(enabled, fixed, objective,
                                                constraints)
        self.evaluator.max_jobs = snapshot["max_jobs"]

        def make_engine(on_log, on_record, snap=snapshot):
            return self._engine_from_snapshot(snap, on_log, on_record)

        self.plot_data = []
        self._records = []
        self._record_states = []
        self._pick_points = {}
        self._headline = objective[0]["metric"]
        self._objective_snapshot = copy.deepcopy(objective)
        self._constraints_snapshot = copy.deepcopy(constraints)
        self._selected_point = None
        self.lbl_point.setText(tr("点击图中点查看 trial、指标和参数"))
        self._show_no_best()
        self.worker = RunWorker(make_engine, snapshot["budget"],
                                snapshot["resume"])
        self.worker.sig_log.connect(self.log)
        self.worker.sig_record.connect(self.on_record)
        self.worker.sig_done.connect(self.on_done)
        self.worker.finished.connect(self._worker_finished)
        self._set_busy(True)
        self._set_status("状态：准备运行 0/%d", snapshot["budget"])
        self.worker.start()

    def stop_run(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.log(tr("停止请求已发出（本轮仿真结束后停）…"))
            self._set_status("状态：正在停止…")
        elif self.nw and self.nw.isRunning():
            self.nw.request_stop()
            self.log(tr("nominal 停止请求已发出（当前仿真结束后停）…"))
            self._set_status("状态：正在停止 nominal…")

    def on_record(self, payload):
        rec, st = payload
        rec = copy.deepcopy(rec or {})
        st = copy.deepcopy(st or {})
        ok = bool(rec.get("ok")) and isinstance(rec.get("metrics"), dict)
        feas = self._record_feasible(rec, st) if ok else False
        head = (rec.get("metrics") or {}).get(getattr(
            self, "_headline", self._metric_name()))
        self._records.append(rec)
        self._record_states.append(st)
        # Keep the old public plot_data shape while richer records live in the
        # parallel lists used by the multi-metric chart.
        self.plot_data.append((len(self.plot_data) + 1, head, feas, ok))
        self._update_status(st, done=False)
        self._redraw()
        best = self._best_feasible_record(st)
        if best is not None:
            self._show_best(best, st)

    def _metric_name(self):
        if getattr(self, "cmb_metric", None) is not None:
            name = self.cmb_metric.currentText().strip()
            if name:
                return name
        return getattr(self, "_headline", "")

    def _record_feasible(self, rec, st=None):
        if st and "last_feasible" in st:
            return bool(st["last_feasible"])
        metrics = rec.get("metrics") or {}
        if not rec.get("ok") or not metrics:
            return False
        constraints = list(getattr(self, "_constraints_snapshot", []) or [])
        for c in constraints:
            try:
                value = float(metrics[c["metric"]])
                if "max" in c and value > float(c["max"]):
                    return False
                if "min" in c and value < float(c["min"]):
                    return False
                if "target" in c and "tol" in c and \
                        abs(value - float(c["target"])) > float(c["tol"]):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        for term in (getattr(self, "_objective_snapshot", []) or []):
            if term.get("goal") == "target" and term.get("tol") is not None:
                try:
                    if abs(float(metrics[term["metric"]]) -
                           float(term["target"])) > float(term["tol"]):
                        return False
                except (KeyError, TypeError, ValueError):
                    return False
        return True

    def _best_feasible_record(self, st=None):
        """Normalize summary variants emitted by old and new Engine builds."""
        st = st or {}
        candidate = st.get("best_feasible")
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            if 0 <= candidate < len(self._records):
                return self._records[candidate]
        if candidate is False:
            return None
        if st.get("n_feasible", 0) <= 0:
            return None
        rec = st.get("best_record")
        if isinstance(rec, dict):
            return rec
        idx = st.get("best_index")
        if isinstance(idx, int) and 0 <= idx < len(self._records):
            return self._records[idx]
        for rec, state in reversed(list(zip(self._records,
                                             self._record_states))):
            if self._record_feasible(rec, state):
                return rec
        return None

    def _update_status(self, st, done=False):
        st = st if isinstance(st, dict) else {}
        n = st.get("n", len(self._records))
        budget = getattr(self.worker, "budget", None)
        n_ok = st.get("n_ok", sum(bool(r.get("ok")) for r in self._records))
        n_feasible = st.get("n_feasible", 0)
        status = st.get("status")
        if done:
            text = tr(status) if status else (tr("已停止") if self.worker and
                              self.worker._stop_requested.is_set()
                              else tr("已完成"))
        else:
            text = tr(status) if status else tr("运行中")
        suffix = (tr("  n=%s%s  成功=%s  可行=%s") %
                   (n, ("/%s" % budget) if budget else "", n_ok,
                    n_feasible))
        self._set_status("状态：%s%s", text, suffix)
        self._status_summary = (copy.deepcopy(st), done)

    def _redraw(self):
        self.ax.clear()
        self._pick_points = {}
        metric = self._metric_name()
        feasible_points, infeasible_points, failed = [], [], []
        entries = []
        for i, (rec, st) in enumerate(zip(self._records,
                                          self._record_states), start=1):
            ok = bool(rec.get("ok")) and isinstance(rec.get("metrics"), dict)
            value = (rec.get("metrics") or {}).get(metric) if ok else None
            try:
                value = float(value)
                if value != value or abs(value) == float("inf"):
                    value = None
            except (TypeError, ValueError):
                value = None
            entry = (i, rec, st, value)
            entries.append(entry)
            if value is None:
                if not ok:
                    failed.append(entry)
                continue
            if self._record_feasible(rec, st):
                feasible_points.append(entry)
            else:
                infeasible_points.append(entry)

        def add_points(points, **kwargs):
            if not points:
                return None
            artist = self.ax.scatter([p[0] for p in points],
                                     [p[3] for p in points],
                                     picker=5, **kwargs)
            self._pick_points[artist] = points
            return artist

        add_points(feasible_points, s=28, color="tab:blue", marker="o",
                   label=tr("可行"))
        add_points(infeasible_points, s=28, color="tab:orange", marker="o",
                   label=tr("约束失败"))
        if failed:
            # These markers use axes coordinates below the plot: no fabricated
            # metric value is assigned to a failed simulation.
            artist = self.ax.scatter(
                [p[0] for p in failed], [-0.18] * len(failed),
                transform=self.ax.get_xaxis_transform(), clip_on=False,
                picker=5, s=34, color="tab:red", marker="x",
                label=tr("仿真失败（下方）"))
            self._pick_points[artist] = failed

        bx, by = self._best_trend(entries, metric)
        if bx:
            self.ax.plot(bx, by, "r-", lw=2, drawstyle="steps-post",
                         label=tr("最佳可行值"))
            self.ax.plot(bx[-1], by[-1], "r*", ms=9, picker=5)
        if failed:
            self.ax.text(0.01, -0.28, tr("×=仿真失败（无指标值）"),
                         transform=self.ax.transAxes, fontsize=8,
                         color="tab:red", va="top")
        self.ax.set_xlabel(tr("trial / 评估次数"))
        self.ax.set_ylabel(metric)
        self.ax.set_title(tr("实时迭代点阵"))
        self.ax.grid(alpha=0.3)
        # Leave room for the bilingual x label and the failure status strip
        # below the data area, both in the live canvas and exported PNG.
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.86,
                                 bottom=0.34)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def _best_trend(self, entries, metric):
        term = None
        for candidate in getattr(self, "_objective_snapshot", []) or []:
            if candidate.get("metric") == metric:
                term = candidate
                break
        if term is None:
            return [], []
        goal = term.get("goal", "maximize")
        target = float(term.get("target", 0.0))
        best_value = None
        best_key = None
        bx, by = [], []
        for x, rec, st, value in entries:
            if value is None or not self._record_feasible(rec, st):
                continue
            key = (abs(value - target) if goal == "target" else
                   -value if goal == "maximize" else value)
            if best_key is None or key < best_key:
                best_key = key
                best_value = value
            bx.append(x)
            by.append(best_value)
        return bx, by

    def _on_metric_changed(self, _index):
        if hasattr(self, "ax"):
            self._redraw()

    def _on_pick(self, event):
        points = self._pick_points.get(event.artist)
        if not points:
            return
        raw_indices = getattr(event, "ind", None)
        if raw_indices is None:
            return
        indices = list(raw_indices)
        if not indices:
            return
        entry = points[indices[0]]
        self._show_record_details(entry[1], self._metric_name(), entry[3])

    def _render_point_label(self, rec, metric, value):
        val = tr("无") if value is None else "%.6g" % value
        status = tr("成功") if rec.get("ok") else tr("仿真失败")
        self.lbl_point.setText("trial=%s  %s=%s  %s%s" %
                               (rec.get("trial", ""), metric, val, status,
                                ("  " + tr("error") + "=" + str(rec.get("error"))
                                 if rec.get("error") else "")))

    def _show_record_details(self, rec, metric, value):
        self._selected_point = (rec, metric, value)
        self._detail_view = ("point", (rec, metric, value))
        self._render_point_label(rec, metric, value)
        val = tr("无") if value is None else "%.6g" % value
        rows = [("trial", rec.get("trial", "")),
                (tr("成功"), tr("是") if rec.get("ok") else tr("否")),
                (tr("指标 ") + metric, val)]
        if rec.get("error"):
            rows.append((tr("error"), rec.get("error")))
        rows.extend((tr("参数 ") + k, fmt_num(v) if isinstance(v, (int, float))
                     else v) for k, v in (rec.get("params") or {}).items())
        self.tbl_best.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            self.tbl_best.setItem(i, 0, _ro_item(a))
            self.tbl_best.setItem(i, 1, _ro_item(b))

    def _show_best(self, rec, st):
        self._detail_view = ("best", (rec, st))
        t = self.tbl_best
        rows = []
        rows.append((tr("最佳可行 trial"), str(rec.get("trial"))))
        rows.append((tr("可行"), tr("是")))
        for k, v in (rec.get("metrics") or {}).items():
            try:
                value = "%.6g" % float(v)
            except (TypeError, ValueError):
                value = str(v)
            rows.append((tr("指标 ") + k, value))
        for k, v in (rec.get("params") or {}).items():
            try:
                value = fmt_num(float(v))
            except (TypeError, ValueError):
                value = str(v)
            rows.append((tr("参数 ") + k, value))
        t.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            t.setItem(i, 0, _ro_item(a))
            t.setItem(i, 1, _ro_item(b))

    def _show_no_best(self):
        self._detail_view = ("empty", ())
        self.tbl_best.setRowCount(1)
        self.tbl_best.setItem(0, 0, _ro_item(tr("最佳可行")))
        self.tbl_best.setItem(0, 1, _ro_item(tr("无可行 best")))

    def on_done(self, summary):
        self._last_summary = summary or {}
        if summary:
            self._update_status(summary, done=True)
            best = self._best_feasible_record(summary)
            n = summary.get("n", len(self._records))
            feasible = summary.get("n_feasible", 0)
            if best is not None and feasible:
                self._show_best(best, summary)
                self.log(tr("运行结束: n=%d 可行=%d（已生成可行 best.json）") %
                         (n, feasible))
            else:
                self._show_no_best()
                self.log(tr("运行结束: n=%d 可行=%d（没有可行 best）") %
                         (n, feasible))
        else:
            self._show_no_best()
            self._set_status("状态：运行失败（详见日志）")

    def _worker_finished(self):
        self._set_busy(False)
        if hasattr(self, "_last_summary"):
            if self._last_summary:
                self._update_status(self._last_summary, done=True)
            else:
                self._set_status("状态：运行失败（详见日志）")
        self._refresh_review()
        self._maybe_close_after_stop()

    def _nominal_finished(self):
        self._set_busy(False)
        self._set_status("状态：nominal 完成")
        self._maybe_close_after_stop()

    def _maybe_close_after_stop(self):
        if self._closing_after_stop and not self._has_active_task():
            self._closing_after_stop = False
            self.close()

    def _result_dir(self):
        if self.evaluator is not None:
            return os.path.abspath(os.path.join(self.evaluator.root,
                                                "gui_run"))
        return os.path.abspath("gui_run")

    def _result_records(self):
        if self.worker is not None and self.worker.engine is not None:
            records = getattr(self.worker.engine, "records", None)
            if records:
                return list(records)
        if self._records:
            return list(self._records)
        if self.evaluator is not None:
            try:
                signature = self.evaluator.spec().get("measurement_signature")
                return [record for record in self.evaluator.history()
                        if not signature or record.get("measurement_signature") == signature]
            except Exception as e:
                self.log(tr("读取历史失败: %s") % e)
        return []

    def export_results_csv(self):
        from .report import write_records_csv
        default = os.path.join(self._result_dir(), "results.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, tr("导出结果 CSV"), default, "CSV (*.csv)")
        if not path:
            return
        try:
            write_records_csv(self._result_records(), path)
            self.log(tr("结果 CSV 已导出: %s") % path)
        except Exception as e:
            self.log(tr("结果 CSV 导出失败: %s") % e)
            QMessageBox.critical(self, tr("导出失败"), str(e))

    def export_plot(self):
        default = os.path.join(self._result_dir(), "convergence.png")
        path, _ = QFileDialog.getSaveFileName(
            self, tr("导出迭代图"), default, "PNG (*.png)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self.log(tr("迭代图已导出: %s") % path)
        except Exception as e:
            self.log(tr("迭代图导出失败: %s") % e)
            QMessageBox.critical(self, tr("导出失败"), str(e))

    def open_results_dir(self):
        path = self._result_dir()
        try:
            os.makedirs(path, exist_ok=True)
            ok = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            if not ok:
                self.log(tr("无法打开结果目录: %s") % path)
            else:
                self.log(tr("已请求打开结果目录: %s") % path)
            return ok
        except Exception as e:
            self.log(tr("打开结果目录失败: %s") % e)
            return False

    def closeEvent(self, event):
        if self._has_active_task():
            answer = QMessageBox.question(
                self, tr("任务仍在运行"),
                tr("当前任务仍在运行。停止任务并在它结束后关闭窗口？"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
            if answer == QMessageBox.Yes:
                self._closing_after_stop = True
                self.stop_run()
            # QThread must finish naturally; never accept a close event while
            # its worker is still executing.
            event.ignore()
            return
        event.accept()

    # ---------------- writeback ----------------
    def writeback(self):
        if not self.evaluator or self._has_active_task():
            return
        summary = getattr(self, "_last_summary", None)
        if summary is not None and summary.get("n_feasible", 0) <= 0:
            QMessageBox.information(self, tr("提示"),
                                    tr("当前运行没有可行 best，不能回写。"))
            return
        from . import writeback as wb
        best_path = os.path.join(self.evaluator.root, "gui_run",
                                 "best.json")
        if not os.path.exists(best_path):
            QMessageBox.information(self, tr("提示"),
                                    tr("还没有 best.json（先跑一次优化）"))
            return
        with open(best_path) as f:
            best = json.load(f)
        if not best.get("feasible", False):
            QMessageBox.information(self, tr("提示"), tr("保存的 best 尚未满足约束，不能回写。"))
            return
        if not self.cfg.get("virtuoso", {}).get("lib"):
            QMessageBox.warning(
                self, tr("缺少配置"),
                tr("YAML 需要 virtuoso: {lib: <库名>, top_cell: <顶层cell>}"))
            return
        if not wb.bridge_alive():
            QMessageBox.warning(self, tr("bridge 未运行"),
                                tr("virtuoso-bridge (127.0.0.1:65036) 未响应"))
            return
        edits, skipped = wb.plan(self.cfg, self.evaluator.spec()["params"],
                                 best["params"])
        ok, cur = wb.read_current(self.cfg, edits)
        lines = [tr("将写入 %d 个属性（lib=%s）:") %
                 (len(edits), self.cfg["virtuoso"]["lib"]), ""]
        lines += [tr("  %s/%s %s = %s   (参数 %s)") % e for e in edits]
        if skipped:
            lines += ["", tr("跳过（需在 YAML 加 sch_prop）:")] + \
                     ["  %s: %s" % s for s in skipped]
        lines += ["", tr("当前原理图值: %s") % (cur if ok else tr("读取失败"))]
        dlg = QDialog(self)
        dlg.setWindowTitle(tr("回写预览"))
        dl = QVBoxLayout(dlg)
        txt = QPlainTextEdit("\n".join(lines))
        txt.setReadOnly(True)
        dl.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.Ok |
                              QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        dl.addWidget(bb)
        dlg.resize(760, 480)
        if dlg.exec_() != QDialog.Accepted:
            return
        ok, msg = wb.apply_edits(self.cfg, edits)
        self.log((tr("回写完成: ") if ok else tr("回写失败: ")) + msg)
        if ok:
            QMessageBox.information(
                self, tr("完成"),
                tr("已写回并保存原理图。\n注意重新 netlist 后才会反映在仿真里。"))


def main(language=None):
    import sys
    app = QApplication(sys.argv)
    win = MainWindow(language=language)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
