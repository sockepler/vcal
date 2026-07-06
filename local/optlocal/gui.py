# -*- coding: utf-8 -*-
"""circuit-opt 本地优化 GUI（PyQt5，全离线：本地仿真 + 本地 BoTorch）。

布局仿 gmid-tool：左侧参数/目标/设置，右侧收敛曲线 + 最优点 + 日志。
"""
import json
import os

import yaml
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from optserver.hspice_netlist import fmt_num, parse_num

from .evaluator import LocalEvaluator
from .objective import Objective

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

    def run(self):
        try:
            self.engine = self.make_engine(
                on_log=lambda s: self.sig_log.emit(s),
                on_record=lambda rec, st: self.sig_record.emit((rec, st)))
            if self.resume:
                self.engine.resume()
            self.engine.run(self.budget)
            self.sig_done.emit(self.engine.summary())
        except Exception as e:
            import traceback
            self.sig_log.emit("ERROR: %s\n%s" % (e,
                                                 traceback.format_exc()))
            self.sig_done.emit(None)

    def request_stop(self):
        if self.engine is not None:
            self.engine.stop_event.set()


class NominalWorker(QThread):
    sig_log = pyqtSignal(str)

    def __init__(self, evaluator, params):
        super().__init__()
        self.ev = evaluator
        self.params = params

    def run(self):
        self.sig_log.emit("nominal 试跑中…")
        r = self.ev.evaluate(self.params)
        if r.get("ok"):
            self.sig_log.emit("nominal 指标: " + json.dumps(
                {k: round(v, 6) for k, v in r["metrics"].items()},
                ensure_ascii=False))
        else:
            self.sig_log.emit("nominal 失败: %s" % r.get("error"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("circuit-opt 本地调参 (BO/DKL, 离线)")
        self.resize(1500, 900)
        self.cfg = None
        self.cfg_path = None
        self.evaluator = None
        self.worker = None
        self.plot_data = []          # (idx, headline, feasible, ok)
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = QWidget()
        tl = QHBoxLayout(top)
        self.btn_open = QPushButton("打开电路配置…")
        self.btn_open.clicked.connect(self.open_cfg)
        self.lbl_cfg = QLabel("<i>未加载</i>")
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cuda", "cpu"])
        self.spin_jobs = QSpinBox()
        self.spin_jobs.setRange(1, 16)
        self.spin_jobs.setValue(4)
        self.btn_nominal = QPushButton("Nominal 试跑")
        self.btn_nominal.clicked.connect(self.run_nominal)
        self.btn_start = QPushButton("▶ 开始优化")
        self.btn_start.clicked.connect(self.start_run)
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_stop.setEnabled(False)
        self.btn_wb = QPushButton("回写原理图…")
        self.btn_wb.clicked.connect(self.writeback)
        self.btn_save = QPushButton("保存配置副本…")
        self.btn_save.clicked.connect(self.save_cfg)
        for w in (self.btn_open, self.lbl_cfg):
            tl.addWidget(w)
        tl.addStretch(1)
        tl.addWidget(QLabel("计算设备:"))
        tl.addWidget(self.cmb_device)
        tl.addWidget(QLabel("并行仿真:"))
        tl.addWidget(self.spin_jobs)
        for w in (self.btn_nominal, self.btn_start, self.btn_stop,
                  self.btn_wb, self.btn_save):
            tl.addWidget(w)

        # ---- left tabs ----
        tabs = QTabWidget()
        # params
        self.tbl_params = QTableWidget(0, 9)
        self.tbl_params.setHorizontalHeaderLabels(
            ["启用", "参数", "器件", "属性", "下限", "上限", "log",
             "整数", "nominal"])
        self.tbl_params.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        tabs.addTab(self.tbl_params, "参数空间")
        # objective + constraints
        w2 = QWidget()
        v2 = QVBoxLayout(w2)
        v2.addWidget(QLabel("优化目标（多项加权求和；「逼近目标值」= 最小化 |指标−目标|）"))
        self.tbl_obj = QTableWidget(0, 5)
        self.tbl_obj.setHorizontalHeaderLabels(
            ["指标", "方向", "目标值", "容差(可选)", "权重"])
        v2.addWidget(self.tbl_obj)
        h2 = QHBoxLayout()
        b_add = QPushButton("+ 目标项")
        b_add.clicked.connect(lambda: self.add_obj_row())
        b_del = QPushButton("− 删除选中")
        b_del.clicked.connect(
            lambda: self.tbl_obj.removeRow(self.tbl_obj.currentRow()))
        h2.addWidget(b_add)
        h2.addWidget(b_del)
        h2.addStretch(1)
        v2.addLayout(h2)
        v2.addWidget(QLabel("硬约束（不满足视为不可行）"))
        self.tbl_cons = QTableWidget(0, 3)
        self.tbl_cons.setHorizontalHeaderLabels(["指标", "类型", "边界值"])
        v2.addWidget(self.tbl_cons)
        h3 = QHBoxLayout()
        b_add2 = QPushButton("+ 约束")
        b_add2.clicked.connect(lambda: self.add_cons_row())
        b_del2 = QPushButton("− 删除选中")
        b_del2.clicked.connect(
            lambda: self.tbl_cons.removeRow(self.tbl_cons.currentRow()))
        h3.addWidget(b_add2)
        h3.addWidget(b_del2)
        h3.addStretch(1)
        v2.addLayout(h3)
        tabs.addTab(w2, "目标与约束")
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
        self.chk_dkl = QCheckBox("样本足够后切换深度核 GP (DKL)")
        self.chk_dkl.setChecked(True)
        self.spin_dkl_after = QSpinBox()
        self.spin_dkl_after.setRange(8, 4096)
        self.spin_dkl_after.setValue(48)
        self.chk_resume = QCheckBox("从历史续跑 (history.jsonl)")
        self.chk_resume.setChecked(True)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        rw = QWidget()
        rh = QHBoxLayout(rw)
        rh.setContentsMargins(0, 0, 0, 0)
        self.edit_remote = QLineEdit()
        self.edit_remote.setPlaceholderText(
            "如 http://your-gpu-host:8494（空=本机计算）")
        btn_rtest = QPushButton("测试")
        btn_rtest.clicked.connect(self.test_remote)
        rh.addWidget(self.edit_remote, 1)
        rh.addWidget(btn_rtest)
        f3.addRow("总评估次数:", self.spin_budget)
        f3.addRow("每轮提案数:", self.spin_batch)
        f3.addRow("初始采样数:", self.spin_init)
        f3.addRow(self.chk_dkl)
        f3.addRow("DKL 启用样本数:", self.spin_dkl_after)
        f3.addRow(self.chk_resume)
        f3.addRow("随机种子:", self.spin_seed)
        f3.addRow("远程GPU计算服务:", rw)
        tabs.addTab(w3, "运行设置")

        # ---- right side ----
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.tbl_best = QTableWidget(0, 2)
        self.tbl_best.setHorizontalHeaderLabels(["项", "值"])
        self.tbl_best.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(5000)
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.canvas)
        right.addWidget(self.tbl_best)
        right.addWidget(self.txt_log)
        right.setSizes([420, 260, 180])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(tabs)
        split.addWidget(right)
        split.setSizes([680, 800])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(top)
        cl.addWidget(split, 1)
        self.setCentralWidget(central)

    def log(self, s):
        self.txt_log.appendPlainText(s)

    # ---------------- config load/save ----------------
    def open_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择电路配置 YAML",
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
            QMessageBox.critical(self, "加载失败", str(e))
            return
        self.cfg_path = path
        self.cfg = self.evaluator.ckt.cfg
        spec = self.evaluator.spec()
        self.lbl_cfg.setText("<b>%s</b>  (%s, %d 参数)"
                             % (spec["circuit"],
                                self.cfg.get("simulator", "spectre"),
                                len(spec["params"])))
        self.metric_names = spec["metrics"]
        self._fill_params(spec["params"])
        self._fill_objective(spec["objective"])
        self._fill_constraints(spec.get("constraints", []))
        self.plot_data = []
        self._redraw()
        self.log("已加载 %s" % path)

    def _fill_params(self, params):
        t = self.tbl_params
        t.setRowCount(0)
        for p in params:
            r = t.rowCount()
            t.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked)
            t.setItem(r, 0, chk)
            t.setItem(r, 1, _ro_item(p["name"]))
            t.setItem(r, 2, _ro_item(",".join(p["devices"])))
            t.setItem(r, 3, _ro_item(p["attr"]))
            t.setItem(r, 4, _num_item(p["lo"]))
            t.setItem(r, 5, _num_item(p["hi"]))
            t.setItem(r, 6, _ro_item("√" if p.get("log") else ""))
            t.setItem(r, 7, _ro_item("√" if p.get("integer") else ""))
            t.setItem(r, 8, _num_item(p.get("nominal"), editable=False))

    def _obj_row_widgets(self, term=None):
        r = self.tbl_obj.rowCount()
        self.tbl_obj.insertRow(r)
        cm = QComboBox()
        cm.addItems(self.metric_names)
        cg = QComboBox()
        for g in GOALS:
            cg.addItem(GOAL_CN[g], g)
        self.tbl_obj.setCellWidget(r, 0, cm)
        self.tbl_obj.setCellWidget(r, 1, cg)
        self.tbl_obj.setItem(r, 2, _num_item(0.0))
        self.tbl_obj.setItem(r, 3, QTableWidgetItem(""))
        self.tbl_obj.setItem(r, 4, _num_item(1.0))
        if term:
            if term.metric in self.metric_names:
                cm.setCurrentText(term.metric)
            cg.setCurrentIndex(GOALS.index(term.goal))
            self.tbl_obj.item(r, 2).setText(fmt_num(term.target))
            if term.tol is not None:
                self.tbl_obj.item(r, 3).setText(fmt_num(float(term.tol)))
            self.tbl_obj.item(r, 4).setText(fmt_num(term.weight))

    def add_obj_row(self):
        self._obj_row_widgets()

    def _fill_objective(self, obj_cfg):
        self.tbl_obj.setRowCount(0)
        for term in Objective(obj_cfg).terms:
            self._obj_row_widgets(term)

    def add_cons_row(self, cfg=None):
        r = self.tbl_cons.rowCount()
        self.tbl_cons.insertRow(r)
        cm = QComboBox()
        cm.addItems(self.metric_names)
        ct = QComboBox()
        ct.addItem("≤ (max)", "max")
        ct.addItem("≥ (min)", "min")
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
            self.add_cons_row(c)

    # ---- collect edited state ----
    def _collect_params(self):
        spec = self.evaluator.spec()["params"]
        enabled, fixed = [], {}
        for r, p in enumerate(spec):
            lo = parse_num(self.tbl_params.item(r, 4).text())
            hi = parse_num(self.tbl_params.item(r, 5).text())
            q = dict(p)
            if lo is not None:
                q["lo"] = lo
            if hi is not None:
                q["hi"] = hi
            if self.tbl_params.item(r, 0).checkState() == Qt.Checked:
                enabled.append(q)
            elif p.get("nominal") is not None:
                fixed[p["name"]] = p["nominal"]
        return enabled, fixed

    def _collect_objective(self):
        terms = []
        for r in range(self.tbl_obj.rowCount()):
            metric = self.tbl_obj.cellWidget(r, 0).currentText()
            goal = self.tbl_obj.cellWidget(r, 1).currentData()
            tgt = parse_num(self.tbl_obj.item(r, 2).text()) or 0.0
            tol_txt = self.tbl_obj.item(r, 3).text().strip()
            w = parse_num(self.tbl_obj.item(r, 4).text()) or 1.0
            t = {"metric": metric, "goal": goal, "weight": w}
            if goal == "target":
                t["target"] = tgt
                if tol_txt:
                    t["tol"] = parse_num(tol_txt)
            terms.append(t)
        return terms

    def _collect_constraints(self):
        out = []
        for r in range(self.tbl_cons.rowCount()):
            metric = self.tbl_cons.cellWidget(r, 0).currentText()
            kind = self.tbl_cons.cellWidget(r, 1).currentData()
            v = parse_num(self.tbl_cons.item(r, 2).text())
            if v is None:
                continue
            out.append({"metric": metric, kind: v})
        return out

    def save_cfg(self):
        if not self.cfg:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存配置副本", self.cfg_path or "", "YAML (*.yaml)")
        if not path:
            return
        cfg = dict(self.cfg)
        enabled, _ = self._collect_params()
        keep = {p["name"]: p for p in enabled}
        newp = []
        for p in cfg["params"]:
            if p["name"] in keep:
                q = dict(p)
                q["lo"] = keep[p["name"]]["lo"]
                q["hi"] = keep[p["name"]]["hi"]
                newp.append(q)
        cfg["params"] = newp
        obj = self._collect_objective()
        cfg["objective"] = obj[0] if len(obj) == 1 else obj
        cfg["constraints"] = self._collect_constraints()
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        self.log("配置已保存: %s（注意：原文件注释不保留）" % path)

    # ---------------- run ----------------
    def test_remote(self):
        url = self.edit_remote.text().strip()
        if not url:
            self.log("远程GPU: 未填写 URL")
            return
        try:
            from .proposer import RemoteProposer
            h = RemoteProposer(url).health()
            self.log("远程GPU可用: %s (torch %s)"
                     % (h.get("device_name"), h.get("torch")))
        except Exception as e:
            self.log("远程GPU不可达: %s" % e)

    def run_nominal(self):
        if not self.evaluator:
            return
        _, fixed = self._collect_params()
        self.nw = NominalWorker(self.evaluator, fixed)
        self.nw.sig_log.connect(self.log)
        self.nw.start()

    def start_run(self):
        if not self.evaluator:
            QMessageBox.information(self, "提示", "先打开电路配置 YAML")
            return
        if self.worker and self.worker.isRunning():
            return
        try:
            enabled, fixed = self._collect_params()
            objective = self._collect_objective()
            constraints = self._collect_constraints()
            if not enabled:
                raise ValueError("至少启用一个参数")
            if not objective:
                raise ValueError("至少一个目标项")
        except Exception as e:
            QMessageBox.critical(self, "配置错误", str(e))
            return
        self.evaluator.max_jobs = self.spin_jobs.value()
        outdir = os.path.join(self.evaluator.root, "gui_run")
        from .engine import Engine

        def make_engine(on_log, on_record):
            return Engine(
                self.evaluator, outdir, objective=objective,
                constraints=constraints,
                batch=self.spin_batch.value(),
                n_init=self.spin_init.value(),
                device=self.cmb_device.currentText(),
                use_dkl=self.chk_dkl.isChecked(),
                dkl_after=self.spin_dkl_after.value(),
                seed=self.spin_seed.value(),
                on_log=on_log, on_record=on_record,
                spec_params=enabled, fixed=fixed,
                remote_gpu=self.edit_remote.text().strip() or None)

        self.plot_data = []
        self._headline = objective[0]["metric"]
        self.worker = RunWorker(make_engine, self.spin_budget.value(),
                                self.chk_resume.isChecked())
        self.worker.sig_log.connect(self.log)
        self.worker.sig_record.connect(self.on_record)
        self.worker.sig_done.connect(self.on_done)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()

    def stop_run(self):
        if self.worker:
            self.worker.request_stop()
            self.log("停止请求已发出（本轮仿真结束后停）…")

    def on_record(self, payload):
        rec, st = payload
        ok = bool(rec.get("ok")) and rec.get("metrics")
        head = None
        feas = False
        if ok:
            head = rec["metrics"].get(self._headline)
            bi = st.get("best_index")
            feas = (st["n_feasible"] > 0 and bi is not None
                    and st["best_record"] is rec)
        self.plot_data.append((len(self.plot_data), head, ok))
        self._redraw()
        if st.get("best_record"):
            self._show_best(st["best_record"], st)

    def _redraw(self):
        self.ax.clear()
        xs = [i for i, h, ok in self.plot_data if ok and h is not None]
        ys = [h for _, h, ok in self.plot_data if ok and h is not None]
        xf = [i for i, h, ok in self.plot_data if not ok]
        self.ax.plot(xs, ys, "o", ms=3, color="tab:blue",
                     label="评估点")
        if xf:
            self.ax.plot(xf, [min(ys) if ys else 0] * len(xf), "x",
                         ms=4, color="red", label="仿真失败")
        self.ax.set_xlabel("trial")
        self.ax.set_ylabel(getattr(self, "_headline", ""))
        self.ax.grid(alpha=0.3)
        if xs:
            self.ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def _show_best(self, rec, st):
        t = self.tbl_best
        rows = []
        rows.append(("最优 trial", str(rec.get("trial"))))
        rows.append(("可行", "是" if st["n_feasible"] > 0 else "否"))
        for k, v in (rec.get("metrics") or {}).items():
            rows.append(("指标 " + k, "%.6g" % v))
        for k, v in (rec.get("params") or {}).items():
            rows.append(("参数 " + k, fmt_num(v)))
        t.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            t.setItem(i, 0, _ro_item(a))
            t.setItem(i, 1, _ro_item(b))

    def on_done(self, summary):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if summary:
            self.log("运行结束: n=%d 可行=%d（best.json 已写入）"
                     % (summary["n"], summary["n_feasible"]))

    # ---------------- writeback ----------------
    def writeback(self):
        if not self.evaluator:
            return
        from . import writeback as wb
        best_path = os.path.join(self.evaluator.root, "gui_run",
                                 "best.json")
        if not os.path.exists(best_path):
            QMessageBox.information(self, "提示",
                                    "还没有 best.json（先跑一次优化）")
            return
        with open(best_path) as f:
            best = json.load(f)
        if not self.cfg.get("virtuoso", {}).get("lib"):
            QMessageBox.warning(
                self, "缺少配置",
                "YAML 需要 virtuoso: {lib: <库名>, top_cell: <顶层cell>}")
            return
        if not wb.bridge_alive():
            QMessageBox.warning(self, "bridge 未运行",
                                "virtuoso-bridge (127.0.0.1:65036) 未响应")
            return
        edits, skipped = wb.plan(self.cfg, self.evaluator.spec()["params"],
                                 best["params"])
        ok, cur = wb.read_current(self.cfg, edits)
        lines = ["将写入 %d 个属性（lib=%s）:" %
                 (len(edits), self.cfg["virtuoso"]["lib"]), ""]
        lines += ["  %s/%s %s = %s   (参数 %s)" % e for e in edits]
        if skipped:
            lines += ["", "跳过（需在 YAML 加 sch_prop）:"] + \
                     ["  %s: %s" % s for s in skipped]
        lines += ["", "当前原理图值: %s" % (cur if ok else "读取失败")]
        dlg = QDialog(self)
        dlg.setWindowTitle("回写预览")
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
        self.log(("回写完成: " if ok else "回写失败: ") + msg)
        if ok:
            QMessageBox.information(
                self, "完成",
                "已写回并保存原理图。\n注意重新 netlist 后才会反映在仿真里。")


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
