"""Circuit definition: YAML load, parameter application, one evaluation.

A circuit YAML declares:
  netlist    — prepared DUT netlist (control statements already stripped,
               e.g. your prepared DUT netlist (control statements stripped))
  stimuli    — stimulus include file
  tran       — {stop, maxstep}
  spectre    — {engine, mt, timeout_s}
  save       — list of node voltages / source currents to save
  params     — optimization variables, each mapping to one or more devices
  metrics    — named expressions over saved waveforms (see metrics.py)
  objective  — {metric, goal: maximize|minimize}
  constraints— [{metric, max|min}]
"""
import copy
import glob
import math
import os
import shutil
import subprocess
import time

import yaml

from . import metrics as metrics_mod
from . import psfread
from . import tr0read
from .hspice_netlist import HspiceNetlist
from .netlist import Netlist, parse_num
from .simulator_env import simulation_environment
from .validation import number, resolve_file, validate_config

SPECTRE = os.environ.get("OPTSERVER_SPECTRE", "spectre")
HSPICE = os.environ.get("OPTSERVER_HSPICE", "hspice")


class Circuit:
    def __init__(self, yaml_path):
        self.yaml_path = os.path.abspath(os.path.expanduser(os.fspath(yaml_path)))
        with open(self.yaml_path) as f:
            self.cfg = yaml.safe_load(f)
        validate_config(self.cfg)
        for key in ("netlist", "stimuli"):
            if self.cfg.get(key):
                self.cfg[key] = resolve_file(self.cfg[key], self.yaml_path, key)
        self.name = self.cfg["name"]
        self.simulator = self.cfg.get("simulator", "spectre")
        if self.simulator == "hspice" and self.cfg.get("dc") is not None:
            raise ValueError("HSpice DC analysis is not supported")
        self.params = self.cfg["params"]          # list of dicts
        for p in self.params:
            if "devices" not in p:
                raise ValueError("param %s: missing devices" % p.get("name"))
        self.metrics = self.cfg["metrics"]
        self._pdk = self._load_pdk()
        base = self._load_netlist()               # validate at load time
        self._nominal = self._read_nominal(base)
        self._validate_devices(base)
        self.fixed = dict(self.cfg.get("fixed", {}))
        for p in self.params:
            if not p.get("enabled", True) and p["name"] not in self.fixed:
                self.fixed[p["name"]] = self._nominal[p["name"]]
        known = {p["name"]: p for p in self.params}
        for name, value in self.fixed.items():
            if name not in known:
                raise ValueError("fixed references unknown parameter: " + name)
            p = known[name]
            v = number(value, "fixed." + name)
            if not parse_num(p["lo"]) <= v <= parse_num(p["hi"]):
                raise ValueError("fixed.%s is outside parameter bounds" % name)
            if p.get("integer") and not v.is_integer():
                raise ValueError("fixed.%s must be an integer" % name)
            self.fixed[name] = v
        for point in self.cfg.get("initial_points", []):
            for name, value in point.items():
                if name in self.fixed and not math.isclose(value, self.fixed[name], rel_tol=1e-9, abs_tol=0):
                    raise ValueError("initial_points conflicts with fixed parameter: " + name)
        # Verify model paths and every selected process corner before simulation.
        for corner in self.cfg.get("corners") or [{}]:
            for lib in corner.get("libs") or self._libs_for(corner.get("pdk_corner")):
                lib["path"] = resolve_file(lib["path"], self.yaml_path, "model library")

    def _validate_devices(self, nl):
        errors = []
        for p in self.params:
            for devspec in p["devices"]:
                scope, dev = self._split_dev(devspec)
                if nl.instance(scope, dev) is None:
                    errors.append("param %s: device %s not found" % (p["name"], devspec))
        if errors:
            raise ValueError("\n".join(errors))

    # ---------- PDK (可更换 / 与工具解耦) ----------
    def _load_pdk(self):
        """cfg['pdk'] 指向 pdks/<name>.yaml，定义模型库路径(可含 ${ENV})
        和各工艺角的 lib section。这样电路配置不写死具体 PDK 路径 —— 换
        PDK 只改一行 pdk: 字段或换 pdks/<name>.yaml。"""
        name = self.cfg.get("pdk")
        if not name:
            return None
        search = [os.environ.get("VCAL_PDK_DIR"),
                  os.path.join(os.path.dirname(os.path.dirname(
                      os.path.dirname(os.path.abspath(__file__)))),
                      "pdks"),
                  os.path.join(os.path.dirname(os.path.abspath(
                      self.yaml_path)), "pdks")]
        for d in search:
            if d and os.path.exists(os.path.join(d, name + ".yaml")):
                with open(os.path.join(d, name + ".yaml")) as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError(
            "PDK 定义 %s.yaml 未找到 (设 VCAL_PDK_DIR 或放 pdks/)" % name)

    def _libs_for(self, pdk_corner=None):
        """展开某工艺角的 lib 包含列表。优先用 PDK 定义，否则回退
        cfg['libs'](直接写死路径,不推荐入库)。"""
        if self._pdk:
            lib = os.path.expandvars(self._pdk["model_lib"])
            corner = pdk_corner or self._pdk.get("default_corner", "tt")
            if corner not in self._pdk.get("corners", {}):
                raise ValueError("unknown PDK corner: %s" % corner)
            secs = self._pdk["corners"][corner]
            if not isinstance(secs, list) or not secs or not all(isinstance(s, str) and s for s in secs):
                raise ValueError("PDK corner sections must be a nonempty list: %s" % corner)
            return [{"path": resolve_file(lib, self.yaml_path, "PDK model library"),
                     "section": s} for s in secs]
        return self.cfg.get("libs", [])

    def _load_netlist(self):
        cls = HspiceNetlist if self.simulator == "hspice" else Netlist
        return cls(self.cfg["netlist"])

    def _read_nominal(self, nl):
        nom = {}
        for p in self.params:
            scope, dev = self._split_dev(p["devices"][0])
            inst = nl.instance(scope, dev)
            if inst is None:
                raise ValueError("param %s: device %s not found"
                                 % (p["name"], p["devices"][0]))
            v = inst.num(p["attr"])
            nom[p["name"]] = v
        return nom

    @staticmethod
    def _split_dev(spec):
        """'ampGA/M2' -> ('ampGA','M2');  'M5' or 'top/M5' -> (None,'M5')."""
        if "/" in spec:
            scope, dev = spec.split("/", 1)
            return (None if scope == "top" else scope), dev
        return None, spec

    def _analysis_config(self):
        """Return configured analysis blocks used to identify measurements."""
        return {name: copy.deepcopy(self.cfg[name])
                for name in ("tran", "dc")
                if name in self.cfg and self.cfg[name] is not None}

    def _configured_analyses(self):
        return [name for name in ("tran", "dc")
                if name in self.cfg and self.cfg[name] is not None]

    def _default_analysis(self):
        return "tran" if "tran" in self._analysis_config() else "dc"

    def _requested_analyses(self):
        """Resolve metric definitions to the analyses they actually need."""
        configured = set(self._configured_analyses())
        if not configured:
            raise ValueError("at least one analysis (tran or dc) is required")
        default = self._default_analysis()
        requested = set()
        for definition in self.metrics.values():
            analysis = (definition.get("analysis", default)
                        if isinstance(definition, dict) else default)
            if analysis not in ("tran", "dc"):
                raise ValueError("unknown metric analysis: %s" % analysis)
            if analysis not in configured:
                raise ValueError("metric requests unavailable analysis: %s" % analysis)
            requested.add(analysis)
        return [name for name in ("tran", "dc") if name in requested]

    def _measurement_signature(self):
        """Compute the current measurement identity when the metric module has it."""
        signature = getattr(metrics_mod, "measurement_signature", None)
        if signature is None:
            return None
        return signature(self.metrics, self._analysis_config(),
                         save=self.cfg.get("save", ()))

    def spec(self):
        """What the optimizer client needs."""
        spec = {
            "circuit": self.name,
            "params": [
                {"name": p["name"],
                 "lo": float(parse_num(p["lo"])),
                 "hi": float(parse_num(p["hi"])),
                 "log": bool(p.get("log", False)),
                 "integer": bool(p.get("integer", False)),
                 "enabled": bool(p.get("enabled", True)) and p["name"] not in self.fixed,
                 "nominal": self._nominal.get(p["name"]),
                 "attr": p["attr"],
                 "devices": p["devices"]}
                for p in self.params
            ],
            "metrics": list(self.metrics.keys()),
            "objective": self.cfg["objective"],
            "constraints": self.cfg.get("constraints", []),
            "fixed": self.fixed,
            "metric_definitions": copy.deepcopy(self.metrics),
            "analyses": self._configured_analyses(),
        }
        signature = self._measurement_signature()
        if signature is not None:
            spec["measurement_signature"] = signature
        for key in ("initial_points", "optimizer"):
            if key in self.cfg:
                spec[key] = copy.deepcopy(self.cfg[key])
        return spec

    # ---------- evaluation ----------
    def apply_params(self, values, out_netlist):
        """values: {param_name: float}. Writes patched netlist."""
        nl = self._load_netlist()
        for p in self.params:
            name = p["name"]
            if name not in values:
                continue
            v = number(values[name], "param " + name)
            v = min(max(v, parse_num(p["lo"])), parse_num(p["hi"]))
            if p.get("integer"):
                v = min(max(int(round(v)), math.ceil(parse_num(p["lo"]))),
                        math.floor(parse_num(p["hi"])))
            for devspec in p["devices"]:
                scope, dev = self._split_dev(devspec)
                inst = nl.instance(scope, dev)   # re-fetch: edits reparse
                if inst is None:
                    raise ValueError("device %s not found" % devspec)
                nl.set_attr(inst, p["attr"], v,
                            integer=bool(p.get("integer")))
        nl.save(out_netlist)

    def _copy_relative_includes(self, workdir):
        """dut.scs may include files by relative path (e.g. ade_e.scs);
        copy them from the source netlist directory."""
        import re
        srcdir = os.path.dirname(os.path.abspath(self.cfg["netlist"]))
        with open(self.cfg["netlist"], errors="ignore") as f:
            for m in re.finditer(r'include\s+[\'"]([^\'"/][^\'"]*)[\'"]',
                                 f.read()):
                src = os.path.join(srcdir, m.group(1))
                dst = os.path.join(workdir, m.group(1))
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)

    def write_tb(self, workdir, corner=None, tbname="tb"):
        requested = self._requested_analyses()
        tr = self.cfg.get("tran")
        dc = self.cfg.get("dc")
        corner = corner or {}
        temp = corner.get("temp")
        # 库来源优先级: corner 显式 libs > PDK(按 corner 工艺角) > cfg.libs
        libs = corner.get("libs") or self._libs_for(corner.get("pdk_corner"))
        tb_extra = list(self.cfg.get("tb_extra", [])) + \
            list(corner.get("tb_extra", []))
        if self.simulator == "hspice":
            if "tran" not in requested or tr is None:
                raise ValueError("HSpice requires a transient analysis")
            tb = ["* %s [%s] — generated by optserver"
                  % (self.name, corner.get("name", "nominal"))]
            if self.cfg.get("options"):        # 全片 deck 用自己的 options
                for o in self.cfg["options"]:
                    tb.append(".options " + o)
            else:                              # 块级默认
                tb += [".options ingold=2 measform=1 redefsub=1"
                       " redefmodel=1",
                       ".options post=2 probe nopage nomod",
                       ".options method=gear maxord=2 converge=1"
                       " gmindc=1e-9 dcstep=10"]
            if temp is not None:
                tb.append(".temp %g" % temp)
            for lib in libs:
                tb.append(".lib '%s' %s" % (lib["path"], lib["section"]))
            if self.cfg.get("stimuli"):
                tb.append(".include '%s'" % self.cfg["stimuli"])
            tb.append(".include 'dut.sp'")
            tb.extend(tb_extra)
            tb.append(".probe tran " + " ".join(self.cfg["save"]))
            tb.append(".tran %s %s%s" % (
                tr["maxstep"], tr["stop"],
                " " + tr["args"] if tr.get("args") else ""))
            tb.append(".end")
            with open(os.path.join(workdir, tbname + ".sp"), "w") as f:
                f.write("\n".join(tb) + "\n")
        else:
            tb = ["// %s [%s] — generated by optserver"
                  % (self.name, corner.get("name", "nominal")),
                  "simulator lang=spectre"]
            if temp is not None:
                tb.append("tempOpt options temp=%g" % temp)
            for lib in libs:
                section = " section=" + lib["section"] if lib.get("section") else ""
                tb.append('include "%s"%s' % (lib["path"], section))
            tb += ['include "%s"' % self.cfg["stimuli"],
                   'include "dut.scs"']
            tb += tb_extra
            analyses = []
            if "tran" in requested:
                if tr is None:
                    raise ValueError("transient analysis is not configured")
                analyses.append("tran1 tran stop=%s maxstep=%s" %
                                (tr["stop"], tr["maxstep"]))
            if "dc" in requested:
                if dc is None:
                    raise ValueError("DC analysis is not configured")
                analyses.append("dc1 dc dev=%s param=dc start=%s stop=%s step=%s" %
                                (dc["source"], dc["start"], dc["stop"],
                                 dc["step"]))
            tb += analyses + ["save %s" % " ".join(self.cfg["save"]),
                              "saveOptions options rawfmt=psfbin", ""]
            with open(os.path.join(workdir, tbname + ".scs"), "w") as f:
                f.write("\n".join(tb))

    def evaluate(self, values, workdir):
        """Run one point. Returns dict with ok/metrics/error/sim_s."""
        os.makedirs(workdir, exist_ok=True)
        t0 = time.time()
        values = {**self.fixed, **values}
        result = {"params": values, "ok": False, "metrics": None,
                  "error": None, "workdir": workdir}
        corners = self.cfg.get("corners") or [{"name": "nominal"}]
        multi = len(corners) > 1
        try:
            signature = self._measurement_signature()
            if signature is not None:
                result["measurement_signature"] = signature
            ext = "sp" if self.simulator == "hspice" else "scs"
            self.apply_params(values, os.path.join(workdir, "dut." + ext))
            self._copy_relative_includes(workdir)
            by_corner = {}
            for corner in corners:
                cname = corner.get("name", "nominal")
                tbn = ("tb_" + cname) if multi else "tb"
                self.write_tb(workdir, corner=corner, tbname=tbn)
                if self.simulator == "hspice":
                    dataset = self._run_hspice(workdir, tbn)
                else:
                    dataset = self._run_spectre(workdir, tbn)
                if isinstance(dataset, dict):
                    m = metrics_mod.compute_metrics(
                        self.metrics, datasets=dataset,
                        default_analysis=self._default_analysis())
                else:
                    t, sigs = dataset
                    m = metrics_mod.compute_metrics(
                        self.metrics, t, sigs,
                        default_analysis=self._default_analysis())
                bad = [k for k, v in m.items()
                       if v != v or abs(v) == float("inf")]
                if bad:
                    raise RuntimeError("non-finite metrics @%s: %s"
                                       % (cname, bad))
                by_corner[cname] = m
            # 跨角点归约：corner_worst 指定每个指标取 min/max/mean
            # (默认 min，适合 ENOB/增益这类越大越好的目标 -> 最差角点)
            worst = self.cfg.get("corner_worst", {})
            reduced = {}
            for mname in self.metrics:
                vals = [by_corner[c.get("name", "nominal")][mname]
                        for c in corners]
                mode = worst.get(mname, "min")
                reduced[mname] = (min(vals) if mode == "min" else
                                  max(vals) if mode == "max" else
                                  sum(vals) / len(vals))
            result["metrics"] = reduced
            if multi:
                result["metrics_by_corner"] = by_corner
            result["ok"] = True
        except subprocess.TimeoutExpired:
            result["error"] = self.simulator + " timeout"
        except Exception as e:
            result["error"] = str(e)
        result["sim_s"] = round(time.time() - t0, 2)
        if result["ok"] and not self.cfg.get("keep_raw", False):
            for d in glob.glob(os.path.join(workdir, "tb*.raw")):
                shutil.rmtree(d, ignore_errors=True)
            for f in glob.glob(os.path.join(workdir, "tb*.tr0")):
                os.unlink(f)
        return result

    def _run_spectre(self, workdir, tbname="tb"):
        requested = self._requested_analyses()
        configured = set(self._configured_analyses())
        scfg = self.cfg.get("spectre", {})
        engine = scfg.get("engine", "+preset=cx")
        mt = int(scfg.get("mt", 2))
        timeout = float(scfg.get("timeout_s", 600))
        cmd = [SPECTRE] + ([engine] if engine else []) + \
            [tbname + ".scs", "+log", tbname + ".log",
             "-raw=" + tbname + ".raw", "+mt=%d" % mt]
        r = subprocess.run(cmd, cwd=workdir, capture_output=True,
                           text=True, timeout=timeout,
                           env=simulation_environment(),
                           preexec_fn=_unlimit_stack)
        with open(os.path.join(workdir, tbname + ".process.log"), "w") as stream:
            stream.write((r.stdout or "") + "\n" + (r.stderr or ""))
        log = ""
        logf = os.path.join(workdir, tbname + ".log")
        if os.path.exists(logf):
            with open(logf, errors="ignore") as f:
                log = f.read()
        if "Segmentation fault" in log or "Segmentation fault" in \
                (r.stderr or ""):
            raise RuntimeError("spectre segfault")
        if r.returncode != 0 or "spectre completes with 0 errors" not in log:
            err = [ln for ln in log.splitlines() if "ERROR" in ln
                   or "Error" in ln][:3]
            raise RuntimeError("spectre failed (exit %s): %s"
                               % (r.returncode, "; ".join(err) or
                                  (r.stderr or r.stdout or "see " + tbname)[:1000]))
        rawdir = os.path.join(workdir, tbname + ".raw")

        def raw_for(analysis):
            patterns = ("*.%s.%s" % (analysis, analysis),
                        "*.%s" % analysis)
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(rawdir, "**", pattern),
                                       recursive=True))
            files = sorted({path for path in files if os.path.isfile(path)})
            if not files:
                raise RuntimeError("no %s raw output" % analysis)
            if len(files) > 1:
                raise RuntimeError("ambiguous %s raw output" % analysis)
            return files[0]

        datasets = {}
        for analysis in requested:
            rawpath = raw_for(analysis)
            if analysis == "tran" and configured == {"tran"}:
                datasets[analysis] = psfread.read_tran_psfbin(rawpath, workdir)
            else:
                datasets[analysis] = psfread.read_sweep_psfbin(
                    rawpath, workdir, name=analysis)
        if configured == {"tran"} and requested == ["tran"]:
            return datasets["tran"]
        return datasets

    def _run_hspice(self, workdir, tbname="tb"):
        scfg = self.cfg.get("hspice", {})
        mt = int(scfg.get("mt", 2))
        timeout = float(scfg.get("timeout_s", 900))
        cmd = [HSPICE, "-i", tbname + ".sp", "-o", tbname, "-mt", str(mt)]
        r = subprocess.run(cmd, cwd=workdir, capture_output=True,
                           text=True, timeout=timeout,
                           env=simulation_environment(),
                           preexec_fn=_unlimit_stack)
        lisf = os.path.join(workdir, tbname + ".lis")
        lis = ""
        if os.path.exists(lisf):
            with open(lisf, errors="ignore") as f:
                lis = f.read()
        tr0 = os.path.join(workdir, tbname + ".tr0")
        if not os.path.exists(tr0):
            err = [ln for ln in lis.splitlines() if "**error**" in ln
                   or "aborted" in ln.lower()][:3]
            raise RuntimeError("hspice failed: %s"
                               % ("; ".join(err)
                                  or (r.stderr or "no tr0")[:300]))
        if "internal timestep too small" in lis:
            raise RuntimeError("hspice non-convergence (timestep)")
        return tr0read.read_tr0(tr0)


def _unlimit_stack():
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY,
                            resource.RLIM_INFINITY))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
