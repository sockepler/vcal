"""Command line interface for the local optimizer.

  python -m optlocal gui
  python -m optlocal check <circuit.yaml> [--json]
  python -m optlocal scopes <circuit.yaml> [--json]
  python -m optlocal review <history.jsonl> --config <circuit.yaml> [--json]
  python -m optlocal gmid <LUT> --length L --vds V --gmid R (--id I | --gm G)
  python -m optlocal run <circuit.yaml> [--budget N] [--batch N] [--init N]
        [--device auto|cuda|cpu] [--objective '<yaml/json list>']
"""
import argparse
import json
import math
import os
import sys

from .i18n import configure_language, tr


_MISSING_LANGUAGE = object()


def _tr(source):
    """Translate display text while retaining a useful source fallback."""
    try:
        return tr(source)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        # A source checkout can be inspected before locale assets are copied
        # into place.  The CLI remains usable in its source language then.
        return source


def _format(source, **values):
    return _tr(source).format(**values)


class _LocalizedParser(argparse.ArgumentParser):
    """ArgumentParser with translated usage/error framing."""

    def format_usage(self):
        usage = super().format_usage()
        return usage.replace("usage:", _tr("用法:"), 1)

    def format_help(self):
        help_text = super().format_help()
        return help_text.replace("usage:", _tr("用法:"), 1)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, "%s: %s: %s\n" %
                  (self.prog, _tr("error"), message))


def _add_help(parser):
    parser.add_argument(
        "-h", "--help", action="help", help=_tr("显示帮助并退出"))
    parser._positionals.title = _tr("位置参数")
    parser._optionals.title = _tr("选项")


def _add_language(parser, *, default=argparse.SUPPRESS):
    parser.add_argument(
        "--lang", dest="lang", default=default, metavar="LANG",
        help="%s zh/ja/en" % _tr("Language:"))


def _requested_language(argv):
    """Return the last --lang value, including options after a subcommand."""
    requested = None
    i = 0
    while i < len(argv):
        value = argv[i]
        if value == "--":
            break
        if value == "--lang":
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                return _MISSING_LANGUAGE
            requested = argv[i + 1]
            i += 2
            continue
        if value.startswith("--lang="):
            requested = value.split("=", 1)[1]
        i += 1
    return requested


def _language_error(value):
    # Set a known language before formatting an invalid-language diagnostic.
    try:
        configure_language("zh")
    except (OSError, ValueError):
        pass
    if value is None:
        value = os.environ.get("VCAL_LANG", "")
    text = _format(
        "不支持的语言: {language}（可选 zh、ja、en）",
        language="" if value is _MISSING_LANGUAGE else value)
    print("optlocal: %s %s" % (_tr("错误:"), text), file=sys.stderr)
    raise SystemExit(2)


def _prepare_language(argv):
    requested = _requested_language(argv)
    if requested is _MISSING_LANGUAGE:
        _language_error(requested)
    if requested == "":
        _language_error(requested)
    try:
        # configure_language(None) applies VCAL_LANG, saved settings, then
        # Chinese.  An explicit option takes precedence over VCAL_LANG.
        configure_language(requested)
    except (OSError, ValueError, TypeError):
        _language_error(requested)


def _positive_int(text):
    """argparse type for counts that must be usable as positive values."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(_tr("must be an integer"))
    if value <= 0:
        raise argparse.ArgumentTypeError(_tr("must be a positive integer"))
    return value


def _nonnegative_int(text):
    """argparse type for counts that may be zero."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(_tr("must be an integer"))
    if value < 0:
        raise argparse.ArgumentTypeError(_tr("must be a nonnegative integer"))
    return value


def _si_number(text):
    """Parse one finite Spectre/engineering numeric command-line value."""
    try:
        from optserver.netlist import parse_num
        value = parse_num(text)
    except (ImportError, TypeError, ValueError, OverflowError):
        value = None
    if value is None or not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            _tr("must be a finite engineering number"))
    return float(value)


def _error(message):
    """Print one concise CLI error and return argparse's error status."""
    message = str(message).strip()
    if not message:
        message = _tr("unknown error")
    print("optlocal: %s: %s" % (_tr("error"), message), file=sys.stderr)
    return 2


def _parser():
    ap = _LocalizedParser(prog="optlocal", add_help=False)
    _add_help(ap)
    _add_language(ap, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True,
                            parser_class=_LocalizedParser,
                            title=_tr("命令"))

    gui = sub.add_parser("gui", help=_tr("启动图形界面"),
                         description=_tr("启动图形界面"), add_help=False)
    _add_help(gui)
    _add_language(gui)

    cp = sub.add_parser(
        "check", help=_tr("校验电路配置并打印 spec"),
        description=_tr("校验电路配置并打印 spec"), add_help=False)
    _add_help(cp)
    _add_language(cp)
    cp.add_argument("circuit_yaml")
    cp.add_argument("--json", action="store_true",
                    help=_tr("以 JSON 输出 spec"))

    sp = sub.add_parser(
        "scopes", help=_tr("inspect parameter scopes"),
        description=_tr("inspect parameter scopes"), add_help=False)
    _add_help(sp)
    _add_language(sp)
    sp.add_argument("circuit_yaml")
    sp.add_argument("--json", action="store_true",
                    help=_tr("output scope inventory as JSON"))

    vp = sub.add_parser(
        "review", help=_tr("review an optimization history"),
        description=_tr("review an optimization history"), add_help=False)
    _add_help(vp)
    _add_language(vp)
    vp.add_argument("history_jsonl")
    vp.add_argument("--config", required=True, metavar="CIRCUIT_YAML",
                     help=_tr("circuit configuration used by the history"))
    vp.add_argument("--json", action="store_true",
                    help=_tr("output review as JSON"))

    gp = sub.add_parser(
        "gmid", help=_tr("size a device from a gm/Id LUT"),
        description=_tr("size a device from a gm/Id LUT"), add_help=False)
    _add_help(gp)
    _add_language(gp)
    gp.add_argument("lut_path", metavar="LUT")
    gp.add_argument("--length", required=True, type=_si_number, metavar="L",
                    help=_tr("channel length in SI units"))
    gp.add_argument("--vds", required=True, type=_si_number, metavar="VDS",
                    help=_tr("drain-source voltage in SI units"))
    gp.add_argument("--vsb", default=0.0, type=_si_number, metavar="VSB",
                    help=_tr("source-bulk voltage in SI units (default: 0)"))
    gp.add_argument("--gmid", required=True, type=_si_number, metavar="GMID",
                    help=_tr("target gm/Id"))
    target = gp.add_mutually_exclusive_group(required=True)
    target.add_argument("--id", dest="ids", type=_si_number, metavar="ID",
                        help=_tr("target drain current in SI units"))
    target.add_argument("--gm", type=_si_number, metavar="GM",
                        help=_tr("target transconductance in SI units"))
    gp.add_argument("--json", action="store_true",
                    help=_tr("output sizing result as JSON"))

    rp = sub.add_parser(
        "run", help=_tr("命令行跑一次优化"),
        description=_tr("命令行跑一次优化"), add_help=False)
    _add_help(rp)
    _add_language(rp)
    rp.add_argument("circuit_yaml")
    rp.add_argument("--budget", type=_positive_int, default=100,
                    help=_tr("总评估数（正整数）"))
    rp.add_argument("--batch", type=_positive_int, default=4,
                    help=_tr("每轮提案数（正整数）"))
    rp.add_argument("--init", type=_positive_int, default=24,
                    help=_tr("Sobol 初始化点数（正整数）"))
    rp.add_argument("--device",
                    choices=("auto", "cuda", "rocm", "gpu", "cpu"),
                    default="auto", help=_tr("计算设备:"))
    rp.add_argument("--max-jobs", type=_positive_int, default=4,
                    help=_tr("并行仿真数（正整数）"))
    rp.add_argument("--no-dkl", action="store_true")
    resume = rp.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true",
                        help=_tr("从历史续跑（默认）"))
    resume.add_argument("--no-resume", dest="resume", action="store_false",
                        help=_tr("忽略已有历史"))
    rp.set_defaults(resume=True)
    rp.add_argument("--seed", type=int, default=0, help=_tr("随机种子:"))
    rp.add_argument("--objective", default=None,
                    help=_tr("覆盖 YAML 目标，YAML/JSON 格式"))
    rp.add_argument("--remote-gpu", default=None,
                    help=_tr("远程 GPU 计算服务 URL，如 http://your-gpu-host:8494"))
    rp.add_argument("--workroot", default=None,
                    help=_tr("独立工作目录（默认按电路配置位置）"))
    rp.add_argument("--dkl-after", type=_positive_int, default=48,
                    help=_tr("达到该评估数后切换 DKL（正整数）"))
    rp.add_argument("--scope", default=None, metavar="NAME",
                    help=_tr("optimize only parameters in this scope"))
    rp.add_argument("--stagnation-rounds", type=_nonnegative_int, default=None,
                    metavar="N", help=_tr("global exploration after N stagnant rounds"))
    rp.add_argument("--initial-points", default=None, metavar="JSON_FILE",
                    help=_tr("JSON file containing physical initial points"))
    return ap


def _emit(value, as_json):
    """Print a JSON or concise YAML representation of a result."""
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return
    import yaml
    print(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), end="")


def _gmid(a):
    try:
        from .gmid import GmIdTable
        table = GmIdTable.load(a.lut_path)
        kwargs = {"ids": a.ids} if a.ids is not None else {"gm": a.gm}
        result = table.size_for(a.length, a.vds, a.vsb, gmid=a.gmid,
                                **kwargs)
        _emit(result, a.json)
        return 0
    except Exception as exc:
        return _error(_tr("gmid failed: %s") % exc)


def _scopes(a):
    try:
        from optserver.circuit import Circuit
        from .scopes import scope_inventory
        spec = Circuit(a.circuit_yaml).spec()
        _emit(scope_inventory(spec["params"]), a.json)
        return 0
    except Exception as exc:
        return _error(_tr("scopes failed: %s") % exc)


def _review(a):
    try:
        from optserver.circuit import Circuit
        from .evaluator import read_history
        from .review import analyze_records
        spec = Circuit(a.config).spec()
        records = read_history(a.history_jsonl)
        result = analyze_records(
            records, spec["objective"], spec.get("constraints", []),
            spec.get("params", []))
        _emit(result, a.json)
        return 0
    except Exception as exc:
        return _error(_tr("review failed: %s") % exc)


def _read_initial_points(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as exc:
        raise ValueError("cannot read initial points %s: %s" % (path, exc)) from exc


def _project_initial_points(points, params, active_params, fixed_values):
    """Validate names, then project full points onto a selected scope."""
    if not isinstance(points, list):
        raise ValueError("initial_points must be a list")
    known = {p["name"] for p in params}
    active = {p["name"] for p in active_params}
    projected = []
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError("initial_points[%d] must be a mapping" % index)
        unknown = [name for name in point if name not in known]
        if unknown:
            raise ValueError("initial_points[%d] has unknown parameter: %s" %
                             (index, ", ".join(str(name) for name in unknown)))
        item = {name: value for name, value in point.items() if name in active}
        # Fixed values win over a full-point value from an unselected scope,
        # keeping that scope at its configured nominal/fixed value.
        item.update(fixed_values)
        projected.append(item)
    return projected


def _check(a):
    try:
        from optserver.circuit import Circuit
        circuit = Circuit(a.circuit_yaml)
        spec = circuit.spec()
        if a.json:
            print(json.dumps(spec, ensure_ascii=False, indent=2))
        else:
            import yaml
            print(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                  end="")
        return 0
    except Exception as exc:
        return _error(_tr("check failed: %s") % exc)


def _run(a):
    import os

    try:
        import yaml
        from .engine import Engine
        from .evaluator import LocalEvaluator
        ev = LocalEvaluator(a.circuit_yaml, max_jobs=a.max_jobs,
                            workroot=a.workroot)
        obj = yaml.safe_load(a.objective) if a.objective else None
        if obj is not None and not isinstance(obj, (dict, list)):
            return _error(_tr("--objective must be a YAML mapping or list"))
        # LocalEvaluator exposes the complete server spec.  Keep the small
        # fallback for lightweight test doubles and older evaluator adapters.
        spec_getter = getattr(ev, "spec", None)
        spec = spec_getter() if callable(spec_getter) else None
        engine_kwargs = {}
        active_params = None
        fixed_values = None
        if isinstance(spec, dict) and isinstance(spec.get("params"), list):
            from .scopes import select_scope
            active_params, fixed_values = select_scope(
                spec["params"], a.scope, fixed=spec.get("fixed", {}))
            engine_kwargs.update(spec_params=active_params, fixed=fixed_values)

        points = None
        if a.initial_points is not None:
            points = _read_initial_points(a.initial_points)
        elif isinstance(spec, dict):
            points = spec.get("initial_points", [])
        if a.scope not in (None, ""):
            if not isinstance(spec, dict) or active_params is None:
                raise ValueError("scope selection requires a complete circuit spec")
            if points is None:
                points = []
            points = _project_initial_points(
                points, spec["params"], active_params, fixed_values)
        if points is not None:
            engine_kwargs["initial_points"] = points

        if a.stagnation_rounds is not None:
            stagnation_rounds = a.stagnation_rounds
        elif isinstance(spec, dict):
            optimizer = spec.get("optimizer", {})
            optimizer = optimizer if isinstance(optimizer, dict) else {}
            stagnation_rounds = optimizer.get("stagnation_rounds", 0)
        else:
            stagnation_rounds = 0
        engine_kwargs["stagnation_rounds"] = stagnation_rounds

        eng = Engine(ev, os.path.join(ev.root, "cli_run"), objective=obj,
                     batch=a.batch, n_init=a.init, device=a.device,
                     use_dkl=not a.no_dkl, dkl_after=a.dkl_after,
                     seed=a.seed, remote_gpu=a.remote_gpu, **engine_kwargs)
    except Exception as exc:
        return _error(_tr("configuration or runtime error: %s") % exc)

    try:
        if a.resume:
            eng.resume()
        bi, rec = eng.run(a.budget)
    except KeyboardInterrupt:
        try:
            eng.save_best()
        except Exception:
            pass
        return 130
    except Exception as exc:
        return _error(_tr("run failed: %s") % exc)

    if rec:
        print(_tr("\nbest trial %s  feasible=%s") %
              (rec.get("trial"), eng.feasible_mask()[bi]))
        for k, v in rec["metrics"].items():
            print("  %-14s %.6g" % (k, v))
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    _prepare_language(args)
    a = _parser().parse_args(args)
    if a.lang is not None:
        # Keep the GUI boundary explicit.  ``None`` means the GUI may apply
        # the persisted preference when no command-line language was given.
        configure_language(a.lang)
    if a.cmd == "gui":
        from .gui import main as gui_main
        gui_main(language=a.lang)
        return 0
    if a.cmd == "check":
        return _check(a)
    if a.cmd == "scopes":
        return _scopes(a)
    if a.cmd == "review":
        return _review(a)
    if a.cmd == "gmid":
        return _gmid(a)
    return _run(a)


if __name__ == "__main__":
    sys.exit(main())
