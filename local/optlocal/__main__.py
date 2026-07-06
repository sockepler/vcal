"""CLI:
  python -m optlocal gui
  python -m optlocal run <circuit.yaml> [--budget N] [--batch q] [--init N]
        [--device auto|cuda|cpu] [--objective '<yaml/json list>']
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optlocal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gui", help="启动图形界面")
    rp = sub.add_parser("run", help="命令行跑一次优化")
    rp.add_argument("circuit_yaml")
    rp.add_argument("--budget", type=int, default=100)
    rp.add_argument("--batch", type=int, default=4)
    rp.add_argument("--init", type=int, default=24)
    rp.add_argument("--device", default="auto")
    rp.add_argument("--max-jobs", type=int, default=4)
    rp.add_argument("--no-dkl", action="store_true")
    rp.add_argument("--no-resume", action="store_true")
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--objective", default=None,
                    help="覆盖 YAML 目标，YAML/JSON 格式")
    rp.add_argument("--remote-gpu", default=None,
                    help="远程 GPU 计算服务 URL，如 http://your-gpu-host:8494")
    rp.add_argument("--workroot", default=None,
                    help="独立工作目录（默认 server/work/<circuit>）")
    rp.add_argument("--dkl-after", type=int, default=48)
    a = ap.parse_args(argv)

    if a.cmd == "gui":
        from .gui import main as gui_main
        gui_main()
        return 0

    import os

    import yaml

    from .engine import Engine
    from .evaluator import LocalEvaluator
    ev = LocalEvaluator(a.circuit_yaml, max_jobs=a.max_jobs,
                        workroot=a.workroot)
    obj = yaml.safe_load(a.objective) if a.objective else None
    eng = Engine(ev, os.path.join(ev.root, "cli_run"), objective=obj,
                 batch=a.batch, n_init=a.init, device=a.device,
                 use_dkl=not a.no_dkl, dkl_after=a.dkl_after,
                 seed=a.seed, remote_gpu=a.remote_gpu)
    if not a.no_resume:
        eng.resume()
    bi, rec = eng.run(a.budget)
    if rec:
        print("\nbest trial %s  feasible=%s" %
              (rec.get("trial"), eng.feasible_mask()[bi]))
        for k, v in rec["metrics"].items():
            print("  %-14s %.6g" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
