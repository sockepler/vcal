"""CLI.

  python -m optclient run  --server http://your-linux-host:8492 \
      --budget 200 --batch 4 --init 32 --out runs/amp1 [--resume]
  python -m optclient plot --server ... --run runs/amp1
  python -m optclient best --run runs/amp1
"""
import argparse
import sys

from .remote import SimServer


def main(argv=None):
    ap = argparse.ArgumentParser(prog="optclient")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("run", help="run Bayesian optimization")
    rp.add_argument("--server", required=True)
    rp.add_argument("--out", required=True, help="run directory")
    rp.add_argument("--budget", type=int, default=200,
                    help="total evaluations")
    rp.add_argument("--batch", type=int, default=4)
    rp.add_argument("--init", type=int, default=24,
                    help="Sobol initialization points")
    rp.add_argument("--resume", action="store_true",
                    help="warm-start from the server's history")
    rp.add_argument("--no-dkl", action="store_true",
                    help="plain GP only")
    rp.add_argument("--dkl-after", type=int, default=48,
                    help="switch to deep-kernel GP after N points")
    rp.add_argument("--device", default=None,
                    help="cuda / cpu (default: auto)")
    rp.add_argument("--seed", type=int, default=0)

    pp = sub.add_parser("plot", help="write convergence.png")
    pp.add_argument("--server", required=True)
    pp.add_argument("--run", required=True)

    bp = sub.add_parser("best", help="print best point")
    bp.add_argument("--run", required=True)

    gp = sub.add_parser("serve-gpu",
                        help="expose this machine's GPU for remote "
                             "surrogate-model computation")
    gp.add_argument("--port", type=int, default=8494)
    gp.add_argument("--host", default="0.0.0.0")

    a = ap.parse_args(argv)

    if a.cmd == "serve-gpu":
        from .gpuserver import main as serve
        serve(port=a.port, host=a.host)
        return 0

    if a.cmd == "best":
        from .report import best_table
        print(best_table(a.run))
        return 0

    server = SimServer(a.server)
    h = server.health()
    print("server ok: circuit=%s done=%d" % (h["circuit"], h["done"]))

    if a.cmd == "plot":
        from .report import convergence_plot
        print(convergence_plot(a.run, server.spec()))
        return 0

    from .optimize import Optimizer
    opt = Optimizer(server, a.out, batch=a.batch, n_init=a.init,
                    device=a.device, use_dkl=not a.no_dkl,
                    dkl_after=a.dkl_after, seed=a.seed)
    if a.resume:
        n = opt.resume_from_server()
        print("resumed %d evaluations from server history" % n)
    opt.run(a.budget)
    from .report import best_table, convergence_plot
    try:
        print(convergence_plot(a.out, opt.spec))
    except Exception as e:
        print("plot failed:", e)
    print(best_table(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
