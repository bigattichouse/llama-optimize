#!/usr/bin/env python3
"""Time ONE llama-server config at several context depths, through
llama-optimize's own server path, and say how it stands against the sweep's
time limits.

Answers "will a sweep be able to measure the config I already run?" before
paying for a sweep: how long the model takes to load (cold and cached), how
long each depth takes, and whether it fits --timeout. Any model: the depths
default to a doubling grid up to the model's own native context (past the
sweep's 64k grid cap — long context is what this is for)
(docs/load-and-stream-design.md, Origin, for why this exists).

    scripts/time-config.py MODEL.gguf --llama-cpp LLAMA_CPP/build/bin \\
        --factor ngl=99 --factor kv_type=q8_0 --factor ubatch=256

Factor names and levels are the sweep's (see `llama-optimize.py --help`).
While it runs: `watch -n1 cat <out>.live` and `tail -f <out>.server.log`.
"""
import argparse
import contextlib
import importlib.util
import io
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "llama_optimize", REPO / "llama-optimize.py")
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["llama-optimize"]   # it reads argv on import
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("--llama-cpp", type=Path, required=True,
                    help="llama.cpp build/bin directory")
    ap.add_argument("--depths", default=None,
                    help="comma-separated context depths (default: a doubling "
                         "grid up to the model's native context)")
    ap.add_argument("--levels", type=int, default=4,
                    help="how many default depths (default %(default)s)")
    ap.add_argument("--factor", action="append", default=[], metavar="NAME=VAL",
                    help="one sweep factor at one level; repeatable")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-config limit to judge against (default 1200)")
    ap.add_argument("--server-start-timeout", type=int, default=180)
    ap.add_argument("--out", type=Path, default=Path("time-config"),
                    help="prefix for the .live and .server.log files")
    args = ap.parse_args()

    m = load_tool()
    row = dict(f.split("=", 1) for f in args.factor)
    unknown = [k for k in row if k not in m.FACTORS]
    if unknown:
        ap.error(f"unknown factor(s): {', '.join(unknown)}")

    hw = m.model_hw(m.read_gguf_metadata(args.model))
    vram = m.detect_vram_mib(args.llama_cpp / "llama-server")
    hw.update(phys=m.detect_physical_cores(), logical=m.detect_logical_cores(),
              vram=vram[0] if vram else None)
    cfg = m.Config(model=args.model.resolve(),
                   llama_bench=args.llama_cpp / "llama-bench",
                   llama_server=args.llama_cpp / "llama-server",
                   array="auto", ctx_floor=0, driver="server", hw=hw,
                   server_start_timeout=args.server_start_timeout)
    m.LIVE.path = Path(str(args.out) + ".live")
    m.SERVER_LOG_PATH = Path(str(args.out) + ".server.log")
    print(f"model {args.model.name}: native ctx {hw.get('n_ctx_train')}, "
          f"reps {cfg.reps}, n_gen {cfg.n_gen}; judged against --timeout "
          f"{args.timeout}s (prefill may run to "
          f"{m.LOAD_HARD_CAP_FACTOR:g}x while it advances)")
    print(f"watch: watch -n1 cat {m.LIVE.path}  |  tail -f {m.SERVER_LOG_PATH}",
          flush=True)

    # Never past what the model can hold: the request adds the prompt and the
    # generation on top of the depth, and llama-server caps a slot at the
    # native context (C5) — the same ceiling the sweep uses.
    ceiling = m.depth_ceiling(hw.get("n_ctx_train"), m.request_reserve(cfg))
    if args.depths:
        depths = [int(d) for d in args.depths.split(",") if d.strip()]
    else:
        # the model's whole range, not the sweep's 64k grid cap: long context
        # is exactly what this script is for
        depths = m.log_levels_span(0, ceiling, args.levels, 1024)
    over = [d for d in depths if d > ceiling]
    if over:
        print(f"note: {', '.join(map(str, over))} exceed what this model holds "
              f"with the request on top; measuring {ceiling} instead")
        depths = sorted({min(d, ceiling) for d in depths})

    for depth in depths:
        f = {**row, "n_depth": str(depth)}
        cfg.factors = {k: [v] for k, v in f.items()}
        n_ctx = max(cfg.n_prompt + depth + cfg.n_gen + 256,
                    int(f.get("ubatch", 0) or 0))
        with contextlib.redirect_stdout(io.StringIO()):
            s = m.ServerSession(cfg, f, n_ctx, args.timeout)
        if not s.ok:
            print(f"depth {depth:>7}: did not load — "
                  f"{'; '.join(m.load_failure_lines(s.err))}", flush=True)
            continue
        load = m.load_report(s).strip()
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            r = m.measure_in_session(cfg, f, s, args.timeout)
        took = time.time() - t0
        s.close()
        fits = "within" if took <= args.timeout else "PAST"
        print(f"depth {depth:>7}: {r['status']:<9} tg {r['tg_tps']:6.1f} t/s  "
              f"pp {r['pp_tps']:7.1f} t/s  config {m.fmt_dur(took):>7} "
              f"({fits} --timeout)  {load}", flush=True)
        for ln in buf.getvalue().splitlines():
            if any(k in ln for k in ("failed", "adjusted", "measurement",
                                     "did not load")):
                print(f"           {ln.strip()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
