"""
Single-init worker for HybridAnalyticalPlacerV2Multi.

Loaded by subprocess via the multi-init wrapper. Loads a benchmark via
its directory, runs HybridAnalyticalPlacerV2 with a specified
init_strategy + seed, and dumps the resulting placement + real-proxy
metrics to a pickle file.

Not called directly by the eval harness.
"""

import argparse
import os
import pickle
import sys

# Repo root on sys.path so we can import macro_place + submissions.
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--init-strategy", required=True,
                        choices=["provided", "initial", "spectral", "perturbed", "halton", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-perturb-sigma", type=float, default=0.05)
    parser.add_argument("--init-spectral-blend", type=float, default=0.70)
    parser.add_argument("--init-spectral-flip-x", type=int, default=0)
    parser.add_argument("--init-spectral-flip-y", type=int, default=0)
    parser.add_argument("--init-jitter-sigma", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--enable-plots", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--vis-dir", default="")
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()

    thread_s = str(max(1, args.threads))
    os.environ.setdefault("OMP_NUM_THREADS", thread_s)
    os.environ.setdefault("MKL_NUM_THREADS", thread_s)
    os.environ.setdefault("NUMBA_NUM_THREADS", thread_s)
    if args.log_dir:
        os.environ["HAP_LOG_DIR"] = args.log_dir
    if args.vis_dir:
        os.environ["HAP_VIS_DIR"] = args.vis_dir
    os.environ.setdefault("HAP_TOTAL_TIME_BUDGET", "2850")
    os.environ.setdefault("HAP_HARD_TIME_BUDGET", "2850")

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from submissions.hybridv2 import HybridAnalyticalPlacerV2

    print(
        f"[worker] bench={args.bench_name} strategy={args.init_strategy} "
        f"seed={args.seed} num_steps={args.num_steps} threads={thread_s}",
        flush=True,
    )

    benchmark, plc = load_benchmark_from_dir(args.bench_dir)
    print(f"[worker]   {benchmark}", flush=True)

    placer = HybridAnalyticalPlacerV2(
        seed=args.seed,
        num_steps=args.num_steps,
        verbose=True,
        enable_plots=bool(args.enable_plots),
        init_strategy=args.init_strategy,
        init_perturb_sigma=args.init_perturb_sigma,
        init_spectral_blend=args.init_spectral_blend,
        init_spectral_flip_x=bool(args.init_spectral_flip_x),
        init_spectral_flip_y=bool(args.init_spectral_flip_y),
        init_jitter_sigma=args.init_jitter_sigma,
    )

    # Tell the placer where to write its early pkl. If it succeeds, we
    # reuse those metrics; if it doesn't (e.g., write failed), we fall
    # back to computing here.
    placer._out_path = args.out_path

    placement = placer.place(benchmark)

    out = None
    if os.path.exists(args.out_path):
        try:
            with open(args.out_path, "rb") as f:
                out = pickle.load(f)
        except Exception:
            out = None

    if out is None:
        # Placer didn't write a pkl — compute proxy ourselves as fallback.
        metrics = compute_proxy_cost(placement, benchmark, plc)
        out = {
            "placement": placement.detach().cpu(),
            "proxy": float(metrics["proxy_cost"]),
            "wl": float(metrics["wirelength_cost"]),
            "den": float(metrics["density_cost"]),
            "cong": float(metrics["congestion_cost"]),
            "overlap_count": int(metrics["overlap_count"]),
        }

    # Add init metadata (placer doesn't know these), then re-pickle.
    out.update({
        "strategy": args.init_strategy,
        "seed": args.seed,
        "init_perturb_sigma": args.init_perturb_sigma,
        "init_spectral_blend": args.init_spectral_blend,
        "init_spectral_flip_x": bool(args.init_spectral_flip_x),
        "init_spectral_flip_y": bool(args.init_spectral_flip_y),
        "init_jitter_sigma": args.init_jitter_sigma,
    })
    tmp = args.out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, args.out_path)

    print(
        f"[worker] done: proxy={out['proxy']:.6f} "
        f"wl={out['wl']:.6f} den={out['den']:.6f} cong={out['cong']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
