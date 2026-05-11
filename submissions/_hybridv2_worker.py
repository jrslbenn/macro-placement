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
                        choices=["ibm", "spectral", "perturbed", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--enable-plots", type=int, default=0)
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from submissions.hybridv2 import HybridAnalyticalPlacerV2

    print(
        f"[worker] bench={args.bench_name} strategy={args.init_strategy} "
        f"seed={args.seed} num_steps={args.num_steps}",
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
    )

    placement = placer.place(benchmark)
    metrics = compute_proxy_cost(placement, benchmark, plc)

    out = {
        "placement": placement.detach().cpu(),
        "proxy": float(metrics["proxy_cost"]),
        "wl": float(metrics["wirelength_cost"]),
        "den": float(metrics["density_cost"]),
        "cong": float(metrics["congestion_cost"]),
        "overlap_count": int(metrics["overlap_count"]),
        "strategy": args.init_strategy,
        "seed": args.seed,
    }
    with open(args.out_path, "wb") as f:
        pickle.dump(out, f)

    print(
        f"[worker] done: proxy={out['proxy']:.6f} "
        f"wl={out['wl']:.6f} den={out['den']:.6f} cong={out['cong']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
