"""
A/B test: hybrid_analytical_placer (baseline) vs hybridv2 with DAS-MP only.

Runs both placers on the same benchmarks with identical settings except
for DAS-MP weighting. WireMask is disabled to isolate the DAS effect.

Usage: python run_ab.py <bench_name> <which: base|das>
"""

import sys
import time
from pathlib import Path

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_ab.py <bench> <base|das>", file=sys.stderr)
        sys.exit(2)
    bench_name = sys.argv[1]
    which = sys.argv[2]
    bench_dir = f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}"

    print(f"=== A/B: bench={bench_name} variant={which} ===", flush=True)
    benchmark, plc = load_benchmark_from_dir(bench_dir)
    print(f"  {benchmark}", flush=True)

    if which == "base":
        from submissions.hybrid_analytical_placer import HybridAnalyticalPlacer
        placer = HybridAnalyticalPlacer(
            seed=42,
            num_steps=2000,
            verbose=True,
            enable_plots=False,
        )
    elif which == "das":
        from submissions.hybridv2 import HybridAnalyticalPlacerV2
        placer = HybridAnalyticalPlacerV2(
            seed=42,
            num_steps=2000,
            verbose=True,
            enable_plots=False,
            das_enable=True,
            wm_enable=False,  # isolate DAS effect
        )
    else:
        print(f"Unknown variant: {which}", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    placement = placer.place(benchmark)
    runtime = time.time() - t0

    is_valid, violations = validate_placement(placement, benchmark)
    metrics = compute_proxy_cost(placement, benchmark, plc)

    print("\n=== RESULT ===", flush=True)
    print(f"bench={bench_name} variant={which} runtime={runtime:.1f}s valid={is_valid}")
    for k in ("proxy_cost", "wirelength_cost", "density_cost", "congestion_cost", "overlap_count"):
        print(f"  {k}={float(metrics[k]):.6f}")


if __name__ == "__main__":
    main()
