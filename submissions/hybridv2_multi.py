"""
HybridAnalyticalPlacerV2Multi — multi-init parallel wrapper.

Spawns 4 subprocess workers per benchmark, each with a distinct
(init_strategy, seed) tuple, runs them in parallel, and picks the
placement with the best real proxy.

Rationale: stage-level tweaks on a single deterministic run plateau
because each bench has a configuration ceiling — one starting basin
yields only so much. Multi-init produces structural diversity: 4
different starting basins → 4 different end states → take the best.

Worker variants chosen for *structural* diversity, not just seed
variation. The IBM initial is usually decent, but on cong-bound benches
(ibm17/18) it's a poor basin; spectral/perturbed inits can find better
basins despite worse Nesterov starts.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import torch

from macro_place.benchmark import Benchmark


# Worker configs: (label, init_strategy, seed)
# These are the variants spawned per benchmark. Tuned for structural
# diversity rather than seed-only variation.
WORKER_VARIANTS: List[Tuple[str, str, int]] = [
    ("ibm_42", "ibm", 42),
    ("ibm_137", "ibm", 137),
    ("spectral_42", "spectral", 42),
    ("perturbed_271", "perturbed", 271),
]


class HybridAnalyticalPlacerV2Multi:
    """Wrapper that spawns N hybridv2 workers, picks best by real proxy."""

    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 50000,
        verbose: bool = True,
        enable_plots: bool = True,
        workers: List[Tuple[str, str, int]] | None = None,
    ):
        self.seed = seed
        self.num_steps = num_steps
        self.verbose = verbose
        self.enable_plots = enable_plots
        self.workers = workers or WORKER_VARIANTS

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost
        from macro_place._plc import PlacementCost  # for loader

        # Stash benchmark to a temp file so subprocesses can load it.
        tmpdir = Path(tempfile.mkdtemp(prefix=f"hv2_multi_{benchmark.name}_"))
        bench_path = tmpdir / "benchmark.pt"
        benchmark.save(str(bench_path))

        # We also need the plc path for proxy computation in workers.
        # The harness already loaded benchmark+plc, but plc isn't on the
        # Benchmark dataclass — the worker reloads via load_benchmark_from_dir
        # using a known root + benchmark name.
        # We assume IBM benchmarks; ng45 path could be added.
        repo_root = Path(__file__).resolve().parent.parent
        bench_dir = repo_root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name

        worker_script = repo_root / "submissions" / "_hybridv2_worker.py"
        if not worker_script.exists():
            raise RuntimeError(f"worker script missing: {worker_script}")

        # Launch all workers in parallel.
        procs = []
        result_paths = []
        for label, strategy, seed in self.workers:
            out_path = tmpdir / f"{label}.pkl"
            result_paths.append((label, strategy, seed, out_path))
            cmd = [
                sys.executable,
                str(worker_script),
                "--bench-dir", str(bench_dir),
                "--bench-name", benchmark.name,
                "--init-strategy", strategy,
                "--seed", str(seed),
                "--num-steps", str(self.num_steps),
                "--enable-plots", "1" if self.enable_plots else "0",
                "--out-path", str(out_path),
            ]
            log_path = tmpdir / f"{label}.log"
            log_f = open(log_path, "w")
            if self.verbose:
                print(f"[multi] launching worker '{label}': strategy={strategy} seed={seed}")
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(repo_root))
            procs.append((label, p, log_f, log_path))

        # Wait for all to complete.
        if self.verbose:
            print(f"[multi] waiting for {len(procs)} workers...", flush=True)
        start_t = time.time()
        for label, p, log_f, log_path in procs:
            p.wait()
            log_f.close()
            if self.verbose:
                print(f"[multi] worker '{label}' exited code={p.returncode} (log: {log_path})")
        if self.verbose:
            print(f"[multi] all workers done in {time.time()-start_t:.1f}s")

        # Collect results.
        results = []
        for label, strategy, seed, out_path in result_paths:
            if not out_path.exists():
                if self.verbose:
                    print(f"[multi] WARNING: '{label}' produced no output (likely crashed)")
                continue
            try:
                with open(out_path, "rb") as f:
                    data = pickle.load(f)
                results.append((label, strategy, seed, data))
                if self.verbose:
                    print(
                        f"[multi] '{label}': proxy={data['proxy']:.4f} "
                        f"wl={data['wl']:.4f} den={data['den']:.4f} cong={data['cong']:.4f}"
                    )
            except Exception as exc:
                if self.verbose:
                    print(f"[multi] failed to load '{label}': {exc}")

        if not results:
            raise RuntimeError("All workers failed; no placement produced.")

        # Pick best by real proxy.
        results.sort(key=lambda r: r[3]["proxy"])
        best_label, best_strategy, best_seed, best_data = results[0]
        if self.verbose:
            print(
                f"[multi] WINNER: '{best_label}' "
                f"(strategy={best_strategy} seed={best_seed}) "
                f"proxy={best_data['proxy']:.4f}"
            )
            for label, strategy, seed, data in results[1:]:
                print(f"        runner-up '{label}': proxy={data['proxy']:.4f}")
        return best_data["placement"]


def main():
    """Smoke test."""
    import sys as _sys
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    bench_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    if len(_sys.argv) > 1:
        bench_dir = _sys.argv[1]
    print(f"Loading {bench_dir}...")
    benchmark, plc = load_benchmark_from_dir(bench_dir)
    print(f"  {benchmark}")

    placer = HybridAnalyticalPlacerV2Multi(
        seed=42, num_steps=2000, verbose=True, enable_plots=False,
    )
    placement = placer.place(benchmark)
    metrics = compute_proxy_cost(placement, benchmark, plc)
    print("Final proxy cost:")
    for k in ("proxy_cost", "wirelength_cost", "density_cost", "congestion_cost", "overlap_count"):
        print(f"  {k}={float(metrics[k]):.6f}")


if __name__ == "__main__":
    main()
