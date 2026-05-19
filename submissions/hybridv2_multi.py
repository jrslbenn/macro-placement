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
variation. The benchmark-provided initial placement is usually decent,
but on congestion-bound benches it can be a poor basin; spectral/perturbed
inits can find better basins despite worse Nesterov starts.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from macro_place.benchmark import Benchmark


# Worker configs. The first four are a seed-only fan-out from the provided
# benchmark initial placement; the rest are structural local sweep extras.
WORKER_VARIANTS: List[Dict[str, Any]] = [
    {"label": "provided_42", "init_strategy": "provided", "seed": 42},
    {"label": "provided_137", "init_strategy": "provided", "seed": 137},
    {"label": "provided_271", "init_strategy": "provided", "seed": 271},
    {"label": "provided_911", "init_strategy": "provided", "seed": 911},
    {"label": "halton_42", "init_strategy": "halton", "seed": 42},
    {
        "label": "perturbed_08_271",
        "init_strategy": "perturbed",
        "seed": 271,
        "init_perturb_sigma": 0.08,
    },
    {
        "label": "spectral_70_42",
        "init_strategy": "spectral",
        "seed": 42,
        "init_spectral_blend": 0.70,
        "init_jitter_sigma": 0.005,
    },
    {
        "label": "perturbed_16_911",
        "init_strategy": "perturbed",
        "seed": 911,
        "init_perturb_sigma": 0.16,
    },
    {
        "label": "spectral_85_flip_314",
        "init_strategy": "spectral",
        "seed": 314,
        "init_spectral_blend": 0.85,
        "init_spectral_flip_x": True,
        "init_jitter_sigma": 0.010,
    },
    {"label": "random_42", "init_strategy": "random", "seed": 42},
]

WORKER_SETS = {
    "provided4": ["provided_42", "provided_137", "provided_271", "provided_911"],
    "four": ["provided_42", "halton_42", "perturbed_08_271", "spectral_70_42"],
    "six": [
        "provided_42",
        "halton_42",
        "perturbed_08_271",
        "spectral_70_42",
        "perturbed_16_911",
        "spectral_85_flip_314",
    ],
    "all": [w["label"] for w in WORKER_VARIANTS],
}


def worker_variants_from_env() -> List[Dict[str, Any]]:
    label_to_worker = {w["label"]: dict(w) for w in WORKER_VARIANTS}
    explicit = os.environ.get("HAP_MULTI_WORKERS", "").strip()
    if explicit:
        labels = [x.strip() for x in explicit.split(",") if x.strip()]
    else:
        labels = WORKER_SETS.get(os.environ.get("HAP_MULTI_SET", "four").strip(), WORKER_SETS["four"])
    workers = []
    for label in labels:
        if label not in label_to_worker:
            raise ValueError(f"unknown HAP multi worker '{label}'")
        workers.append(label_to_worker[label])
    return workers


class HybridAnalyticalPlacerV2Multi:
    """Wrapper that spawns N hybridv2 workers, picks best by real proxy."""

    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 50000,
        verbose: bool = True,
        enable_plots: bool = True,
        workers: List[Dict[str, Any]] | None = None,
        threads_per_worker: int = 3,
    ):
        self.seed = seed
        self.num_steps = num_steps
        self.verbose = verbose
        self.enable_plots = enable_plots
        self.workers = workers or worker_variants_from_env()
        self.threads_per_worker = max(1, threads_per_worker)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # We also need the plc path for proxy computation in workers.
        # The harness already loaded benchmark+plc, but plc isn't on the
        # Benchmark dataclass — the worker reloads via load_benchmark_from_dir
        # using a known root + benchmark name.
        # Workers reload from the public benchmark directory for the current
        # Tier-1 benchmark name.
        repo_root = Path(__file__).resolve().parent.parent
        bench_dir = repo_root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
        run_root = (
            Path(os.environ.get("HAP_MULTI_OUT_DIR", "multi_runs"))
            / benchmark.name
            / time.strftime("%Y%m%d_%H%M%S")
        )
        run_root.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[multi] run dir: {run_root}")

        worker_script = repo_root / "submissions" / "_hybridv2_worker.py"
        if not worker_script.exists():
            raise RuntimeError(f"worker script missing: {worker_script}")

        # Launch all workers in parallel.
        procs = []
        result_paths = []
        for worker in self.workers:
            label = str(worker["label"])
            strategy = str(worker["init_strategy"])
            seed = int(worker.get("seed", self.seed))
            out_path = run_root / f"{label}.pkl"
            result_paths.append((label, strategy, seed, out_path))
            worker_log_dir = run_root / "logs" / label
            worker_vis_dir = run_root / "vis" / label
            worker_log_dir.mkdir(parents=True, exist_ok=True)
            worker_vis_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(worker_script),
                "--bench-dir", str(bench_dir),
                "--bench-name", benchmark.name,
                "--init-strategy", strategy,
                "--seed", str(seed),
                "--init-perturb-sigma", str(worker.get("init_perturb_sigma", 0.05)),
                "--init-spectral-blend", str(worker.get("init_spectral_blend", 0.70)),
                "--init-spectral-flip-x", "1" if worker.get("init_spectral_flip_x", False) else "0",
                "--init-spectral-flip-y", "1" if worker.get("init_spectral_flip_y", False) else "0",
                "--init-jitter-sigma", str(worker.get("init_jitter_sigma", 0.0)),
                "--num-steps", str(self.num_steps),
                "--enable-plots", "1" if self.enable_plots else "0",
                "--threads", str(self.threads_per_worker),
                "--log-dir", str(worker_log_dir),
                "--vis-dir", str(worker_vis_dir),
                "--out-path", str(out_path),
            ]
            log_path = run_root / f"{label}.stdout.log"
            log_f = open(log_path, "w")
            if self.verbose:
                print(f"[multi] launching worker '{label}': strategy={strategy} seed={seed}")
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(repo_root))
            procs.append((label, p, log_f, log_path))

        # Wait for all to complete.
        if self.verbose:
            print(f"[multi] waiting for {len(procs)} workers...", flush=True)
        start_t = time.time()
        worker_timeout = float(os.environ.get("HAP_MULTI_WORKER_TIMEOUT", "3200"))
        for label, p, log_f, log_path in procs:
            remaining_timeout = max(1.0, worker_timeout - (time.time() - start_t))
            try:
                p.wait(timeout=remaining_timeout)
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print(
                        f"[multi] worker '{label}' timed out after "
                        f"{worker_timeout:.0f}s; terminating"
                    )
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
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
        seed=42,
        num_steps=2000,
        verbose=True,
        enable_plots=False,
        workers=worker_variants_from_env(),
        threads_per_worker=int(os.environ.get("HAP_MULTI_THREADS", "3")),
    )
    placement = placer.place(benchmark)
    metrics = compute_proxy_cost(placement, benchmark, plc)
    print("Final proxy cost:")
    for k in ("proxy_cost", "wirelength_cost", "density_cost", "congestion_cost", "overlap_count"):
        print(f"  {k}={float(metrics[k]):.6f}")


if __name__ == "__main__":
    main()
