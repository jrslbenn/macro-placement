# SPIRAL Placer — James Bennett

Submission for the Partcl/HRT Macro Placement Challenge 2026.

## Run

```bash
uv run evaluate placer.py --all
```

The repo-root `placer.py` is the canonical entry point. It exposes a single class
`Placer` that delegates to `submissions.hybridv2.HybridAnalyticalPlacerV2`, the
full pipeline described below.

## Pipeline

1. **Parent analytical phase** (`submissions/hybrid_analytical_placer.py`):
   Nesterov gradient descent on WL + Poisson-based density spreading + periodic
   legalization, with RePlAce-style routability inflation (per-macro effective
   size grows in high-congestion bins) and top-K candidate selection.
2. **Multi-stage SA refinement** (parent): hard-macro swap, soft swap, soft
   spread, SA soft displace, SA displace — the last with calibrated fast
   surrogate (WL + scaled density + HV-routing congestion) and per-checkpoint
   recalibration.
3. **Channel relocate** (`submissions/hybridv2.py`): pressure-ranked hard-macro
   relocation with structural region targets and incremental HV congestion
   updates; gated by a momentum-based adaptive early-stop ("ProgressGate").
4. **Soft channel relocate**: same idea on soft macros (no overlap check,
   density-aware via top-N percentile), redistributes the soft sea.
5. **Polish**: constrained-tether Nesterov over WL + density + HV-congestion
   gradients with normalized cong forces, real-proxy gating, and early stop.

DAS-MP-style dataflow-aware net weighting boosts weights of nets connecting
hard macros that share many indirect soft-macro paths.

## Dependencies

`torch numpy scipy numba matplotlib tqdm absl-py` — all bundled in the
competition Docker image. A repo-root `Dockerfile` is also provided.

## Contact

James Bennett — jrslbenn@gmail.com
