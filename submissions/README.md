# HAPpyPlace — James Bennett

## Run

```bash
uv run evaluate placer.py --all
```

The canonical entrypoint is `../placer.py` at the repo root, which delegates to
`submissions/hybridv2.HybridAnalyticalPlacerV2`.

## Files in this directory

- **`hybridv2.py`** — main placer class (subclasses `HybridAnalyticalPlacer`,
  adds DAS-MP net weighting, channel relocate with structural region centers,
  soft channel relocate, congestion-aware polish, ProgressGate adaptive
  early-stop).
- **`hybrid_analytical_placer.py`** — parent class providing the analytical
  (Nesterov+Poisson) backbone, multi-stage SA refinement (swap, soft swap,
  soft spread, soft displace, displace), RePlAce-style routability inflation,
  HV-separated routing congestion grids and macro blockage modeling.
- **`hap*.py`** — earlier experimental variants kept for reference; not part
  of the submission flow.

## Dependencies

`torch numpy scipy numba matplotlib tqdm absl-py` (all in the eval Docker image).
