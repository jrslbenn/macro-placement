# HAPpy Placer — Session Handoff Prompt

Use this as the opening message in a new Claude session to resume work without context loss.

---

## Context for Claude

I'm James Bennett, competing in the ICCAD 2026 Partcl/HRT Macro Placement Challenge. Submitting as "HAPpy Placer". My current avg proxy is **~1.30** (best confirmed 1.3296 overnight; ibm14 dropped to 1.4405 in latest A/B). Top-7 cutoff is ~1.20-1.22. Leaderboard #1 is vmallela at 1.0109.

Working directory: `/Users/james.bennett/code/macro-place-challenge-2026/`
Hardware: Mac (Darwin 24.6.0, CPU-only locally). Eval target: AMD EPYC 9655P 16-core + RTX 6000 Ada 48GB, 60-min/bench cap.

## Code structure

- **`placer.py`** at repo root — canonical entry. Subclasses `HybridAnalyticalPlacerV2` from `submissions/hybridv2.py`.
- **`submissions/hybridv2.py`** — main pipeline class. Inherits from parent, adds: DAS-MP weighting, channel relocate (CD-style w/ structural region centers), soft channel relocate, congestion-aware polish, ProgressGate momentum gate.
- **`submissions/hybrid_analytical_placer.py`** — parent. Nesterov + multi-stage SA pipeline. Module-level numba helpers for density grids, HV routing (with pin-level), top-K cost functions.
- **`Dockerfile`** — for judges' build (pytorch 2.3.0-cuda12.1 base).
- **`README.md`** — upstream Partcl/HRT README (restored from upstream after merge conflict resolution).

## Recent verified wins (session 2026-05-12)

1. **Density bug fix** — `_density_cost_top5` was using top 5% without 0.5× multiplier. Now top 10% × 0.5, exact match to TILOS (ratio = 1.0000 verified on ibm01).
2. **Pin-level HV routing** — was using macro centers, now uses pin positions via `_build_pin_hv_route_grid` / `_update_pin_hv_route_incr_single`. Cong calibration drift dropped from 16% to ~4% on ibm01 initial.
3. **ProgressGate outer-loop fix** — channel relocate / soft channel relocate previously had patience going negative because outer while loop didn't check it. Fixed.
4. **Budget bump** — `total_time_budget=1800`, `hard_time_budget=2400` (was 900/1200). SA displace now runs to completion on big benches.

Per-bench impact of combined density + pin-level cong fixes (A/B vs prior 1.3296 baseline):
- ibm14: 1.4941 → **1.4405** (−0.054, biggest single win)
- ibm17: 1.6035 → 1.6165 (+0.013, small regression but recovered most of density-only's +0.019)
- ibm18: 1.7627 → **1.7613** (−0.001)

## What's defaulted OFF (kept as opt-in flags)
- `use_smooth_density=False` — bench-selective: ibm01 −0.015, ibm10 +0.053, ibm14 +0.084
- Cong-Nesterov gradient — commented out in parent's Nesterov loop, net-negative everywhere tested
- `use_routability_inflation=False` — RePlAce-style, marginal regression on ibm14/18

## Critical pending work

### 1. Incremental top-K + smoothed grids (THE KEYSTONE)
vmallela's secret is real-cost-per-move via an incremental evaluator. We have most of the pieces (per-move HPWL, density grid, HV route grid, macro grid). What's missing:

- **`v_smooth`/`h_smooth` incremental maintenance.** Currently `_hv_congestion_cost_top5` rebuilds smoothed grids every call (O(N × smooth_range)). Fix: when `v_route_grid[r,c]` changes by delta, propagate `delta/(2*smooth_range+1)` to `v_smooth[r, c-smooth_range:c+smooth_range+1]`. ~100 lines.

- **Top-K trackers with running sums.** Currently `np.argpartition` recomputes per call (O(N)). Fix: max-heap (Python `heapq`) + hashmap for lazy deletion. Maintain `top_k_sum`. Cost reads become O(1). ~150 lines for one tracker class + 2 instances (density top-10%, cong top-5%).

- **Wire into acceptance.** Replace fast-surrogate-with-calibration in SA stages with direct real-proxy. Drop `den_scale`/`cong_scale` (should be ~1.0 post-pin-level).

Effort: 1-2 days focused. Expected score gain: pipeline becomes vmallela's. Likely lands 1.20-1.28 range.

### 2. Simplify pipeline after keystone
vmallela: "CD + pairwise swaps + parallel restarts. Nothing fancy."
We have 8 sequential stages (SA swap, SA soft swap, soft spread, SA soft displace, SA displace, channel relocate, soft channel relocate, polish). With per-move real-cost eval working, simplify to: CD loop + pair-swap loop + multi-init restarts via existing `hybridv2_multi.py`.

### 3. Multi-init wrapper (already built, never executed)
`submissions/hybridv2_multi.py` spawns 4 subprocess workers per benchmark with different (init_strategy, seed) tuples — `ibm/seed42`, `ibm/seed137`, `spectral/seed42`, `perturbed/seed271`. Picks best by real proxy. Just hasn't been run yet. Quick win, ~4 hours overnight to verify, modest expected gain.

### 4. Smooth-cong for Nesterov (real-cost direct objective on Nesterov)
We have `_compute_smooth_density_loss` (smooth top-10% via sigmoid). Need a smooth-cong analog using HV grids built differentiably. ~200-300 lines. Combined with smooth-density, Nesterov would optimize the actual TILOS proxy directly. Risk: same retuning issue we keep hitting (the pipeline implicitly tunes around quirks).

## Key lessons (re-tuning trap)

This session's repeated finding: **principled structural fixes are usually score-neutral or slightly negative because the pipeline implicitly tunes around the bugs being fixed.** Each fix requires re-tuning everything downstream. Cases:
- Density bug fix alone: +0.0002 / +0.019 / +0.005 on ibm14/17/18 (net negative)
- Cong-Nesterov gradient: net-negative everywhere
- Smooth-density Nesterov objective: bench-selective wins/losses
- Routability inflation: marginal regression
- Pin-level cong wiring: **net positive when COMBINED with density fix** (−0.054 ibm14)

Pattern: pairs of fixes that close the surrogate-vs-real gap together can win where either alone fails.

## Critical TILOS code refs (already verified)
- `external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py:1083-1109` — density = 0.5 × top-10% mean
- `external/.../plc_client_os.py:905-912` — cong = abu(V+H concatenated, 0.05)
- `external/.../plc_client_os.py:1514-1606` — get_routing (pin-level, source-per-MACRO_PIN iteration; we partially match)
- `external/.../plc_client_os.py:1608-1660` — `__smooth_routing_cong` (box filter, H along rows, V along cols, smooth_range typically 2; only routing — NOT macro blockage — is smoothed)

## Run commands

```bash
# Single-bench test
python /tmp/run_hv2_bench.py ibm14

# Full sweep (overnight)
caffeinate -i uv run evaluate ./placer.py --all --vis

# Multi-init parallel (try this!)
python submissions/hybridv2_multi.py external/MacroPlacement/Testcases/ICCAD04/ibm14
```

## Git state
Branch `main`, ahead of `origin/main` by 5 commits. Most recent:
- `10d572d` Wire pin-level HV routing into channel + soft channel relocate
- `875c9a4` Wire pin-level HV routing into SA displace
- `f787975` Add pin-level routing helpers
- `4bcd7f5` Fix density cost to match TILOS exactly
- `c4149b6` Submission cleanup (Dockerfile, READMEs, placer rename)

**Run `git push` to publish before continuing.**

## What I'd ask Claude in the new session

Pick ONE of these depending on time available:

**[1-2 days]** "Build the incremental top-K trackers + incremental smoothed grids. Follow the design in `~/.claude/projects/.../memory/incremental_eval_plan.md`. After it works on ibm14 (target: −0.05 from current best), simplify the pipeline to CD+swap+restart on real cost."

**[~4 hours]** "Run the multi-init wrapper (`submissions/hybridv2_multi.py`) on ibm14, ibm17, ibm18. If it improves any of them by ≥0.02, prepare for full overnight sweep with `MAX_PARALLEL=8` in `eval_all.sh`."

**[~1 hour]** "Verify the placer.py end-to-end via `uv run evaluate ./placer.py --benchmark ibm01`. Then push commits, update Google form entrypoint to `placer.py`, email contact@partcl.com that submission is ready for re-eval."

---

End of handoff. The placer is in a clean, verified-working state. The biggest remaining structural win is the incremental top-K keystone (1-2 days). The quickest verification is the multi-init wrapper (~4 hours overnight).
