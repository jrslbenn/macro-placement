"""
HybridAnalyticalPlacerV2 — extends HybridAnalyticalPlacer with two ideas:

1. DAS-MP dataflow-aware net weighting (arXiv 2505.16445, May 2025).
   Boost weights for nets connecting hard-macro pairs that share many
   indirect paths through soft macros. Encodes macro-cell-cell-macro
   coupling that the macro-only graph would otherwise miss.

2. Channel-relocate (ported from hap2._real_proxy_channel_relocate, cleaned up):
   HV-separated routing congestion grid + macro-blockage routing model,
   pressure-ranked macro relocation with incremental fast-proxy updates,
   real-proxy gated checkpoints. Helps congestion-dominated benchmarks
   the most (e.g. ibm12: cong=2.36 initially, lots of slack to claim).
   The pin-route variant and structural-region targeting from hap2 are
   dropped — multi-radius ring sampling around the anchor covers the
   same ground with less code.

WireMask refinement was tried in an earlier version of this file but
did not improve proxy cost (it's WL-only and the parent's pipeline has
already converged WL). Removed in favor of channel-relocate.
"""

import importlib.util
import math
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter as time
from typing import Dict, List, Tuple

import numpy as np
import torch
from numba import njit

from macro_place.benchmark import Benchmark
from macro_place.objective import _set_placement, compute_proxy_cost

# Load the parent placer via file path (the eval harness loads each
# submission as a standalone module by file path, so "submissions" isn't
# on sys.path as a package).
_PARENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_analytical_placer.py")
_parent_spec = importlib.util.spec_from_file_location("_hap_parent", _PARENT_PATH)
_parent_mod = importlib.util.module_from_spec(_parent_spec)
_parent_spec.loader.exec_module(_parent_mod)

_IRE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "incremental_real_eval.py")
_ire_spec = importlib.util.spec_from_file_location("_hap_incremental_real_eval", _IRE_PATH)
_ire_mod = importlib.util.module_from_spec(_ire_spec)
_ire_spec.loader.exec_module(_ire_mod)

HybridAnalyticalPlacer = _parent_mod.HybridAnalyticalPlacer
strong_legalize = _parent_mod.strong_legalize
_build_density_grid = _parent_mod._build_density_grid
_update_density_incr = _parent_mod._update_density_incr
_density_cost_top5 = _parent_mod._density_cost_top5
_hpwl_batch = _parent_mod._hpwl_batch
ProgressGate = _parent_mod.ProgressGate
# Pin-level routing helpers (TILOS-faithful — closes cong calibration drift).
_build_pin_hv_route_grid = _parent_mod._build_pin_hv_route_grid
_update_pin_hv_route_incr_single = _parent_mod._update_pin_hv_route_incr_single
build_pin_route_tensors = _parent_mod.build_pin_route_tensors
build_macro_to_pin_nets = _parent_mod.build_macro_to_pin_nets
TopKMeanTracker = _ire_mod.TopKMeanTracker
SmoothHVTopKTracker = _ire_mod.SmoothHVTopKTracker
SmoothHVCostTracker = _ire_mod.SmoothHVCostTracker
update_density_tracker_from_diff = _ire_mod.update_density_tracker_from_diff
update_density_incr_tracked = _ire_mod.update_density_incr_tracked
update_macro_route_incr_tracked = _ire_mod.update_macro_route_incr_tracked
update_hv_route_incr_single_tracked = _ire_mod.update_hv_route_incr_single_tracked
update_pin_hv_route_incr_single_tracked = _ire_mod.update_pin_hv_route_incr_single_tracked
update_hv_route_incr_single_smooth = _ire_mod.update_hv_route_incr_single_smooth
update_pin_hv_route_incr_single_smooth = _ire_mod.update_pin_hv_route_incr_single_smooth


# ──────────────────────────────────────────────────────────────────
# Numba helpers for HV-separated routing congestion + macro blockage.
# Ported verbatim from hap2 (these are performance-critical and tested).
# The HV separation matches TILOS's routing model more accurately than
# isotropic RUDY (which is what the parent's _compute_rudy_map uses).
# ──────────────────────────────────────────────────────────────────


@njit
def _grid_cell_from_xy(x, y, bin_w, bin_h, n_rows, n_cols):
    r = int(y / bin_h)
    c = int(x / bin_w)
    if r < 0:
        r = 0
    elif r >= n_rows:
        r = n_rows - 1
    if c < 0:
        c = 0
    elif c >= n_cols:
        c = n_cols - 1
    return r, c


@njit
def _add_h_segment(h_grid, row, c0, c1, weight, hcap, sign, n_rows, n_cols):
    if row < 0:
        row = 0
    elif row >= n_rows:
        row = n_rows - 1
    lo = c0
    hi = c1
    if lo > hi:
        tmp = lo
        lo = hi
        hi = tmp
    if lo < 0:
        lo = 0
    if hi > n_cols:
        hi = n_cols
    delta = sign * weight / hcap
    for c in range(lo, hi):
        h_grid[row, c] += delta


@njit
def _add_v_segment(v_grid, col, r0, r1, weight, vcap, sign, n_rows, n_cols):
    if col < 0:
        col = 0
    elif col >= n_cols:
        col = n_cols - 1
    lo = r0
    hi = r1
    if lo > hi:
        tmp = lo
        lo = hi
        hi = tmp
    if lo < 0:
        lo = 0
    if hi > n_rows:
        hi = n_rows
    delta = sign * weight / vcap
    for r in range(lo, hi):
        v_grid[r, col] += delta


@njit
def _add_two_pin_route_hv(h_grid, v_grid, sr, sc, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols):
    _add_h_segment(h_grid, sr, min(sc, tc), max(sc, tc), weight, hcap, sign, n_rows, n_cols)
    _add_v_segment(v_grid, tc, min(sr, tr), max(sr, tr), weight, vcap, sign, n_rows, n_cols)


@njit
def _sort3_by_col_row(r0, c0, r1, c1, r2, c2):
    if c0 > c1 or (c0 == c1 and r0 > r1):
        tr, tc = r0, c0
        r0, c0 = r1, c1
        r1, c1 = tr, tc
    if c1 > c2 or (c1 == c2 and r1 > r2):
        tr, tc = r1, c1
        r1, c1 = r2, c2
        r2, c2 = tr, tc
    if c0 > c1 or (c0 == c1 and r0 > r1):
        tr, tc = r0, c0
        r0, c0 = r1, c1
        r1, c1 = tr, tc
    return r0, c0, r1, c1, r2, c2


@njit
def _sort3_by_row_col(r0, c0, r1, c1, r2, c2):
    if r0 > r1 or (r0 == r1 and c0 > c1):
        tr, tc = r0, c0
        r0, c0 = r1, c1
        r1, c1 = tr, tc
    if r1 > r2 or (r1 == r2 and c1 > c2):
        tr, tc = r1, c1
        r1, c1 = r2, c2
        r2, c2 = tr, tc
    if r0 > r1 or (r0 == r1 and c0 > c1):
        tr, tc = r0, c0
        r0, c0 = r1, c1
        r1, c1 = tr, tc
    return r0, c0, r1, c1, r2, c2


@njit
def _add_three_pin_route_hv(h_grid, v_grid, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols):
    r1, c1, r2, c2, r3, c3 = _sort3_by_col_row(
        rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]
    )
    if c1 < c2 and c2 < c3 and min(r1, r3) < r2 and max(r1, r3) > r2:
        _add_h_segment(h_grid, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_h_segment(h_grid, r2, c2, c3, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment(v_grid, c2, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols)
        _add_v_segment(v_grid, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols)
    elif c2 == c3 and c1 < c2 and r1 < min(r2, r3):
        _add_h_segment(h_grid, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment(v_grid, c2, r1, max(r2, r3), weight, vcap, sign, n_rows, n_cols)
    elif r2 == r3:
        _add_h_segment(h_grid, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_h_segment(h_grid, r2, c2, c3, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment(v_grid, c2, min(r2, r1), max(r2, r1), weight, vcap, sign, n_rows, n_cols)
    else:
        r1, c1, r2, c2, r3, c3 = _sort3_by_row_col(
            rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]
        )
        xmin = min(c1, min(c2, c3))
        xmax = max(c1, max(c2, c3))
        _add_h_segment(h_grid, r2, xmin, xmax, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment(v_grid, c1, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols)
        _add_v_segment(v_grid, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols)


@njit
def _add_net_route_hv(
    h_grid, v_grid, net_idx, ni_np, nm_np, nw_np, pos,
    macro_i, override_x, override_y,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap, sign,
):
    """Star+L pattern routing model. macro_i / override_xy temporarily
    relocate one endpoint of the net for incremental updates."""
    max_degree = ni_np.shape[1]
    rows = np.empty(max_degree, dtype=np.int32)
    cols = np.empty(max_degree, dtype=np.int32)
    count = 0
    src_r = 0
    src_c = 0
    source_seen = False
    weight = nw_np[net_idx]

    for d in range(max_degree):
        if not nm_np[net_idx, d]:
            break
        idx = ni_np[net_idx, d]
        if idx == macro_i and override_x >= 0.0:
            x = override_x
            y = override_y
        else:
            x = pos[idx, 0]
            y = pos[idx, 1]
        r, c = _grid_cell_from_xy(x, y, bin_w, bin_h, n_rows, n_cols)
        if not source_seen:
            src_r = r
            src_c = c
            source_seen = True
        duplicate = False
        for k in range(count):
            if rows[k] == r and cols[k] == c:
                duplicate = True
                break
        if not duplicate:
            rows[count] = r
            cols[count] = c
            count += 1

    if count == 2:
        if rows[0] == src_r and cols[0] == src_c:
            tr = rows[1]
            tc = cols[1]
        else:
            tr = rows[0]
            tc = cols[0]
        _add_two_pin_route_hv(h_grid, v_grid, src_r, src_c, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count == 3:
        _add_three_pin_route_hv(h_grid, v_grid, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count > 3:
        for k in range(count):
            if rows[k] == src_r and cols[k] == src_c:
                continue
            _add_two_pin_route_hv(
                h_grid, v_grid, src_r, src_c, rows[k], cols[k],
                weight, hcap, vcap, sign, n_rows, n_cols,
            )


@njit
def _build_hv_route_grid(
    h_grid, v_grid, ni_np, nm_np, nw_np, pos, num_nets,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
):
    for n in range(num_nets):
        _add_net_route_hv(
            h_grid, v_grid, n, ni_np, nm_np, nw_np, pos, -1, -1.0, -1.0,
            bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0,
        )


@njit
def _update_hv_route_incr_single(
    h_grid, v_grid, ni_np, nm_np, nw_np, pos, affected_nets,
    macro_i, old_x, old_y,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
):
    """Subtract net's contribution using old position, re-add with new."""
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        _add_net_route_hv(
            h_grid, v_grid, n, ni_np, nm_np, nw_np, pos, macro_i, old_x, old_y,
            bin_w, bin_h, n_rows, n_cols, hcap, vcap, -1.0,
        )
        _add_net_route_hv(
            h_grid, v_grid, n, ni_np, nm_np, nw_np, pos, -1, -1.0, -1.0,
            bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0,
        )


@njit
def _add_macro_route_blockage(
    h_macro, v_macro, cx, cy, hw, hh, bl, br, bb, bt,
    n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc, sign,
):
    """Macros block routing tracks in proportion to their overlap with bins."""
    left = cx - hw
    right = cx + hw
    bottom = cy - hh
    top = cy + hh
    c0 = max(0, int(left / bin_w))
    c1 = min(n_cols - 1, int(right / bin_w))
    r0 = max(0, int(bottom / bin_h))
    r1 = min(n_rows - 1, int(top / bin_h))
    partial_vertical = False
    partial_horizontal = False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0.0 and oy > 0.0:
                if r0 != r1 and (r == r0 or r == r1) and abs(oy - bin_h) > 1e-5:
                    partial_vertical = True
                if c0 != c1 and (c == c0 or c == c1) and abs(ox - bin_w) > 1e-5:
                    partial_horizontal = True
                v_macro[r, c] += sign * ox * v_alloc / vcap
                h_macro[r, c] += sign * oy * h_alloc / hcap
    if partial_vertical:
        r = r1
        for c in range(c0, c1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0.0 and oy > 0.0:
                v_macro[r, c] -= sign * ox * v_alloc / vcap
    if partial_horizontal:
        c = c1
        for r in range(r0, r1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0.0 and oy > 0.0:
                h_macro[r, c] -= sign * oy * h_alloc / hcap


@njit
def _build_macro_route_grid(
    h_macro, v_macro, pos, sizes, num_hard, bl, br, bb, bt,
    n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
):
    for m in range(num_hard):
        _add_macro_route_blockage(
            h_macro, v_macro, pos[m, 0], pos[m, 1],
            sizes[m, 0] / 2.0, sizes[m, 1] / 2.0,
            bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h,
            hcap, vcap, h_alloc, v_alloc, 1.0,
        )


@njit
def _update_macro_route_incr_single(
    h_macro, v_macro, old_x, old_y, new_x, new_y, hw, hh,
    bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h,
    hcap, vcap, h_alloc, v_alloc,
):
    _add_macro_route_blockage(
        h_macro, v_macro, old_x, old_y, hw, hh, bl, br, bb, bt,
        n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc, -1.0,
    )
    _add_macro_route_blockage(
        h_macro, v_macro, new_x, new_y, hw, hh, bl, br, bb, bt,
        n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc, 1.0,
    )


@njit
def _hv_congestion_cost_top5(h_grid, v_grid, h_macro, v_macro, smooth_range):
    """Mean of top 5% across the smoothed H+macro / V+macro distributions
    (matches TILOS get_congestion_cost = abu(V+H, 0.05) shape)."""
    n_rows = h_grid.shape[0]
    n_cols = h_grid.shape[1]
    v_smooth = np.zeros((n_rows, n_cols), dtype=np.float32)
    h_smooth = np.zeros((n_rows, n_cols), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            lp = c - smooth_range
            if lp < 0:
                lp = 0
            rp = c + smooth_range
            if rp >= n_cols:
                rp = n_cols - 1
            count = rp - lp + 1
            val = v_grid[r, c] / count
            for cc in range(lp, rp + 1):
                v_smooth[r, cc] += val
    for r in range(n_rows):
        for c in range(n_cols):
            lp = r - smooth_range
            if lp < 0:
                lp = 0
            up = r + smooth_range
            if up >= n_rows:
                up = n_rows - 1
            count = up - lp + 1
            val = h_grid[r, c] / count
            for rr in range(lp, up + 1):
                h_smooth[rr, c] += val

    n = n_rows * n_cols
    flat = np.empty(n * 2, dtype=np.float32)
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            flat[idx] = v_smooth[r, c] + v_macro[r, c]
            flat[n + idx] = h_smooth[r, c] + h_macro[r, c]
            idx += 1
    k = max(1, int(flat.shape[0] * 0.05))
    top_idx = np.argpartition(flat, -k)[-k:]
    return flat[top_idx].mean()


@njit
def _hv_pressure_grid(h_grid, v_grid, h_macro, v_macro, smooth_range, out):
    """Build a single per-bin pressure value = max(H_total, V_total) after smoothing."""
    n_rows = h_grid.shape[0]
    n_cols = h_grid.shape[1]
    v_smooth = np.zeros((n_rows, n_cols), dtype=np.float32)
    h_smooth = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            lp = c - smooth_range
            if lp < 0:
                lp = 0
            rp = c + smooth_range
            if rp >= n_cols:
                rp = n_cols - 1
            count = rp - lp + 1
            val = v_grid[r, c] / count
            for cc in range(lp, rp + 1):
                v_smooth[r, cc] += val
    for r in range(n_rows):
        for c in range(n_cols):
            lp = r - smooth_range
            if lp < 0:
                lp = 0
            up = r + smooth_range
            if up >= n_rows:
                up = n_rows - 1
            count = up - lp + 1
            val = h_grid[r, c] / count
            for rr in range(lp, up + 1):
                h_smooth[rr, c] += val
    for r in range(n_rows):
        for c in range(n_cols):
            v_val = v_smooth[r, c] + v_macro[r, c]
            h_val = h_smooth[r, c] + h_macro[r, c]
            out[r, c] = max(v_val, h_val)


# ──────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────


class HybridAnalyticalPlacerV2(HybridAnalyticalPlacer):
    """Parent's pipeline + DAS-MP net weighting + channel-relocate refinement."""

    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 50000,
        lr: float = 1.0,
        momentum: float = 0.9,
        soft_macro_lr: float = 0.15,
        verbose: bool = True,
        enable_plots: bool = True,
        # Initial placement strategy — for multi-init parallel runs.
        # 'ibm':      use benchmark's IBM 2004 floorplan (default)
        # 'spectral': Fiedler-vector spectral layout (parent's helper)
        # 'perturbed': IBM + Gaussian noise (seed-controlled), then legalize
        # 'random':   uniform random, then legalize
        init_strategy: str = "ibm",
        init_perturb_sigma: float = 0.05,  # used when init_strategy='perturbed'
        # ── DAS-MP dataflow weighting ──
        das_enable: bool = True,
        das_indirect_alpha: float = 0.5,
        das_amp_beta: float = 0.6,
        das_max_amp: float = 4.0,
        # ── Channel relocate ──
        ch_enable: bool = True,
        ch_budget: float = 180.0,
        ch_max_stale_sweeps: int = 3,
        ch_max_stale_checkpoints: int = 3,
        ch_checkpoint_accepts: int = 24,
        ch_checkpoint_seconds: float = 45.0,
        ch_trial_cap: int = 24,
        # ── Soft channel relocate ──
        # Same algorithm as hard channel relocate, but operating on
        # standard-cell clusters [num_hard, num_all). No overlap check
        # (soft macros pile up — cost is density, not legality), no
        # macro-route updates (hard macros stay fixed). This was only a
        # tiny win in the incremental-real-eval sweep, so keep it as an
        # opt-in experiment and spend default runtime on stronger stages.
        sch_enable: bool = False,
        sch_budget: float = 180.0,
        sch_max_stale_sweeps: int = 3,
        sch_max_stale_checkpoints: int = 3,
        sch_checkpoint_accepts: int = 60,
        sch_checkpoint_seconds: float = 30.0,
        sch_trial_cap: int = 16,
        # ── Post-channel Nesterov polish ──
        # Soft macros redistribute under WL+density gradient; hard macros
        # are tethered to their channel-relocated positions (±tether_frac
        # × canvas span). Targets the "sea of soft macros" issue on
        # congestion-bound benches (ibm17/18) where channel relocate
        # alone stalls because the bottleneck is downstream of where
        # hard-macro moves operate.
        # Disabled by default after the incremental-real-eval sweep: every
        # logged IBM run restored the pre-polish best checkpoint, so this is
        # currently an experiment knob rather than a productive stage.
        polish_enable: bool = False,
        # On cong-bound benches (ibm17/18) channel relocate stalls
        # immediately because congestion is global (no low-pressure
        # target bins exist for single-macro relocation). Polish is the
        # only stage producing gains there because it does coordinated
        # multi-macro gradient moves. Default settings now favor that
        # case: longer step budget, looser hard-macro tether (so polish
        # can actually re-distribute the collapsed-into-corner layouts),
        # and stronger cong-avoidance weights for both hard and soft.
        # Real-proxy guard reverts if these settings tank an easy bench.
        polish_steps: int = 1000,
        polish_hard_lr: float = 0.5,
        polish_soft_lr: float = 0.15,
        polish_hard_tether_frac: float = 0.25,
        polish_density_weight: float = 0.001,
        polish_soft_density_weight: float = 0.01,
        # Push macros AWAY from already-congested routing bins. Both
        # weights bumped 3-4× from previous defaults — the cong gradient
        # is normalized to [-1,1] so these are the actual fraction-of-
        # per-step-budget moved per step. Refreshed every cong_refresh
        # steps so the map tracks the moving placement.
        polish_soft_cong_weight: float = 0.015,
        polish_hard_cong_weight: float = 0.008,
        polish_cong_refresh: int = 20,
        polish_checkpoint_every: int = 50,
        # ── Incremental real-eval keystone ──
        # Exact top-K density/congestion trackers for channel stages. Keeps
        # the old rebuild path available for A/B via ire_enable=False.
        ire_enable: bool = True,
        ire_debug_validate: bool = False,
    ):
        super().__init__(
            seed=seed,
            num_steps=num_steps,
            lr=lr,
            momentum=momentum,
            soft_macro_lr=soft_macro_lr,
            verbose=verbose,
            enable_plots=enable_plots,
        )
        self.das_enable = das_enable
        self.das_indirect_alpha = das_indirect_alpha
        self.das_amp_beta = das_amp_beta
        self.das_max_amp = das_max_amp
        self.ch_enable = ch_enable
        self.ch_budget = ch_budget
        self.ch_max_stale_sweeps = ch_max_stale_sweeps
        self.ch_max_stale_checkpoints = ch_max_stale_checkpoints
        self.ch_checkpoint_accepts = ch_checkpoint_accepts
        self.ch_checkpoint_seconds = ch_checkpoint_seconds
        self.ch_trial_cap = ch_trial_cap
        self.sch_enable = sch_enable
        self.sch_budget = sch_budget
        self.sch_max_stale_sweeps = sch_max_stale_sweeps
        self.sch_max_stale_checkpoints = sch_max_stale_checkpoints
        self.sch_checkpoint_accepts = sch_checkpoint_accepts
        self.sch_checkpoint_seconds = sch_checkpoint_seconds
        self.sch_trial_cap = sch_trial_cap
        self.polish_enable = polish_enable
        self.polish_steps = polish_steps
        self.polish_hard_lr = polish_hard_lr
        self.polish_soft_lr = polish_soft_lr
        self.polish_hard_tether_frac = polish_hard_tether_frac
        self.polish_density_weight = polish_density_weight
        self.polish_soft_density_weight = polish_soft_density_weight
        self.polish_soft_cong_weight = polish_soft_cong_weight
        self.polish_hard_cong_weight = polish_hard_cong_weight
        self.polish_cong_refresh = polish_cong_refresh
        self.polish_checkpoint_every = polish_checkpoint_every
        self.ire_enable = ire_enable
        self.ire_debug_validate = ire_debug_validate
        self.init_strategy = init_strategy
        self.init_perturb_sigma = init_perturb_sigma
        self._das_benchmark_ref: Benchmark | None = None
        self._cached_nets: List[List[int]] | None = None
        self._cached_net_indices: torch.Tensor | None = None
        self._cached_net_mask: torch.Tensor | None = None
        self._cached_net_weights: torch.Tensor | None = None

    # ────────────────────────────────────────────────────────────────
    # DAS-MP dataflow-aware net weighting
    # ────────────────────────────────────────────────────────────────

    def _compute_dataflow_multipliers(
        self, nets: List[List[int]], num_hard: int, num_macros: int
    ) -> torch.Tensor:
        """
        Per-net multiplier reflecting hard-pair affinity:

            direct[i,j]   = # of nets containing both hard macros i and j
            indirect[i,j] = # of soft macros whose net-neighborhood
                            contains both i and j (transitive coupling)
            affinity[i,j] = direct[i,j] + alpha * indirect[i,j]

        A net containing >=2 hard macros is amplified by mean affinity
        over its hard pairs, normalized and clipped.
        """
        num_soft = num_macros - num_hard
        soft_neighbors: List[set] = [set() for _ in range(num_soft)]
        direct: Dict[Tuple[int, int], int] = defaultdict(int)
        for net in nets:
            hards = sorted(n for n in net if n < num_hard)
            softs = [n - num_hard for n in net if n >= num_hard]
            for s in softs:
                for h in hards:
                    soft_neighbors[s].add(h)
            for i in range(len(hards)):
                for j in range(i + 1, len(hards)):
                    direct[(hards[i], hards[j])] += 1
        indirect: Dict[Tuple[int, int], int] = defaultdict(int)
        for hs_set in soft_neighbors:
            if len(hs_set) > max(8, num_hard // 4):
                # Skip clusters that touch nearly every macro (no signal).
                continue
            hs = sorted(hs_set)
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    indirect[(hs[i], hs[j])] += 1

        multipliers = torch.ones(len(nets), dtype=torch.float32)
        if not direct and not indirect:
            return multipliers
        keys = set(direct) | set(indirect)
        max_aff = max(
            direct.get(k, 0) + self.das_indirect_alpha * indirect.get(k, 0)
            for k in keys
        )
        max_aff = max(max_aff, 1e-6)
        for ni, net in enumerate(nets):
            hards = sorted(n for n in net if n < num_hard)
            if len(hards) < 2:
                continue
            score = 0.0
            count = 0
            for i in range(len(hards)):
                for j in range(i + 1, len(hards)):
                    key = (hards[i], hards[j])
                    score += (
                        direct.get(key, 0)
                        + self.das_indirect_alpha * indirect.get(key, 0)
                    )
                    count += 1
            score /= count
            normalized = score / max_aff
            amp = 1.0 + self.das_amp_beta * normalized * (self.das_max_amp - 1.0)
            multipliers[ni] = min(self.das_max_amp, max(1.0, amp))
        return multipliers

    def _precompute_net_tensors(
        self, nets: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        net_indices, net_mask, base_weights = super()._precompute_net_tensors(nets)
        self._cached_nets = nets
        if not self.das_enable or self._das_benchmark_ref is None:
            new_weights = base_weights
        else:
            bench = self._das_benchmark_ref
            multipliers = self._compute_dataflow_multipliers(
                nets, bench.num_hard_macros, bench.num_macros
            )
            new_weights = base_weights * multipliers
            # Preserve average net weight so WL/density balance is unchanged.
            scale = base_weights.mean() / new_weights.mean().clamp_min(1e-8)
            new_weights = new_weights * scale
            if self.verbose:
                m = multipliers.numpy()
                boosted = int((m > 1.05).sum())
                print(
                    f"[DAS] dataflow weights: min={m.min():.2f} max={m.max():.2f} "
                    f"mean={m.mean():.2f} boosted={boosted}/{len(m)} nets"
                )
        self._cached_net_indices = net_indices
        self._cached_net_mask = net_mask
        self._cached_net_weights = new_weights
        return net_indices, net_mask, new_weights

    # ────────────────────────────────────────────────────────────────
    # Channel relocate
    # ────────────────────────────────────────────────────────────────

    def _channel_relocate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        nets: List[List[int]],
        net_indices: torch.Tensor,
        net_mask: torch.Tensor,
        net_weights: torch.Tensor,
        canvas_norm: float,
        budget: float,
    ) -> torch.Tensor:
        """
        Pressure-ranked hard-macro relocation with HV-separated routing
        congestion, macro-blockage modeling, incremental fast-proxy
        evaluation, and real-proxy gated checkpoints.

        Move proposals: rings of growing radius around (a) the macro's
        current position and (b) its net-neighbor centroid. As staleness
        grows, the radii widen and the uphill tolerance loosens.
        """
        if not self.ch_enable or budget <= 0 or plc is None:
            return placement

        num_hard = benchmark.num_hard_macros
        num_all = benchmark.num_macros
        if num_hard <= 0:
            return placement

        fixed = benchmark.macro_fixed
        sizes = benchmark.macro_sizes[:num_all].numpy().astype(np.float32).copy()
        hw_np = sizes[:, 0] / 2
        hh_np = sizes[:, 1] / 2
        sep_x_np = self._leg_sep_x_base.numpy().copy()
        sep_y_np = self._leg_sep_y_base.numpy().copy()

        cur = placement.detach().clone()
        _set_placement(plc, cur.detach(), benchmark)
        best_metrics = compute_proxy_cost(cur.detach(), benchmark, plc)
        best_proxy = best_metrics["proxy_cost"]
        best = cur.clone()

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = max(1, ni_np.shape[0])

        # Per-macro nets index (for incremental updates).
        macro_to_nets: List[List[int]] = [[] for _ in range(num_all)]
        for net_idx in range(ni_np.shape[0]):
            for d in range(ni_np.shape[1]):
                if not nm_np[net_idx, d]:
                    break
                node = int(ni_np[net_idx, d])
                if 0 <= node < num_all:
                    macro_to_nets[node].append(net_idx)
        macro_to_nets = [np.array(v, dtype=np.int32) for v in macro_to_nets]

        # Grid geometry (parent already set up _bin_w/_bin_h/etc).
        bl = self._bin_left.numpy().copy()
        br = self._bin_right.numpy().copy()
        bb = self._bin_bottom.numpy().copy()
        bt = self._bin_top.numpy().copy()
        bin_w = float(self._bin_w)
        bin_h = float(self._bin_h)
        bin_area = bin_w * bin_h
        n_rows = benchmark.grid_rows
        n_cols = benchmark.grid_cols

        # Routing capacity per bin (tracks/micron × bin extent).
        hcap = max(1e-6, bin_h * float(getattr(benchmark, "hroutes_per_micron", 1.0) or 1.0))
        vcap = max(1e-6, bin_w * float(getattr(benchmark, "vroutes_per_micron", 1.0) or 1.0))
        try:
            smooth_range = int(plc.get_congestion_smooth_range())
        except Exception:
            smooth_range = int(getattr(plc, "smooth_range", 2) or 2)
        smooth_range = max(0, smooth_range)
        try:
            h_alloc, v_alloc = plc.get_macro_routing_allocation()
        except Exception:
            h_alloc = float(getattr(plc, "hrouting_alloc", 0.0) or 0.0)
            v_alloc = float(getattr(plc, "vrouting_alloc", 0.0) or 0.0)
        h_alloc = float(h_alloc)
        v_alloc = float(v_alloc)

        pos = cur.detach().numpy().copy()
        density_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        h_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        v_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        h_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        v_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        pressure_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        _build_density_grid(
            density_grid, pos, sizes, num_all, bl, br, bb, bt, bin_area,
            n_rows, n_cols, bin_w, bin_h,
        )
        # Pin-level routing matches TILOS — closes cong calibration drift.
        # Falls back to macro-center routing if pin data missing.
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos, ni_np.shape[0],
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, pos, sizes, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        net_hpwl = _hpwl_batch(np.arange(ni_np.shape[0], dtype=np.int32), ni_np, nm_np, pos)
        total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
        use_ire = bool(self.ire_enable)
        # Density top-10 is already cheap in numba. The incremental win is
        # maintaining smoothed routing incrementally and avoiding a smoothing
        # rebuild on every candidate.
        density_tracker = None
        cong_tracker = (
            SmoothHVCostTracker(h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range)
            if use_ire
            else None
        )

        def read_density_cost() -> float:
            return density_tracker.cost() if density_tracker is not None else _density_cost_top5(density_grid)

        def read_congestion_cost() -> float:
            return (
                cong_tracker.cost()
                if cong_tracker is not None
                else _hv_congestion_cost_top5(
                    h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
                )
            )

        def sync_density_tracker(old_density_grid: np.ndarray) -> None:
            if density_tracker is not None:
                update_density_tracker_from_diff(density_tracker, old_density_grid, density_grid)

        def sync_route_tracker(old_h_route_grid: np.ndarray, old_v_route_grid: np.ndarray) -> None:
            if cong_tracker is not None:
                cong_tracker.apply_route_grid_diff(old_h_route_grid, old_v_route_grid)

        def sync_macro_tracker(old_h_macro_grid: np.ndarray, old_v_macro_grid: np.ndarray) -> None:
            if cong_tracker is not None:
                cong_tracker.apply_macro_grid_diff(old_h_macro_grid, old_v_macro_grid)

        def validate_incremental_costs(label: str) -> None:
            if not self.ire_debug_validate or not use_ire:
                return
            den_ref = _density_cost_top5(density_grid)
            cong_ref = _hv_congestion_cost_top5(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            den_err = abs(read_density_cost() - den_ref)
            cong_err = abs(read_congestion_cost() - cong_ref)
            if den_err > 1e-5 or cong_err > 2e-5:
                raise RuntimeError(
                    f"Incremental evaluator drift at {label}: "
                    f"den_err={den_err:.3e} cong_err={cong_err:.3e}"
                )

        def apply_density_move(macro_idx: int, from_x: float, from_y: float, to_x: float, to_y: float) -> None:
            if density_tracker is not None:
                update_density_incr_tracked(
                    density_grid, density_tracker, from_x, from_y, to_x, to_y,
                    hw_np[macro_idx], hh_np[macro_idx],
                    bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
                )
            else:
                _update_density_incr(
                    density_grid, from_x, from_y, to_x, to_y,
                    hw_np[macro_idx], hh_np[macro_idx],
                    bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
                )

        def apply_macro_route_move(macro_idx: int, from_x: float, from_y: float, to_x: float, to_y: float) -> None:
            _update_macro_route_incr_single(
                h_macro_grid, v_macro_grid, from_x, from_y, to_x, to_y,
                hw_np[macro_idx], hh_np[macro_idx],
                bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h,
                hcap, vcap, h_alloc, v_alloc,
            )

        def apply_route_move(macro_idx: int, aff: np.ndarray, from_x: float, from_y: float) -> None:
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[macro_idx]
                if len(pin_aff) > 0:
                    if cong_tracker is not None:
                        update_pin_hv_route_incr_single_smooth(
                            h_route_grid, v_route_grid, cong_tracker,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            pos, pin_aff, macro_idx, from_x, from_y,
                            bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
                    else:
                        _update_pin_hv_route_incr_single(
                            h_route_grid, v_route_grid,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            pos, pin_aff, macro_idx, from_x, from_y,
                            bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
            elif len(aff):
                if cong_tracker is not None:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker, ni_np, nm_np, nw_np, pos, aff,
                        macro_idx, from_x, from_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
                else:
                    _update_hv_route_incr_single(
                        h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos, aff,
                        macro_idx, from_x, from_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        current_den = read_density_cost()
        current_cong = read_congestion_cost()

        # Calibrate fast surrogate against real proxy (recalibrated on every
        # successful checkpoint to track drift).
        wl_scale = best_metrics["wirelength_cost"] / (total_wl + 1e-8)
        den_scale = best_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = best_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = (
            total_wl * wl_scale
            + 0.5 * current_den * den_scale
            + 0.5 * current_cong * cong_scale
        )

        if self.verbose:
            print(
                f"Channel relocate start: proxy={best_proxy:.4f} "
                f"wl={best_metrics['wirelength_cost']:.4f} "
                f"den={best_metrics['density_cost']:.4f} "
                f"cong={best_metrics['congestion_cost']:.4f} "
                f"(smooth={smooth_range} hcap={hcap:.2f} vcap={vcap:.2f} "
                f"alloc=({h_alloc:.2f},{v_alloc:.2f}))"
            )

        def make_pressure_score():
            _hv_pressure_grid(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid,
                smooth_range, pressure_grid,
            )
            route_norm = pressure_grid / (float(pressure_grid.mean()) + 1e-6)
            density_norm = density_grid / (float(density_grid.mean()) + 1e-6)
            return np.minimum(route_norm, 8.0) + 0.45 * np.minimum(density_norm, 8.0)

        # Coarse region grid for structural (long-range) move targets.
        # These are essential for catching macros that need to MOVE FAR
        # to clear a channel — the ring sampling alone misses them.
        region_rows = min(8, max(3, n_rows // 4))
        region_cols = min(8, max(3, n_cols // 5))

        def structural_region_centers(
            score_grid_arg, macro_idx: int, old_x: float, old_y: float,
            anchor_x: float, anchor_y: float, stale_level: int,
        ) -> List[Tuple[float, float, float, float, float, float]]:
            """Return low-pressure / high-capacity region centers as
            attractive long-range move targets, ranked by composite cost.

            Early sweeps prefer local regions; staleness widens the radius
            so macros can escape basins they're trapped in."""
            density_norm = density_grid / (float(density_grid.mean()) + 1e-6)
            entries: List[Tuple[float, float, float, float, float, float]] = []
            for rr in range(region_rows):
                r0 = int(rr * n_rows / region_rows)
                r1 = max(r0 + 1, int((rr + 1) * n_rows / region_rows))
                for cc in range(region_cols):
                    c0 = int(cc * n_cols / region_cols)
                    c1 = max(c0 + 1, int((cc + 1) * n_cols / region_cols))
                    p_mean = float(score_grid_arg[r0:r1, c0:c1].mean())
                    d_mean = float(density_norm[r0:r1, c0:c1].mean())
                    route_capacity = max(0.0, 1.05 - p_mean)
                    density_capacity = max(0.0, 0.90 - d_mean)
                    capacity = 0.55 * density_capacity + 0.45 * route_capacity
                    x = ((c0 + c1) * 0.5) * bin_w
                    y = ((r0 + r1) * 0.5) * bin_h
                    x = float(np.clip(x, hw_np[macro_idx], benchmark.canvas_width - hw_np[macro_idx]))
                    y = float(np.clip(y, hh_np[macro_idx], benchmark.canvas_height - hh_np[macro_idx]))
                    anchor_dist = (abs(x - anchor_x) + abs(y - anchor_y)) / span
                    move_dist = (abs(x - old_x) + abs(y - old_y)) / span
                    if stale_level == 0 and anchor_dist > 0.28:
                        continue
                    if stale_level == 1 and anchor_dist > 0.45:
                        continue
                    score = (
                        p_mean
                        + 0.35 * d_mean
                        + 0.18 * anchor_dist
                        + 0.05 * move_dist
                        - 0.38 * capacity
                    )
                    entries.append((score, x, y, p_mean, d_mean, capacity))
            entries.sort(key=lambda item: item[0])
            keep = min(len(entries), 4 + 4 * stale_level)
            return entries[:keep]

        def anchor_centroid(macro_idx: int) -> Tuple[float, float]:
            xs: List[float] = []
            ys: List[float] = []
            for net_idx in macro_to_nets[macro_idx]:
                for d in range(ni_np.shape[1]):
                    if not nm_np[net_idx, d]:
                        break
                    other = int(ni_np[net_idx, d])
                    if other == macro_idx:
                        continue
                    xs.append(float(pos[other, 0]))
                    ys.append(float(pos[other, 1]))
            if not xs:
                return float(pos[macro_idx, 0]), float(pos[macro_idx, 1])
            return float(np.mean(xs)), float(np.mean(ys))

        def legal_at(macro_idx: int, x: float, y: float) -> bool:
            if fixed[macro_idx].item():
                return False
            for k in range(num_hard):
                if k == macro_idx:
                    continue
                if (
                    abs(x - pos[k, 0]) < sep_x_np[macro_idx, k]
                    and abs(y - pos[k, 1]) < sep_y_np[macro_idx, k]
                ):
                    return False
            return True

        def rebuild_fast_state(from_tensor: torch.Tensor) -> None:
            nonlocal pos, net_hpwl, total_wl, current_den, current_cong, current_proxy
            nonlocal density_tracker, cong_tracker
            pos = from_tensor.detach().numpy().copy()
            density_grid[:] = 0
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            h_macro_grid[:] = 0
            v_macro_grid[:] = 0
            _build_density_grid(
                density_grid, pos, sizes, num_all, bl, br, bb, bt, bin_area,
                n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos, ni_np.shape[0],
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, pos, sizes, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )
            net_hpwl = _hpwl_batch(np.arange(ni_np.shape[0], dtype=np.int32), ni_np, nm_np, pos)
            total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            if use_ire:
                density_tracker = None
                cong_tracker = SmoothHVCostTracker(
                    h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
                )
            current_den = read_density_cost()
            current_cong = read_congestion_cost()
            current_proxy = (
                total_wl * wl_scale
                + 0.5 * current_den * den_scale
                + 0.5 * current_cong * cong_scale
            )

        t0 = time()
        accepts = 0
        fast_evals = 0
        real_evals = 1
        stale_sweeps = 0
        checkpoint_accepts = 0
        last_checkpoint_t = t0
        score_grid = make_pressure_score()
        span = benchmark.canvas_width + benchmark.canvas_height
        stop_reason = "budget"
        # ProgressGate: adaptive momentum-based early stop. Improvements
        # build patience; non-improvements drain it. Replaces the fixed
        # stale_checkpoints counter with a self-tuning version.
        gate = ProgressGate(base_patience=4, bonus_per_gain=2, max_patience=12)
        gate.best = best_proxy

        while (
            time() - t0 < budget
            and stale_sweeps < self.ch_max_stale_sweeps
            and gate.patience > 0
        ):
            # Order macros by descending bin-pressure → tackle worst first.
            hard_bins_r = np.clip((pos[:num_hard, 1] / bin_h).astype(np.int32), 0, n_rows - 1)
            hard_bins_c = np.clip((pos[:num_hard, 0] / bin_w).astype(np.int32), 0, n_cols - 1)
            pressure = score_grid[hard_bins_r, hard_bins_c]
            movable = [
                i for i in np.argsort(-pressure).tolist() if not fixed[i].item()
            ]
            stale_level = min(stale_sweeps, 4)
            max_macros = min(
                len(movable),
                max(24, num_hard // 3) + stale_level * max(8, num_hard // 8),
            )
            moved_this_sweep = 0

            for macro_idx in movable[:max_macros]:
                if time() - t0 > budget:
                    break
                old_x = float(pos[macro_idx, 0])
                old_y = float(pos[macro_idx, 1])
                cx, cy = anchor_centroid(macro_idx)
                old_score = float(
                    score_grid[
                        min(n_rows - 1, max(0, int(old_y / bin_h))),
                        min(n_cols - 1, max(0, int(old_x / bin_w))),
                    ]
                )

                # ── Candidate generation: rings around current + anchor +
                # midpoint, PLUS structural region centers (long-range
                # targets — essential for "channel" moves that ring
                # sampling alone misses). ──
                radii = [0.0, 0.015 * span, 0.030 * span, 0.055 * span, 0.085 * span]
                if stale_level >= 1:
                    radii.extend([0.120 * span, 0.170 * span])
                if stale_level >= 2:
                    radii.extend([0.240 * span, 0.320 * span])
                centers = [(old_x, old_y), (cx, cy), ((old_x + cx) * 0.5, (old_y + cy) * 0.5)]
                candidates: List[Tuple[float, float, float, float]] = []

                # Region centers serve a dual role: direct move targets
                # AND extra ring-sampling anchors.
                region_targets = structural_region_centers(
                    score_grid, macro_idx, old_x, old_y, cx, cy, stale_level
                )
                for _region_score, tx, ty, _p_mean, _d_mean, _capacity in region_targets:
                    centers.append((tx, ty))
                    if legal_at(macro_idx, tx, ty):
                        tx_r = min(n_rows - 1, max(0, int(ty / bin_h)))
                        tx_c = min(n_cols - 1, max(0, int(tx / bin_w)))
                        pressure_drop = old_score - float(score_grid[tx_r, tx_c])
                        anchor_dist = (abs(tx - cx) + abs(ty - cy)) / span
                        move_dist = (abs(tx - old_x) + abs(ty - old_y)) / span
                        fast_score = -pressure_drop + 0.10 * anchor_dist + 0.04 * move_dist
                        candidates.append((fast_score, pressure_drop, tx, ty))

                for center_x, center_y in centers:
                    for radius in radii:
                        angle_count = 20 + stale_level * 6
                        for angle_idx in range(angle_count):
                            if radius == 0.0 and angle_idx > 0:
                                break
                            angle = 2 * math.pi * angle_idx / angle_count
                            x = float(np.clip(
                                center_x + math.cos(angle) * radius,
                                hw_np[macro_idx],
                                benchmark.canvas_width - hw_np[macro_idx],
                            ))
                            y = float(np.clip(
                                center_y + math.sin(angle) * radius,
                                hh_np[macro_idx],
                                benchmark.canvas_height - hh_np[macro_idx],
                            ))
                            if not legal_at(macro_idx, x, y):
                                continue
                            cand_r = min(n_rows - 1, max(0, int(y / bin_h)))
                            cand_c = min(n_cols - 1, max(0, int(x / bin_w)))
                            pressure_drop = old_score - float(score_grid[cand_r, cand_c])
                            anchor_dist = (abs(x - cx) + abs(y - cy)) / span
                            move_dist = (abs(x - old_x) + abs(y - old_y)) / span
                            fast_score = -pressure_drop + 0.10 * anchor_dist + 0.04 * move_dist
                            candidates.append((fast_score, pressure_drop, x, y))

                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0])
                trial_cap = self.ch_trial_cap + 12 * stale_level
                trial_candidates = candidates[:trial_cap]
                local_best_score = current_proxy
                local_best = None
                uphill_allowance = 0.00025 * stale_level
                pressure_floor = -0.25 - 0.10 * stale_level

                for _fast_score, pressure_drop, x, y in trial_candidates:
                    if time() - t0 > budget:
                        break
                    if pressure_drop < pressure_floor:
                        continue
                    aff = macro_to_nets[macro_idx]
                    old_hpwl = net_hpwl[aff].copy() if len(aff) else np.zeros(0, dtype=np.float32)

                    # ── Apply move incrementally ──
                    pos[macro_idx, 0] = x
                    pos[macro_idx, 1] = y
                    apply_density_move(macro_idx, old_x, old_y, x, y)
                    apply_macro_route_move(macro_idx, old_x, old_y, x, y)
                    apply_route_move(macro_idx, aff, old_x, old_y)
                    if len(aff):
                        new_hpwl = _hpwl_batch(aff, ni_np, nm_np, pos)
                        wl_delta = float(((new_hpwl - old_hpwl) * nw_np[aff]).sum()) / (
                            num_nets * canvas_norm
                        )
                    else:
                        new_hpwl = old_hpwl
                        wl_delta = 0.0
                    new_den = read_density_cost()
                    new_cong = read_congestion_cost()
                    validate_incremental_costs("hard-trial-apply")
                    proxy = (
                        (total_wl + wl_delta) * wl_scale
                        + 0.5 * new_den * den_scale
                        + 0.5 * new_cong * cong_scale
                    )
                    fast_evals += 1

                    # ── Always revert; only commit local_best below ──
                    pos[macro_idx, 0] = old_x
                    pos[macro_idx, 1] = old_y
                    apply_density_move(macro_idx, x, y, old_x, old_y)
                    apply_macro_route_move(macro_idx, x, y, old_x, old_y)
                    apply_route_move(macro_idx, aff, x, y)
                    validate_incremental_costs("hard-trial-revert")

                    pressure_bonus = min(0.003, max(0.0, pressure_drop) * (0.00035 * stale_level))
                    selection_score = proxy - pressure_bonus
                    if (
                        selection_score < local_best_score - 5e-5
                        and proxy <= current_proxy + uphill_allowance
                    ):
                        local_best_score = selection_score
                        local_best = (x, y, aff, new_hpwl.copy(), wl_delta, new_den, new_cong, proxy)

                # ── Commit best candidate (if any) ──
                if local_best is not None:
                    x, y, aff, new_hpwl, wl_delta, new_den, new_cong, proxy = local_best
                    pos[macro_idx, 0] = x
                    pos[macro_idx, 1] = y
                    cur[macro_idx, 0] = x
                    cur[macro_idx, 1] = y
                    apply_density_move(macro_idx, old_x, old_y, x, y)
                    apply_macro_route_move(macro_idx, old_x, old_y, x, y)
                    apply_route_move(macro_idx, aff, old_x, old_y)
                    validate_incremental_costs("hard-commit")
                    if len(aff):
                        net_hpwl[aff] = new_hpwl
                    total_wl += wl_delta
                    current_den = new_den
                    current_cong = new_cong
                    current_proxy = proxy
                    accepts += 1
                    checkpoint_accepts += 1
                    moved_this_sweep += 1
                    score_grid = make_pressure_score()

                    # ── Real-proxy checkpoint ──
                    should_checkpoint = (
                        checkpoint_accepts >= self.ch_checkpoint_accepts
                        or time() - last_checkpoint_t >= self.ch_checkpoint_seconds
                    )
                    if should_checkpoint:
                        _set_placement(plc, cur.detach(), benchmark)
                        metrics = compute_proxy_cost(cur.detach(), benchmark, plc)
                        real_evals += 1
                        real_proxy = metrics["proxy_cost"]
                        if self.verbose:
                            print(
                                f"  channel chk accepts={accepts} fast_evals={fast_evals} "
                                f"fast={current_proxy:.4f} real={real_proxy:.4f} "
                                f"best={best_proxy:.4f} wl={metrics['wirelength_cost']:.4f} "
                                f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                                f"[{time()-t0:.0f}s]"
                            )
                        checkpoint_accepts = 0
                        last_checkpoint_t = time()
                        improved, should_stop = gate.update(real_proxy)
                        if improved:
                            best_proxy = real_proxy
                            best = cur.clone()
                            # Recalibrate surrogate scales to track real proxy.
                            wl_scale = metrics["wirelength_cost"] / (total_wl + 1e-8)
                            den_scale = metrics["density_cost"] / (current_den + 1e-8)
                            cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                            current_proxy = (
                                total_wl * wl_scale
                                + 0.5 * current_den * den_scale
                                + 0.5 * current_cong * cong_scale
                            )
                            if self.verbose:
                                print(f"    gate: +imp, patience={gate.patience}/{gate.max_patience}")
                        else:
                            cur = best.clone()
                            rebuild_fast_state(cur)
                            score_grid = make_pressure_score()
                            if self.verbose:
                                print(f"    gate: miss, patience={gate.patience}/{gate.max_patience}")
                            if should_stop:
                                stop_reason = f"progress_gate (best={gate.best:.4f})"
                                if self.verbose:
                                    print("Channel relocate stalled (progress gate)")
                                break

            if moved_this_sweep == 0:
                stale_sweeps += 1
                if self.verbose and stale_sweeps < self.ch_max_stale_sweeps:
                    print(
                        f"  channel widen: stale_sweeps={stale_sweeps} "
                        f"fast_evals={fast_evals} [{time()-t0:.0f}s]"
                    )
            else:
                stale_sweeps = 0

        if time() - t0 >= budget:
            stop_reason = "budget"
        elif stale_sweeps >= self.ch_max_stale_sweeps:
            stop_reason = f"stale_sweeps={stale_sweeps}"
        elif gate.patience <= 0:
            stop_reason = f"progress_gate(best={gate.best:.4f})"

        # Final checkpoint if any uncommitted accepts.
        if checkpoint_accepts > 0:
            _set_placement(plc, cur.detach(), benchmark)
            metrics = compute_proxy_cost(cur.detach(), benchmark, plc)
            real_evals += 1
            if metrics["proxy_cost"] < best_proxy - 1e-4:
                best_proxy = metrics["proxy_cost"]
                best = cur.clone()

        if self.verbose:
            print(
                f"Channel relocate done: accepts={accepts} fast_evals={fast_evals} "
                f"real_evals={real_evals} best proxy={best_proxy:.4f} stop={stop_reason}"
            )
        return best

    # ────────────────────────────────────────────────────────────────
    # Soft channel relocate
    # ────────────────────────────────────────────────────────────────

    def _soft_channel_relocate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        nets: List[List[int]],
        net_indices: torch.Tensor,
        net_mask: torch.Tensor,
        net_weights: torch.Tensor,
        canvas_norm: float,
        budget: float,
    ) -> torch.Tensor:
        """
        Long-range relocation for soft macros (standard-cell clusters).

        Mirrors `_channel_relocate` but iterates over [num_hard, num_all):
          - no overlap check (soft macros pile up; cost is density)
          - no macro-route blockage updates (hard macros stay fixed)
          - HV routing congestion + density still updated per move
          - real-proxy gated checkpoints with revert on regression
        """
        if not self.sch_enable or budget <= 0 or plc is None:
            return placement
        num_hard = benchmark.num_hard_macros
        num_all = benchmark.num_macros
        num_soft = num_all - num_hard
        if num_soft <= 0:
            return placement

        fixed = benchmark.macro_fixed
        sizes = benchmark.macro_sizes[:num_all].numpy().astype(np.float32).copy()
        hw_np = sizes[:, 0] / 2
        hh_np = sizes[:, 1] / 2

        cur = placement.detach().clone()
        _set_placement(plc, cur.detach(), benchmark)
        best_metrics = compute_proxy_cost(cur.detach(), benchmark, plc)
        best_proxy = best_metrics["proxy_cost"]
        best = cur.clone()

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = max(1, ni_np.shape[0])
        macro_to_nets: List[List[int]] = [[] for _ in range(num_all)]
        for net_idx in range(ni_np.shape[0]):
            for d in range(ni_np.shape[1]):
                if not nm_np[net_idx, d]:
                    break
                node = int(ni_np[net_idx, d])
                if 0 <= node < num_all:
                    macro_to_nets[node].append(net_idx)
        macro_to_nets = [np.array(v, dtype=np.int32) for v in macro_to_nets]

        bl = self._bin_left.numpy().copy()
        br = self._bin_right.numpy().copy()
        bb = self._bin_bottom.numpy().copy()
        bt = self._bin_top.numpy().copy()
        bin_w = float(self._bin_w)
        bin_h = float(self._bin_h)
        bin_area = bin_w * bin_h
        n_rows = benchmark.grid_rows
        n_cols = benchmark.grid_cols

        hcap = max(1e-6, bin_h * float(getattr(benchmark, "hroutes_per_micron", 1.0) or 1.0))
        vcap = max(1e-6, bin_w * float(getattr(benchmark, "vroutes_per_micron", 1.0) or 1.0))
        try:
            smooth_range = int(plc.get_congestion_smooth_range())
        except Exception:
            smooth_range = int(getattr(plc, "smooth_range", 2) or 2)
        smooth_range = max(0, smooth_range)
        try:
            h_alloc, v_alloc = plc.get_macro_routing_allocation()
        except Exception:
            h_alloc = float(getattr(plc, "hrouting_alloc", 0.0) or 0.0)
            v_alloc = float(getattr(plc, "vrouting_alloc", 0.0) or 0.0)
        h_alloc = float(h_alloc)
        v_alloc = float(v_alloc)

        pos = cur.detach().numpy().copy()
        density_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        h_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        v_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        # Hard-macro blockage: built ONCE, never updated (hard macros are fixed).
        h_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        v_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        pressure_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        _build_density_grid(
            density_grid, pos, sizes, num_all, bl, br, bb, bt, bin_area,
            n_rows, n_cols, bin_w, bin_h,
        )
        # Pin-level routing matches TILOS — closes cong calibration drift.
        # Falls back to macro-center routing if pin data missing.
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos, ni_np.shape[0],
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, pos, sizes, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        net_hpwl = _hpwl_batch(np.arange(ni_np.shape[0], dtype=np.int32), ni_np, nm_np, pos)
        total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
        use_ire = bool(self.ire_enable)
        density_tracker = None
        cong_tracker = (
            SmoothHVCostTracker(h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range)
            if use_ire
            else None
        )

        def read_density_cost() -> float:
            return density_tracker.cost() if density_tracker is not None else _density_cost_top5(density_grid)

        def read_congestion_cost() -> float:
            return (
                cong_tracker.cost()
                if cong_tracker is not None
                else _hv_congestion_cost_top5(
                    h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
                )
            )

        def sync_density_tracker(old_density_grid: np.ndarray) -> None:
            if density_tracker is not None:
                update_density_tracker_from_diff(density_tracker, old_density_grid, density_grid)

        def sync_route_tracker(old_h_route_grid: np.ndarray, old_v_route_grid: np.ndarray) -> None:
            if cong_tracker is not None:
                cong_tracker.apply_route_grid_diff(old_h_route_grid, old_v_route_grid)

        def validate_incremental_costs(label: str) -> None:
            if not self.ire_debug_validate or not use_ire:
                return
            den_ref = _density_cost_top5(density_grid)
            cong_ref = _hv_congestion_cost_top5(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            den_err = abs(read_density_cost() - den_ref)
            cong_err = abs(read_congestion_cost() - cong_ref)
            if den_err > 1e-5 or cong_err > 2e-5:
                raise RuntimeError(
                    f"Incremental evaluator drift at {label}: "
                    f"den_err={den_err:.3e} cong_err={cong_err:.3e}"
                )

        def apply_density_move(macro_idx: int, from_x: float, from_y: float, to_x: float, to_y: float) -> None:
            if density_tracker is not None:
                update_density_incr_tracked(
                    density_grid, density_tracker, from_x, from_y, to_x, to_y,
                    hw_np[macro_idx], hh_np[macro_idx],
                    bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
                )
            else:
                _update_density_incr(
                    density_grid, from_x, from_y, to_x, to_y,
                    hw_np[macro_idx], hh_np[macro_idx],
                    bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
                )

        def apply_route_move(macro_idx: int, aff: np.ndarray, from_x: float, from_y: float) -> None:
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[macro_idx]
                if len(pin_aff) > 0:
                    if cong_tracker is not None:
                        update_pin_hv_route_incr_single_smooth(
                            h_route_grid, v_route_grid, cong_tracker,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            pos, pin_aff, macro_idx, from_x, from_y,
                            bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
                    else:
                        _update_pin_hv_route_incr_single(
                            h_route_grid, v_route_grid,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            pos, pin_aff, macro_idx, from_x, from_y,
                            bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
            elif len(aff):
                if cong_tracker is not None:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker, ni_np, nm_np, nw_np, pos, aff,
                        macro_idx, from_x, from_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
                else:
                    _update_hv_route_incr_single(
                        h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos, aff,
                        macro_idx, from_x, from_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        current_den = read_density_cost()
        current_cong = read_congestion_cost()
        wl_scale = best_metrics["wirelength_cost"] / (total_wl + 1e-8)
        den_scale = best_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = best_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = (
            total_wl * wl_scale
            + 0.5 * current_den * den_scale
            + 0.5 * current_cong * cong_scale
        )

        if self.verbose:
            print(
                f"Soft channel relocate start: proxy={best_proxy:.4f} "
                f"wl={best_metrics['wirelength_cost']:.4f} "
                f"den={best_metrics['density_cost']:.4f} "
                f"cong={best_metrics['congestion_cost']:.4f} "
                f"num_soft={num_soft}"
            )

        def make_pressure_score():
            _hv_pressure_grid(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid,
                smooth_range, pressure_grid,
            )
            route_norm = pressure_grid / (float(pressure_grid.mean()) + 1e-6)
            density_norm = density_grid / (float(density_grid.mean()) + 1e-6)
            # Soft macros suffer more from density (clustering hurts more
            # than for hard since they have no legal-separation buffer),
            # so weight density higher in the move-target score.
            return np.minimum(route_norm, 8.0) + 0.65 * np.minimum(density_norm, 8.0)

        def anchor_centroid(macro_idx: int) -> Tuple[float, float]:
            xs: List[float] = []
            ys: List[float] = []
            for net_idx in macro_to_nets[macro_idx]:
                for d in range(ni_np.shape[1]):
                    if not nm_np[net_idx, d]:
                        break
                    other = int(ni_np[net_idx, d])
                    if other == macro_idx:
                        continue
                    xs.append(float(pos[other, 0]))
                    ys.append(float(pos[other, 1]))
            if not xs:
                return float(pos[macro_idx, 0]), float(pos[macro_idx, 1])
            return float(np.mean(xs)), float(np.mean(ys))

        def rebuild_fast_state(from_tensor: torch.Tensor) -> None:
            nonlocal pos, net_hpwl, total_wl, current_den, current_cong, current_proxy
            nonlocal density_tracker, cong_tracker
            pos = from_tensor.detach().numpy().copy()
            density_grid[:] = 0
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            # Note: macro_grid does NOT get rebuilt — hard macros are fixed.
            _build_density_grid(
                density_grid, pos, sizes, num_all, bl, br, bb, bt, bin_area,
                n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos, ni_np.shape[0],
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            net_hpwl = _hpwl_batch(np.arange(ni_np.shape[0], dtype=np.int32), ni_np, nm_np, pos)
            total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            if use_ire:
                density_tracker = None
                cong_tracker = SmoothHVCostTracker(
                    h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
                )
            current_den = read_density_cost()
            current_cong = read_congestion_cost()
            current_proxy = (
                total_wl * wl_scale
                + 0.5 * current_den * den_scale
                + 0.5 * current_cong * cong_scale
            )

        t0 = time()
        accepts = 0
        fast_evals = 0
        real_evals = 1
        stale_sweeps = 0
        checkpoint_accepts = 0
        last_checkpoint_t = t0
        score_grid = make_pressure_score()
        span = benchmark.canvas_width + benchmark.canvas_height
        stop_reason = "budget"
        gate = ProgressGate(base_patience=4, bonus_per_gain=2, max_patience=12)
        gate.best = best_proxy

        # Coarse region grid for soft long-range moves.
        region_rows = min(8, max(3, n_rows // 4))
        region_cols = min(8, max(3, n_cols // 5))

        def soft_region_centers(
            score_grid_arg, macro_idx, old_x, old_y, anchor_x, anchor_y, stale_level,
        ) -> List[Tuple[float, float, float, float, float, float]]:
            density_norm = density_grid / (float(density_grid.mean()) + 1e-6)
            entries: List[Tuple[float, float, float, float, float, float]] = []
            for rr in range(region_rows):
                r0 = int(rr * n_rows / region_rows)
                r1 = max(r0 + 1, int((rr + 1) * n_rows / region_rows))
                for cc in range(region_cols):
                    c0 = int(cc * n_cols / region_cols)
                    c1 = max(c0 + 1, int((cc + 1) * n_cols / region_cols))
                    p_mean = float(score_grid_arg[r0:r1, c0:c1].mean())
                    d_mean = float(density_norm[r0:r1, c0:c1].mean())
                    route_capacity = max(0.0, 1.05 - p_mean)
                    density_capacity = max(0.0, 0.90 - d_mean)
                    capacity = 0.60 * density_capacity + 0.40 * route_capacity
                    x = ((c0 + c1) * 0.5) * bin_w
                    y = ((r0 + r1) * 0.5) * bin_h
                    x = float(np.clip(x, hw_np[macro_idx], benchmark.canvas_width - hw_np[macro_idx]))
                    y = float(np.clip(y, hh_np[macro_idx], benchmark.canvas_height - hh_np[macro_idx]))
                    anchor_dist = (abs(x - anchor_x) + abs(y - anchor_y)) / span
                    move_dist = (abs(x - old_x) + abs(y - old_y)) / span
                    # Soft macros: be more aggressive about widening on stale.
                    if stale_level == 0 and anchor_dist > 0.35:
                        continue
                    if stale_level == 1 and anchor_dist > 0.55:
                        continue
                    score = (
                        p_mean
                        + 0.45 * d_mean
                        + 0.15 * anchor_dist
                        + 0.05 * move_dist
                        - 0.40 * capacity
                    )
                    entries.append((score, x, y, p_mean, d_mean, capacity))
            entries.sort(key=lambda item: item[0])
            keep = min(len(entries), 4 + 4 * stale_level)
            return entries[:keep]

        while (
            time() - t0 < budget
            and stale_sweeps < self.sch_max_stale_sweeps
            and gate.patience > 0
        ):
            soft_bins_r = np.clip((pos[num_hard:num_all, 1] / bin_h).astype(np.int32), 0, n_rows - 1)
            soft_bins_c = np.clip((pos[num_hard:num_all, 0] / bin_w).astype(np.int32), 0, n_cols - 1)
            pressure = score_grid[soft_bins_r, soft_bins_c]
            # Soft macro indices are num_hard + i where i is local soft index.
            movable_local = [
                i for i in np.argsort(-pressure).tolist()
                if not fixed[num_hard + i].item()
            ]
            stale_level = min(stale_sweeps, 4)
            max_macros = min(
                len(movable_local),
                max(48, num_soft // 3) + stale_level * max(16, num_soft // 8),
            )
            moved_this_sweep = 0

            for local_idx in movable_local[:max_macros]:
                if time() - t0 > budget:
                    break
                macro_idx = num_hard + local_idx
                old_x = float(pos[macro_idx, 0])
                old_y = float(pos[macro_idx, 1])
                cx, cy = anchor_centroid(macro_idx)
                old_score = float(
                    score_grid[
                        min(n_rows - 1, max(0, int(old_y / bin_h))),
                        min(n_cols - 1, max(0, int(old_x / bin_w))),
                    ]
                )

                radii = [0.0, 0.02 * span, 0.04 * span, 0.07 * span, 0.11 * span]
                if stale_level >= 1:
                    radii.extend([0.15 * span, 0.22 * span])
                if stale_level >= 2:
                    radii.extend([0.30 * span, 0.40 * span])
                centers = [(old_x, old_y), (cx, cy), ((old_x + cx) * 0.5, (old_y + cy) * 0.5)]
                candidates: List[Tuple[float, float, float, float]] = []

                region_targets = soft_region_centers(
                    score_grid, macro_idx, old_x, old_y, cx, cy, stale_level
                )
                for _region_score, tx, ty, _p_mean, _d_mean, _capacity in region_targets:
                    centers.append((tx, ty))
                    tx_r = min(n_rows - 1, max(0, int(ty / bin_h)))
                    tx_c = min(n_cols - 1, max(0, int(tx / bin_w)))
                    pressure_drop = old_score - float(score_grid[tx_r, tx_c])
                    anchor_dist = (abs(tx - cx) + abs(ty - cy)) / span
                    move_dist = (abs(tx - old_x) + abs(ty - old_y)) / span
                    fast_score = -pressure_drop + 0.08 * anchor_dist + 0.04 * move_dist
                    candidates.append((fast_score, pressure_drop, tx, ty))

                for center_x, center_y in centers:
                    for radius in radii:
                        angle_count = 12 + stale_level * 4
                        for angle_idx in range(angle_count):
                            if radius == 0.0 and angle_idx > 0:
                                break
                            angle = 2 * math.pi * angle_idx / angle_count
                            x = float(np.clip(
                                center_x + math.cos(angle) * radius,
                                hw_np[macro_idx],
                                benchmark.canvas_width - hw_np[macro_idx],
                            ))
                            y = float(np.clip(
                                center_y + math.sin(angle) * radius,
                                hh_np[macro_idx],
                                benchmark.canvas_height - hh_np[macro_idx],
                            ))
                            # No legality check — soft macros may overlap.
                            cand_r = min(n_rows - 1, max(0, int(y / bin_h)))
                            cand_c = min(n_cols - 1, max(0, int(x / bin_w)))
                            pressure_drop = old_score - float(score_grid[cand_r, cand_c])
                            anchor_dist = (abs(x - cx) + abs(y - cy)) / span
                            move_dist = (abs(x - old_x) + abs(y - old_y)) / span
                            fast_score = -pressure_drop + 0.08 * anchor_dist + 0.04 * move_dist
                            candidates.append((fast_score, pressure_drop, x, y))

                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0])
                trial_cap = self.sch_trial_cap + 8 * stale_level
                trial_candidates = candidates[:trial_cap]
                local_best_score = current_proxy
                local_best = None
                uphill_allowance = 0.00020 * stale_level
                pressure_floor = -0.30 - 0.10 * stale_level

                for _fast_score, pressure_drop, x, y in trial_candidates:
                    if time() - t0 > budget:
                        break
                    if pressure_drop < pressure_floor:
                        continue
                    aff = macro_to_nets[macro_idx]
                    old_hpwl = (
                        net_hpwl[aff].copy() if len(aff) else np.zeros(0, dtype=np.float32)
                    )

                    # Apply move incrementally (density + HV routing only).
                    pos[macro_idx, 0] = x
                    pos[macro_idx, 1] = y
                    apply_density_move(macro_idx, old_x, old_y, x, y)
                    apply_route_move(macro_idx, aff, old_x, old_y)
                    if len(aff):
                        new_hpwl = _hpwl_batch(aff, ni_np, nm_np, pos)
                        wl_delta = float(((new_hpwl - old_hpwl) * nw_np[aff]).sum()) / (
                            num_nets * canvas_norm
                        )
                    else:
                        new_hpwl = old_hpwl
                        wl_delta = 0.0
                    new_den = read_density_cost()
                    new_cong = read_congestion_cost()
                    validate_incremental_costs("soft-trial-apply")
                    proxy = (
                        (total_wl + wl_delta) * wl_scale
                        + 0.5 * new_den * den_scale
                        + 0.5 * new_cong * cong_scale
                    )
                    fast_evals += 1

                    # Always revert; only commit local_best below.
                    pos[macro_idx, 0] = old_x
                    pos[macro_idx, 1] = old_y
                    apply_density_move(macro_idx, x, y, old_x, old_y)
                    apply_route_move(macro_idx, aff, x, y)
                    validate_incremental_costs("soft-trial-revert")

                    pressure_bonus = min(0.003, max(0.0, pressure_drop) * (0.00035 * stale_level))
                    selection_score = proxy - pressure_bonus
                    if (
                        selection_score < local_best_score - 5e-5
                        and proxy <= current_proxy + uphill_allowance
                    ):
                        local_best_score = selection_score
                        local_best = (x, y, aff, new_hpwl.copy(), wl_delta, new_den, new_cong, proxy)

                if local_best is not None:
                    x, y, aff, new_hpwl, wl_delta, new_den, new_cong, proxy = local_best
                    pos[macro_idx, 0] = x
                    pos[macro_idx, 1] = y
                    cur[macro_idx, 0] = x
                    cur[macro_idx, 1] = y
                    apply_density_move(macro_idx, old_x, old_y, x, y)
                    apply_route_move(macro_idx, aff, old_x, old_y)
                    validate_incremental_costs("soft-commit")
                    if len(aff):
                        net_hpwl[aff] = new_hpwl
                    total_wl += wl_delta
                    current_den = new_den
                    current_cong = new_cong
                    current_proxy = proxy
                    accepts += 1
                    checkpoint_accepts += 1
                    moved_this_sweep += 1
                    score_grid = make_pressure_score()

                    should_checkpoint = (
                        checkpoint_accepts >= self.sch_checkpoint_accepts
                        or time() - last_checkpoint_t >= self.sch_checkpoint_seconds
                    )
                    if should_checkpoint:
                        _set_placement(plc, cur.detach(), benchmark)
                        metrics = compute_proxy_cost(cur.detach(), benchmark, plc)
                        real_evals += 1
                        real_proxy = metrics["proxy_cost"]
                        if self.verbose:
                            print(
                                f"  soft channel chk accepts={accepts} fast_evals={fast_evals} "
                                f"fast={current_proxy:.4f} real={real_proxy:.4f} "
                                f"best={best_proxy:.4f} wl={metrics['wirelength_cost']:.4f} "
                                f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                                f"[{time()-t0:.0f}s]"
                            )
                        checkpoint_accepts = 0
                        last_checkpoint_t = time()
                        improved, should_stop = gate.update(real_proxy)
                        if improved:
                            best_proxy = real_proxy
                            best = cur.clone()
                            wl_scale = metrics["wirelength_cost"] / (total_wl + 1e-8)
                            den_scale = metrics["density_cost"] / (current_den + 1e-8)
                            cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                            current_proxy = (
                                total_wl * wl_scale
                                + 0.5 * current_den * den_scale
                                + 0.5 * current_cong * cong_scale
                            )
                            if self.verbose:
                                print(f"    gate: +imp, patience={gate.patience}/{gate.max_patience}")
                        else:
                            cur = best.clone()
                            rebuild_fast_state(cur)
                            score_grid = make_pressure_score()
                            if self.verbose:
                                print(f"    gate: miss, patience={gate.patience}/{gate.max_patience}")
                            if should_stop:
                                stop_reason = f"progress_gate(best={gate.best:.4f})"
                                if self.verbose:
                                    print("Soft channel relocate stalled (progress gate)")
                                break

            if moved_this_sweep == 0:
                stale_sweeps += 1
                if self.verbose and stale_sweeps < self.sch_max_stale_sweeps:
                    print(
                        f"  soft channel widen: stale_sweeps={stale_sweeps} "
                        f"fast_evals={fast_evals} [{time()-t0:.0f}s]"
                    )
            else:
                stale_sweeps = 0

        if time() - t0 >= budget:
            stop_reason = "budget"
        elif stale_sweeps >= self.sch_max_stale_sweeps:
            stop_reason = f"stale_sweeps={stale_sweeps}"
        elif gate.patience <= 0:
            stop_reason = f"progress_gate(best={gate.best:.4f})"

        if checkpoint_accepts > 0:
            _set_placement(plc, cur.detach(), benchmark)
            metrics = compute_proxy_cost(cur.detach(), benchmark, plc)
            real_evals += 1
            if metrics["proxy_cost"] < best_proxy - 1e-4:
                best_proxy = metrics["proxy_cost"]
                best = cur.clone()

        if self.verbose:
            print(
                f"Soft channel relocate done: accepts={accepts} fast_evals={fast_evals} "
                f"real_evals={real_evals} best proxy={best_proxy:.4f} stop={stop_reason}"
            )
        return best

    # ────────────────────────────────────────────────────────────────
    # Post-channel Nesterov polish
    # ────────────────────────────────────────────────────────────────

    def _post_channel_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
        nets: List[List[int]],
        net_indices: torch.Tensor,
        net_mask: torch.Tensor,
        net_weights: torch.Tensor,
        canvas_norm: float,
    ) -> torch.Tensor:
        """
        Constrained Nesterov refinement run *after* channel relocate.

        Hard macros are clamped to a ±tether_frac × span box around their
        channel-relocated positions — so the analytical gradient can only
        nudge them, not undo channel relocate. Soft macros move freely
        under WL + density gradient (heavier density weight than parent's
        soft Nesterov to actively spread the "soft sea"). Periodic
        real-proxy checkpoint with revert on regression.
        """
        if not self.polish_enable or self.polish_steps <= 0 or plc is None:
            return placement

        num_hard = benchmark.num_hard_macros
        num_all = benchmark.num_macros
        if num_all == 0 or num_hard == 0:
            return placement

        anchor = placement.detach().clone()
        span = benchmark.canvas_width + benchmark.canvas_height
        max_disp = self.polish_hard_tether_frac * span

        placement = placement.detach().clone()
        velocity = torch.zeros_like(placement)
        momentum = 0.85
        fixed = benchmark.macro_fixed
        all_sizes = benchmark.macro_sizes[:num_all]
        hw_all = all_sizes[:, 0] / 2
        hh_all = all_sizes[:, 1] / 2

        _set_placement(plc, placement.detach(), benchmark)
        start_metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
        start_proxy = float(start_metrics["proxy_cost"])
        best_placement = placement.clone()
        best_proxy = start_proxy

        if self.verbose:
            print(
                f"[polish] start: proxy={start_proxy:.4f} "
                f"wl={start_metrics['wirelength_cost']:.4f} "
                f"den={start_metrics['density_cost']:.4f} "
                f"cong={start_metrics['congestion_cost']:.4f} "
                f"tether=±{max_disp:.3f} ({self.polish_hard_tether_frac*100:.1f}% of span) "
                f"cong_w=(hard={self.polish_hard_cong_weight}, soft={self.polish_soft_cong_weight}) "
                f"(HV cong gradient)"
            )

        # HV infrastructure (replaces RUDY in cong gradient computation —
        # RUDY is isotropic and ignores macro routing blockage, so it was
        # pulling polish toward "low-RUDY but high-actual-TILOS-cong" bins
        # and breaking ibm18-style benches).
        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = max(1, ni_np.shape[0])
        sizes_np = benchmark.macro_sizes[:num_all].numpy().astype(np.float32).copy()
        bl = self._bin_left.numpy().copy()
        br = self._bin_right.numpy().copy()
        bb = self._bin_bottom.numpy().copy()
        bt = self._bin_top.numpy().copy()
        bin_w = float(self._bin_w)
        bin_h = float(self._bin_h)
        n_rows = benchmark.grid_rows
        n_cols = benchmark.grid_cols
        hcap = max(1e-6, bin_h * float(getattr(benchmark, "hroutes_per_micron", 1.0) or 1.0))
        vcap = max(1e-6, bin_w * float(getattr(benchmark, "vroutes_per_micron", 1.0) or 1.0))
        try:
            h_alloc, v_alloc = plc.get_macro_routing_allocation()
        except Exception:
            h_alloc = float(getattr(plc, "hrouting_alloc", 0.0) or 0.0)
            v_alloc = float(getattr(plc, "vrouting_alloc", 0.0) or 0.0)
        h_alloc = float(h_alloc)
        v_alloc = float(v_alloc)
        h_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        v_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        h_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        v_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        cong_forces = None  # refreshed periodically
        # Early-stop: bail after N consecutive checkpoints without improvement.
        # Polish has a tendency to drift away after the easy gains, burning
        # budget. Real-proxy guard reverts the final, but the loop still runs.
        polish_gate = ProgressGate(base_patience=4, bonus_per_gain=2, max_patience=10)
        polish_gate.best = best_proxy
        for step in range(self.polish_steps):
            placement = placement.detach()
            placement.requires_grad_(True)
            # Anneal alpha for sharper gradients near end (matches
            # parent's _settle_soft_macros pattern).
            alpha = 5.0 + 8.0 * (step / max(1, self.polish_steps))
            loss, _ = self._compute_wl_loss(
                placement, net_indices, net_mask, nets, canvas_norm,
                net_weights=net_weights, alpha=alpha,
            )
            loss.backward()
            wl_grad = placement.grad.detach().clone()
            placement.requires_grad_(False)
            placement = placement.detach()

            grid = self._compute_density_grid_fast(placement, benchmark)
            density_forces = self._compute_density_force_fast(
                self._solve_poisson(grid), placement, benchmark
            )

            # Periodically refresh the HV-based congestion gradient.
            # HV routing + macro blockage matches TILOS's actual cost
            # structure (RUDY did not), so polish now moves macros
            # toward bins that are genuinely less congested.
            if cong_forces is None or step % max(1, self.polish_cong_refresh) == 0:
                pos_np = placement.detach().numpy().astype(np.float32)
                h_route_grid[:] = 0
                v_route_grid[:] = 0
                h_macro_grid[:] = 0
                v_macro_grid[:] = 0
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos_np, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
                _build_macro_route_grid(
                    h_macro_grid, v_macro_grid, pos_np, sizes_np, num_hard,
                    bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h,
                    hcap, vcap, h_alloc, v_alloc,
                )
                # Combined HV cong map: H_total + V_total per bin (no
                # smoothing here — finite-diff cares about local slope,
                # not smoothed neighbors).
                cong_map = torch.from_numpy(
                    h_route_grid + v_route_grid + h_macro_grid + v_macro_grid
                )
                cong_forces = self._compute_congestion_force(
                    cong_map, placement, benchmark, num_all
                )
                # Normalize by max abs-magnitude → range [-1, 1].
                mag = cong_forces.abs().max().clamp_min(1e-6)
                cong_forces = cong_forces / mag

            hard_grad = (
                wl_grad[:num_hard]
                - self.polish_density_weight * density_forces[:num_hard]
                + self.polish_hard_cong_weight * cong_forces[:num_hard]
            )
            if fixed.any():
                hard_grad[fixed[:num_hard]] = 0.0
            soft_grad = (
                wl_grad[num_hard:]
                - self.polish_soft_density_weight * density_forces[num_hard:]
                + self.polish_soft_cong_weight * cong_forces[num_hard:]
            )

            velocity[:num_hard] = momentum * velocity[:num_hard] - self.polish_hard_lr * hard_grad
            velocity[num_hard:] = momentum * velocity[num_hard:] - self.polish_soft_lr * soft_grad
            placement = placement + velocity

            # Hard macro tether clamp (per-axis box around channel anchor).
            placement[:num_hard, 0] = torch.clamp(
                placement[:num_hard, 0],
                min=anchor[:num_hard, 0] - max_disp,
                max=anchor[:num_hard, 0] + max_disp,
            )
            placement[:num_hard, 1] = torch.clamp(
                placement[:num_hard, 1],
                min=anchor[:num_hard, 1] - max_disp,
                max=anchor[:num_hard, 1] + max_disp,
            )
            # Canvas bounds for everything.
            placement[:, 0] = torch.clamp(
                placement[:, 0], min=hw_all, max=benchmark.canvas_width - hw_all
            )
            placement[:, 1] = torch.clamp(
                placement[:, 1], min=hh_all, max=benchmark.canvas_height - hh_all
            )
            if fixed.any():
                placement[fixed] = benchmark.macro_positions[fixed]

            # Periodic checkpoint: legalize hard overlaps and verify real proxy.
            if (step + 1) % self.polish_checkpoint_every == 0:
                if self._hard_overlap_count(placement, benchmark) > 0:
                    placement = self._legalize_fast(
                        placement, benchmark, gap=0.01, max_iters=100
                    )
                    velocity[:num_hard] = 0.0
                _set_placement(plc, placement.detach(), benchmark)
                metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
                proxy = float(metrics["proxy_cost"])
                if self.verbose:
                    print(
                        f"[polish] step {step+1}/{self.polish_steps}: "
                        f"proxy={proxy:.4f} wl={metrics['wirelength_cost']:.4f} "
                        f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                        f"best={best_proxy:.4f}"
                    )
                improved, should_stop = polish_gate.update(proxy)
                if improved:
                    best_proxy = proxy
                    best_placement = placement.clone()
                elif should_stop:
                    if self.verbose:
                        print(
                            f"[polish] early-stop at step {step+1}: "
                            f"progress gate exhausted (best={polish_gate.best:.4f})"
                        )
                    break

        if self.verbose:
            print(
                f"[polish] done: proxy {start_proxy:.4f} → {best_proxy:.4f} "
                f"(delta={best_proxy - start_proxy:+.4f})"
            )
        return best_placement

    # ────────────────────────────────────────────────────────────────
    # Entry points
    # ────────────────────────────────────────────────────────────────

    def _apply_init_strategy(self, benchmark: Benchmark) -> Benchmark:
        """Replace benchmark.macro_positions per init_strategy. Returns the
        (possibly modified) benchmark. Does NOT mutate the original."""
        if self.init_strategy == "ibm":
            return benchmark  # use the default IBM 2004 floorplan as-is

        import copy
        bench = copy.copy(benchmark)
        bench.macro_positions = benchmark.macro_positions.clone()
        num_hard = benchmark.num_hard_macros
        num_all = benchmark.num_macros
        sizes = benchmark.macro_sizes
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height
        gen = torch.Generator().manual_seed(self.seed)

        if self.init_strategy == "spectral":
            from submissions.hybrid_analytical_placer import spectral_init_placement
            nets_list = [net.tolist() for net in benchmark.net_nodes if len(net) >= 2]
            bench.macro_positions = spectral_init_placement(
                bench.macro_positions, benchmark, nets_list, blend=0.7,
            )
        elif self.init_strategy == "perturbed":
            sigma_x = self.init_perturb_sigma * canvas_w
            sigma_y = self.init_perturb_sigma * canvas_h
            noise = torch.empty_like(bench.macro_positions)
            noise[:, 0].normal_(0.0, sigma_x, generator=gen)
            noise[:, 1].normal_(0.0, sigma_y, generator=gen)
            bench.macro_positions = bench.macro_positions + noise
        elif self.init_strategy == "random":
            n = num_all
            bench.macro_positions = torch.empty_like(bench.macro_positions)
            bench.macro_positions[:, 0].uniform_(0.0, canvas_w, generator=gen)
            bench.macro_positions[:, 1].uniform_(0.0, canvas_h, generator=gen)
        else:
            raise ValueError(f"unknown init_strategy: {self.init_strategy}")

        # Clamp to canvas bounds (per-macro size). Fixed macros stay put.
        bench.macro_positions[:, 0] = torch.clamp(
            bench.macro_positions[:, 0], min=hw, max=canvas_w - hw
        )
        bench.macro_positions[:, 1] = torch.clamp(
            bench.macro_positions[:, 1], min=hh, max=canvas_h - hh
        )
        if benchmark.macro_fixed.any():
            bench.macro_positions[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        if self.verbose:
            print(f"[init] applied strategy '{self.init_strategy}' (seed={self.seed})")
        return bench

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        benchmark = self._apply_init_strategy(benchmark)
        self._das_benchmark_ref = benchmark
        self._cached_nets = None
        self._cached_net_indices = None
        self._cached_net_mask = None
        self._cached_net_weights = None
        return super().place(benchmark)

    def _place_impl(self, benchmark: Benchmark) -> torch.Tensor:
        base = super()._place_impl(benchmark)
        if not self.ch_enable:
            return base

        plc = self._load_plc_for_logging(benchmark)
        nets = self._cached_nets or self._extract_nets(benchmark, plc)
        if (
            self._cached_net_indices is None
            or self._cached_net_mask is None
            or self._cached_net_weights is None
        ):
            net_indices, net_mask, net_weights = self._precompute_net_tensors(nets)
        else:
            net_indices = self._cached_net_indices
            net_mask = self._cached_net_mask
            net_weights = self._cached_net_weights
        canvas_norm = benchmark.canvas_width + benchmark.canvas_height

        refined = self._channel_relocate(
            base, benchmark, plc, nets,
            net_indices, net_mask, net_weights, canvas_norm,
            budget=self.ch_budget,
        )
        self._save_v2_stage(plc, benchmark, refined, "08_channel")

        # Soft channel relocate is opt-in. It measured as a tiny gain on
        # only two IBM benches, so default runs skip it.
        soft_relocated = refined
        if self.sch_enable:
            soft_relocated = self._soft_channel_relocate(
                refined, benchmark, plc, nets,
                net_indices, net_mask, net_weights, canvas_norm,
                budget=self.sch_budget,
            )
            self._save_v2_stage(plc, benchmark, soft_relocated, "09_soft_channel")

        # Post-channel polish is opt-in. The current incremental-real-eval
        # logs show it consistently restores the incoming checkpoint, so the
        # default submission path spends this time elsewhere.
        polished = soft_relocated
        if self.polish_enable:
            polished = self._post_channel_polish(
                soft_relocated, benchmark, plc, nets,
                net_indices, net_mask, net_weights, canvas_norm,
            )
            self._save_v2_stage(plc, benchmark, polished, "10_polish")

        # Safety: ensure no overlaps slipped through either stage.
        if self._hard_overlap_count(polished, benchmark) > 0:
            print("[v2] final v2 placement has overlaps, legalizing")
            polished = self._legalize_fast(polished, benchmark, gap=0.01, max_iters=400)
            if self._hard_overlap_count(polished, benchmark) > 0:
                polished = strong_legalize(polished, benchmark, gap=0.02, max_iters=80)
        return polished

    def _save_v2_stage(self, plc, benchmark: Benchmark, placement: torch.Tensor, label: str) -> None:
        if not self.enable_plots:
            return
        try:
            vis_dir = Path("vis") / benchmark.name
            vis_dir.mkdir(parents=True, exist_ok=True)
            from macro_place.utils import visualize_placement
            _set_placement(plc, placement.detach(), benchmark)
            visualize_placement(
                placement.detach(), benchmark,
                save_path=str(vis_dir / f"{label}.png"), plc=plc,
            )
            print(f"[vis] saved {vis_dir / f'{label}.png'}")
        except Exception as exc:
            print(f"[vis] failed to save {label}: {exc}")


def main():
    """Smoke test."""
    import sys
    from macro_place.loader import load_benchmark_from_dir

    bench_dir = "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    if len(sys.argv) > 1:
        bench_dir = sys.argv[1]
    print(f"Loading {bench_dir}...")
    benchmark, plc = load_benchmark_from_dir(bench_dir)
    print(f"  {benchmark}")

    placer = HybridAnalyticalPlacerV2(
        seed=42,
        num_steps=2000,
        verbose=True,
        enable_plots=False,
        das_enable=True,
        ch_enable=True,
        ch_budget=120.0,
    )
    placement = placer.place(benchmark)
    metrics = compute_proxy_cost(placement, benchmark, plc)
    print("Final proxy cost:")
    for k in ("proxy_cost", "wirelength_cost", "density_cost", "congestion_cost", "overlap_count"):
        print(f"  {k}={float(metrics[k]):.6f}")


if __name__ == "__main__":
    main()
