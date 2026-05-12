"""
Hybrid analytical placer built from the faster grad_place_benches flow.

Design goals:
1. Keep the staged optimization loop that already runs fast.
2. Stay CPU-first and submission-friendly.
3. Print exact progress stats every 1/10 of the run without spamming.
"""

import math
import cProfile
import pstats
import io
import random
import builtins
from pathlib import Path
from time import perf_counter as time
from typing import List, Optional, Tuple
import numpy as np
from numba import njit

import torch
from scipy.fft import dctn, idctn

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost


class ProgressGate:
    """
    Adaptive early-stop with momentum, for SA-style refinement stages.

    Each accepted real-proxy IMPROVEMENT adds `bonus` to a patience
    budget (capped at `max_patience`). Each non-improvement burns 1
    patience. Stop when patience hits 0.

    Behavior model:
      - Stage finds gains → momentum grows → more time granted
      - Gains dry up → patience drains, stage winds down
      - If gains return → patience refills automatically
      - Self-tuning vs fixed stale_checkpoints thresholds

    Usage:
        gate = ProgressGate(base_patience=4, bonus_per_gain=2, max_patience=12)
        ...
        improved, should_stop = gate.update(real_proxy)
        if improved:
            commit_new_best()
        elif should_stop:
            break
    """
    __slots__ = ("base_patience", "bonus", "max_patience", "patience", "best", "history")

    def __init__(self, base_patience: int = 4, bonus_per_gain: int = 2, max_patience: int = 12):
        self.base_patience = base_patience
        self.bonus = bonus_per_gain
        self.max_patience = max_patience
        self.patience = base_patience
        self.best = float("inf")
        self.history: List[Tuple[str, float, int]] = []

    def update(self, proxy: float) -> Tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if proxy < self.best - 1e-4:
            self.best = proxy
            self.patience = min(self.max_patience, self.patience + self.bonus)
            self.history.append(("+", proxy, self.patience))
            return True, False
        self.patience -= 1
        self.history.append((".", proxy, self.patience))
        return False, self.patience <= 0

    def __repr__(self) -> str:
        return (
            f"ProgressGate(patience={self.patience}/{self.max_patience}, "
            f"best={self.best:.4f}, history={len(self.history)})"
        )


def strong_legalize(placement, benchmark, gap=0.021, max_iters=200):
    """
    Push overlapping hard macros apart with minimum displacement.
    Phase 1: Vectorized bulk resolution (fast, handles most overlaps).
    Phase 2: Sequential cleanup (slow but guaranteed to converge).
    """
    placement = placement.clone()
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2

    # Precompute minimum separations: (num_hard, num_hard)
    sep_x = (sizes[:, 0].unsqueeze(1) + sizes[:, 0].unsqueeze(0)) / 2 + gap
    sep_y = (sizes[:, 1].unsqueeze(1) + sizes[:, 1].unsqueeze(0)) / 2 + gap
    tri_mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)

    # ── Phase 1: Vectorized (bulk) ──
    for iteration in range(min(max_iters, 500)):
        pos = placement[:num_hard]
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)

        overlap = (abs_dx < sep_x) & (abs_dy < sep_y) & tri_mask
        if not overlap.any():
            return placement

        ratio_x = abs_dx / sep_x
        ratio_y = abs_dy / sep_y
        push_x_axis = ratio_x > ratio_y

        push_amount_x = (sep_x - abs_dx) / 2 + gap
        push_amount_y = (sep_y - abs_dy) / 2 + gap
        sign_x = torch.sign(dx)
        sign_y = torch.sign(dy)

        push_amount_x = torch.where(
            overlap & push_x_axis, push_amount_x, torch.zeros_like(push_amount_x)
        )
        push_amount_y = torch.where(
            overlap & ~push_x_axis, push_amount_y, torch.zeros_like(push_amount_y)
        )
        sign_x = torch.where(overlap & push_x_axis, sign_x, torch.zeros_like(sign_x))
        sign_y = torch.where(overlap & ~push_x_axis, sign_y, torch.zeros_like(sign_y))

        force_x = (sign_x * push_amount_x).sum(dim=1) - (sign_x * push_amount_x).sum(dim=0)
        force_y = (sign_y * push_amount_y).sum(dim=1) - (sign_y * push_amount_y).sum(dim=0)

        num_overlaps = overlap.sum().float()
        damping = min(1.0, 1 / (num_overlaps.item() ** 0.5 + 1))
        placement[:num_hard, 0] += force_x * damping
        placement[:num_hard, 1] += force_y * damping

        placement[:num_hard, 0].clamp_(min=half_w, max=benchmark.canvas_width - half_w)
        placement[:num_hard, 1].clamp_(min=half_h, max=benchmark.canvas_height - half_h)

    # ── Phase 2: Sequential cleanup (guaranteed resolution) ──
    for iteration in range(max_iters):
        moved = False
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                dx_val = (placement[i, 0] - placement[j, 0]).item()
                dy_val = (placement[i, 1] - placement[j, 1]).item()
                sx = (sizes[i, 0] + sizes[j, 0]).item() / 2 + gap
                sy = (sizes[i, 1] + sizes[j, 1]).item() / 2 + gap

                if abs(dx_val) < sx and abs(dy_val) < sy:
                    if abs(dx_val) / sx > abs(dy_val) / sy:
                        push = (sx - abs(dx_val)) / 2 + gap
                        sign = 1.0 if dx_val >= 0 else -1.0
                        placement[i, 0] += push * sign
                        placement[j, 0] -= push * sign
                    else:
                        push = (sy - abs(dy_val)) / 2 + gap
                        sign = 1.0 if dy_val >= 0 else -1.0
                        placement[i, 1] += push * sign
                        placement[j, 1] -= push * sign
                    moved = True

        placement[:num_hard, 0].clamp_(min=half_w, max=benchmark.canvas_width - half_w)
        placement[:num_hard, 1].clamp_(min=half_h, max=benchmark.canvas_height - half_h)

        if not moved:
            break

    return placement


@njit
def _update_density_incr(
    grid,
    old_cx,
    old_cy,
    new_cx,
    new_cy,
    hw,
    hh,
    bl,
    br,
    bb,
    bt,
    bin_area,
    n_rows,
    n_cols,
    bin_w,
    bin_h,
):
    left = old_cx - hw
    right = old_cx + hw
    bottom = old_cy - hh
    top = old_cy + hh
    c0 = max(0, int(left / bin_w))
    c1 = min(n_cols - 1, int(right / bin_w))
    r0 = max(0, int(bottom / bin_h))
    r1 = min(n_rows - 1, int(top / bin_h))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0 and oy > 0:
                grid[r, c] -= ox * oy / bin_area
    left = new_cx - hw
    right = new_cx + hw
    bottom = new_cy - hh
    top = new_cy + hh
    c0 = max(0, int(left / bin_w))
    c1 = min(n_cols - 1, int(right / bin_w))
    r0 = max(0, int(bottom / bin_h))
    r1 = min(n_rows - 1, int(top / bin_h))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0 and oy > 0:
                grid[r, c] += ox * oy / bin_area

@njit
def _update_rudy_incr_single(rudy_grid, ni_np, nm_np, pos, affected_nets,
                              macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols):
    """Update RUDY grid for single macro displacement."""
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        xmin_old = np.float32(1e18)
        xmax_old = np.float32(-1e18)
        ymin_old = np.float32(1e18)
        ymax_old = np.float32(-1e18)
        xmin_new = np.float32(1e18)
        xmax_new = np.float32(-1e18)
        ymin_new = np.float32(1e18)
        ymax_new = np.float32(-1e18)

        for d in range(ni_np.shape[1]):
            if not nm_np[n, d]:
                break
            idx = ni_np[n, d]
            nx = pos[idx, 0]
            ny = pos[idx, 1]
            if nx < xmin_new: xmin_new = nx
            if nx > xmax_new: xmax_new = nx
            if ny < ymin_new: ymin_new = ny
            if ny > ymax_new: ymax_new = ny

            if idx == macro_i:
                ox, oy = old_x, old_y
            else:
                ox, oy = nx, ny
            if ox < xmin_old: xmin_old = ox
            if ox > xmax_old: xmax_old = ox
            if oy < ymin_old: ymin_old = oy
            if oy > ymax_old: ymax_old = oy

        old_area = (xmax_old - xmin_old + 1e-6) * (ymax_old - ymin_old + 1e-6)
        old_demand = min(np.float32(1.0) / old_area, np.float32(100.0))
        c0 = max(0, int(xmin_old / bin_w))
        c1 = min(n_cols - 1, int(xmax_old / bin_w))
        r0 = max(0, int(ymin_old / bin_h))
        r1 = min(n_rows - 1, int(ymax_old / bin_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rudy_grid[r, c] -= old_demand

        new_area = (xmax_new - xmin_new + 1e-6) * (ymax_new - ymin_new + 1e-6)
        new_demand = min(np.float32(1.0) / new_area, np.float32(100.0))
        c0 = max(0, int(xmin_new / bin_w))
        c1 = min(n_cols - 1, int(xmax_new / bin_w))
        r0 = max(0, int(ymin_new / bin_h))
        r1 = min(n_rows - 1, int(ymax_new / bin_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rudy_grid[r, c] += new_demand

@njit
def _density_cost_top5(grid):
    """
    TILOS-faithful density cost: 0.5 * mean(top 10% of grid cells).

    Function name is legacy (was buggy: used 5% without 0.5 multiplier).
    Now structurally matches plc.get_density_cost() exactly — fast surrogate
    becomes the real cost, eliminating calibration drift.
    See external/.../plc_client_os.py:1083-1109.
    """
    flat = grid.flatten()
    n = len(flat)
    k = max(1, int(n * 0.10))
    idx = np.argpartition(flat, -k)[-k:]
    return 0.5 * flat[idx].mean()


@njit
def _congestion_cost_top5(grid):
    flat = grid.flatten()
    n = len(flat)
    k = max(1, int(n * 0.05))
    idx = np.argpartition(flat, -k)[-k:]
    return flat[idx].mean()


@njit
def _build_density_grid(
    grid, pos, sizes, num_all, bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h
):
    for m in range(num_all):
        cx, cy = pos[m, 0], pos[m, 1]
        hw, hh = sizes[m, 0] / 2, sizes[m, 1] / 2
        left = cx - hw
        right = cx + hw
        bottom = cy - hh
        top = cy + hh
        c0 = max(0, int(left / bin_w))
        c1 = min(n_cols - 1, int(right / bin_w))
        r0 = max(0, int(bottom / bin_h))
        r1 = min(n_rows - 1, int(top / bin_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                ox = min(right, br[c]) - max(left, bl[c])
                oy = min(top, bt[r]) - max(bottom, bb[r])
                if ox > 0 and oy > 0:
                    grid[r, c] += ox * oy / bin_area


@njit
def _build_rudy_grid(grid, ni_np, nm_np, pos, num_nets, bin_w, bin_h, n_rows, n_cols):
    for n in range(num_nets):
        xmin = np.float32(1e18)
        xmax = np.float32(-1e18)
        ymin = np.float32(1e18)
        ymax = np.float32(-1e18)
        for d in range(ni_np.shape[1]):
            if not nm_np[n, d]:
                break
            idx = ni_np[n, d]
            x, y = pos[idx, 0], pos[idx, 1]
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        area = (xmax - xmin + 1e-6) * (ymax - ymin + 1e-6)
        demand = min(np.float32(1.0) / area, np.float32(100.0))
        c0 = max(0, int(xmin / bin_w))
        c1 = min(n_cols - 1, int(xmax / bin_w))
        r0 = max(0, int(ymin / bin_h))
        r1 = min(n_rows - 1, int(ymax / bin_h))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                grid[r, c] += demand


@njit
def _swap_creates_overlap(placement_np, i, j, sep_x_np, sep_y_np, num_hard):
    for idx in [i, j]:
        for k in range(num_hard):
            if k == i or k == j:
                continue
            dx = abs(placement_np[idx, 0] - placement_np[k, 0])
            dy = abs(placement_np[idx, 1] - placement_np[k, 1])
            if dx < sep_x_np[idx, k] and dy < sep_y_np[idx, k]:
                return True
    return False


@njit
def _hpwl_batch(net_idxs, ni_np, nm_np, pos):
    """Compute HPWL for a set of net indices."""
    result = np.zeros(len(net_idxs), dtype=np.float32)
    for k in range(len(net_idxs)):
        n = net_idxs[k]
        xmin = np.float32(1e18)
        xmax = np.float32(-1e18)
        ymin = np.float32(1e18)
        ymax = np.float32(-1e18)
        for d in range(ni_np.shape[1]):
            if not nm_np[n, d]:
                break
            idx = ni_np[n, d]
            x = pos[idx, 0]
            y = pos[idx, 1]
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        result[k] = (xmax - xmin) + (ymax - ymin)
    return result


# ──────────────────────────────────────────────────────────────────
# HV-separated routing congestion + macro routing blockage.
# Matches TILOS's congestion model far more accurately than isotropic
# RUDY: H/V tracked separately, smoothed across cols/rows, with hard
# macro blockage allocated by overlap area. Originally ported from hap2.
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
    """Per-bin pressure value = max(H_total, V_total) after smoothing."""
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


def spectral_init_placement(
    placement: torch.Tensor,
    benchmark,
    nets,
    blend: float = 0.7,
    max_net_size: int = 20,
) -> torch.Tensor:
    """
    Spectral initialization for macro placement.

    Args:
        placement: (N, 2) tensor
        benchmark: object with:
            - num_hard_macros
            - macro_fixed
            - canvas_width
            - canvas_height
        nets: List[List[int]] (macro connectivity)
        blend: how much to trust spectral coords (0.7 = strong)
        max_net_size: ignore huge nets (noise)

    Returns:
        Updated placement tensor
    """
    placement = placement.clone()
    device = placement.device

    num_hard = benchmark.num_hard_macros
    movable = ~benchmark.macro_fixed[:num_hard]
    movable_indices = torch.where(movable)[0]

    # --- Build adjacency ---
    A = torch.zeros((num_hard, num_hard), dtype=torch.float32)

    for net in nets:
        if len(net) < 2 or len(net) > max_net_size:
            continue

        w = 1.0 / (len(net) - 1)

        for i in range(len(net)):
            u = net[i]
            if u >= num_hard:
                continue
            for j in range(i + 1, len(net)):
                v = net[j]
                if v >= num_hard:
                    continue
                A[u, v] += w
                A[v, u] += w

    # --- Laplacian ---
    D = torch.diag(A.sum(dim=1))
    L = D - A

    # --- Eigen decomposition (CPU for stability) ---
    L_np = L.cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(L_np)

    # skip first eigenvector (constant)
    coords = torch.tensor(eigvecs[:, 1:3], dtype=torch.float32)

    # --- Add small noise to avoid collapse ---
    coords += 0.01 * torch.randn_like(coords)

    # --- Normalize to [0, 1] ---
    for d in range(2):
        min_v = coords[:, d].min()
        max_v = coords[:, d].max()
        coords[:, d] = (coords[:, d] - min_v) / (max_v - min_v + 1e-8)

    # --- Scale to canvas ---
    coords[:, 0] *= benchmark.canvas_width
    coords[:, 1] *= benchmark.canvas_height

    coords = coords.to(device)

    # --- Blend into placement (only movable macros) ---
    placement[movable_indices] = (
        (1 - blend) * placement[movable_indices]
        + blend * coords[movable_indices]
    )

    return placement

class HybridAnalyticalPlacer:
    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 50000,
        lr: float = 1.0,
        momentum: float = 0.9,
        soft_macro_lr: float = 0.15,
        verbose: bool = True,
        enable_plots: bool = True,
        # Differentiable density: smooth top-frac mean of bin densities,
        # added to WL loss as a single combined objective. Replaces
        # Poisson-based density forces (which optimized L2-spread, NOT
        # top-N percentile — causing Nesterov divergence on cong-bound
        # benches like ibm18 where density goes from 1.04 → 0.74 but
        # cong + WL both spike). Set use_smooth_density=False to fall
        # back to the old Poisson approach for A/B.
        # NOTE: default OFF after A/B showed it's bench-selective:
        # ibm01 -0.015 (helps), but ibm10 +0.053 / ibm14 +0.084 (hurts).
        # Smooth-density diverts Nesterov from the WL+density basin on
        # benches where density isn't the bottleneck. Keep as opt-in
        # for cong-heavy benches like ibm17/ibm18 if needed.
        use_smooth_density: bool = False,
        smooth_density_frac: float = 0.10,
        smooth_density_tau: float = 0.05,
        smooth_density_weight: float = 0.05,
        smooth_density_soft_scale: float = 10.0,
        # RePlAce routability inflation: per-macro effective size grows in
        # hot routing bins, biasing Nesterov to push them out → opens
        # routing channels → cong drops. Ported from hap2.
        # NOTE: default OFF after A/B showed marginal regression on
        # ibm14 (+0.003) and ibm18 (+0.004). Theory: doesn't compose well
        # with other downstream stages we have that didn't exist in hap2.
        # Keep as opt-in flag.
        use_routability_inflation: bool = False,
        inflation_update_every: int = 200,
        inflation_growth: float = 0.030,
        inflation_decay: float = 0.040,
        inflation_hard_cap: float = 1.28,
        inflation_soft_cap: float = 1.55,
    ):
        self.seed = seed
        self.num_steps = num_steps
        self.lr = lr
        self.momentum = momentum
        self.soft_macro_lr = soft_macro_lr
        self.verbose = verbose
        self.enable_plots = enable_plots
        self.use_smooth_density = use_smooth_density
        self.smooth_density_frac = smooth_density_frac
        self.smooth_density_tau = smooth_density_tau
        self.smooth_density_weight = smooth_density_weight
        self.smooth_density_soft_scale = smooth_density_soft_scale
        self.use_routability_inflation = use_routability_inflation
        self.inflation_update_every = inflation_update_every
        self.inflation_growth = inflation_growth
        self.inflation_decay = inflation_decay
        self.inflation_hard_cap = inflation_hard_cap
        self.inflation_soft_cap = inflation_soft_cap

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{benchmark.name}.log"
        original_print = builtins.print

        with log_path.open("w", encoding="utf-8") as log_file:
            def tee_print(*args, **kwargs):
                original_print(*args, **kwargs)
                log_kwargs = dict(kwargs)
                log_kwargs["file"] = log_file
                if log_kwargs.get("end") == "\r":
                    log_kwargs["end"] = "\n"
                log_kwargs.setdefault("flush", True)
                original_print(*args, **log_kwargs)

            builtins.print = tee_print
            try:
                return self._place_impl(benchmark)
            finally:
                builtins.print = original_print

    def _place_impl(self, benchmark: Benchmark) -> torch.Tensor:
        # pr = cProfile.Profile()
        # pr.enable()
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        placement = benchmark.macro_positions.clone().float()
        num_hard = benchmark.num_hard_macros
        num_all = benchmark.num_macros
        if num_all == 0:
            return placement

        plc = self._load_plc_for_logging(benchmark)
        _vis_dir = Path("vis") / benchmark.name

        def _save_plot(label: str, pos: torch.Tensor) -> None:
            if not self.enable_plots:
                return
            try:
                _vis_dir.mkdir(parents=True, exist_ok=True)
                from macro_place.utils import visualize_placement

                if plc is not None:
                    _set_placement(plc, pos.detach(), benchmark)
                visualize_placement(
                    pos.detach(), benchmark, save_path=str(_vis_dir / f"{label}.png"), plc=plc
                )
                print(f"[vis] saved {_vis_dir / f'{label}.png'}")
            except Exception as exc:
                print(f"[vis] failed to save {label}: {exc}")

        nets = self._extract_nets(benchmark, plc)
        if not nets:
            return placement

        net_indices, net_mask, net_weights = self._precompute_net_tensors(nets)
        # placement = self._make_initial_placement(placement, benchmark, nets, net_indices, net_mask)

        hard_sizes = benchmark.macro_sizes[:num_hard]
        hw_hard = hard_sizes[:, 0] / 2
        hh_hard = hard_sizes[:, 1] / 2

        # ── Precompute benchmark-fixed tensors ──
        _j = torch.arange(benchmark.grid_rows, dtype=torch.float32)
        _k = torch.arange(benchmark.grid_cols, dtype=torch.float32)
        _eig = (
            (2 * torch.cos(torch.pi * _j / benchmark.grid_rows)).unsqueeze(1)
            + (2 * torch.cos(torch.pi * _k / benchmark.grid_cols)).unsqueeze(0)
            - 4
        )
        _eig[0, 0] = 1.0
        self._poisson_eigenvalues = _eig
        self._bin_w = benchmark.canvas_width / benchmark.grid_cols
        self._bin_h = benchmark.canvas_height / benchmark.grid_rows
        self._bin_left = torch.arange(benchmark.grid_cols, dtype=torch.float32) * self._bin_w
        self._bin_right = self._bin_left + self._bin_w
        self._bin_bottom = torch.arange(benchmark.grid_rows, dtype=torch.float32) * self._bin_h
        self._bin_top = self._bin_bottom + self._bin_h
        self._leg_sep_x_base = (hard_sizes[:, 0].unsqueeze(1) + hard_sizes[:, 0].unsqueeze(0)) / 2
        self._leg_sep_y_base = (hard_sizes[:, 1].unsqueeze(1) + hard_sizes[:, 1].unsqueeze(0)) / 2
        self._leg_tri = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
        self._leg_half_w = hard_sizes[:, 0] / 2
        self._leg_half_h = hard_sizes[:, 1] / 2

        # ── Initial legalization ──
        placement = self._legalize_fast(placement, benchmark, gap=0.02, max_iters=400)
        overlaps = self._hard_overlap_count(placement, benchmark)
        print(
            f"Initial placement has {overlaps} overlaps among {num_hard} macros after fast legalization"
        )
        if overlaps > num_hard // 3:
            print(f"fast legal failed, applying strong legalization to fix {overlaps}")
            placement = strong_legalize(placement, benchmark, gap=0.02, max_iters=40)
            print("Done strong legalization of initial placement")

        all_sizes = benchmark.macro_sizes[:num_all]
        hw_all = all_sizes[:, 0] / 2
        hh_all = all_sizes[:, 1] / 2
        fixed = benchmark.macro_fixed

        velocity = torch.zeros_like(placement)
        base_lr = self.lr
        density_weight = 0.001
        canvas_norm = benchmark.canvas_width + benchmark.canvas_height
        log_every = max(1, self.num_steps // 20)
        track_every = max(1, self.num_steps // 50)
        plc_synced_at = -1
        K = 5
        recent_proxies = []
        top_k_candidates = []

        def sync_plc(step: int) -> None:
            nonlocal plc_synced_at
            if plc is None:
                return
            if plc_synced_at != step:
                _set_placement(plc, placement.detach(), benchmark)
                plc.FLAG_UPDATE_WIRELENGTH = False
                plc_synced_at = step

        self._log_stats("start", benchmark, placement, plc, wl=None, density_weight=density_weight)
        _save_plot("00_initial", placement)

        if plc is not None:
            _set_placement(plc, placement.detach(), benchmark)
            initial_proxy = compute_proxy_cost(placement.detach(), benchmark, plc)["proxy_cost"]
            top_k_candidates.append((initial_proxy, -1, placement.detach().clone()))

        start_time = time()
        total_time_budget = 1800
        hard_time_budget = 2400
        min_stage_budget = 120
        nesterov_time_budget = 240
        step = 0
        track_proxies = []
        print("starting iters")
        target_grid = None

        # ── HV-cong gradient for Nesterov ──
        # Parent's Nesterov historically optimized WL + density only. On
        # cong-bound benches (ibm17/18, cong=2.4) that's optimizing the
        # WRONG OBJECTIVE — Nesterov pushes macros toward layouts that
        # MAXIMIZE the dominant cost component, and downstream stages
        # can't undo the structural misdirection.
        # Adding HV-cong gradient as a third force pushes macros away
        # from already-congested bins during the analytical phase itself.
        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets_np = max(1, ni_np.shape[0])
        sizes_np = benchmark.macro_sizes[:num_all].numpy().astype(np.float32).copy()
        bl_np = self._bin_left.numpy().copy()
        br_np = self._bin_right.numpy().copy()
        bb_np = self._bin_bottom.numpy().copy()
        bt_np = self._bin_top.numpy().copy()
        bin_w_f = float(self._bin_w)
        bin_h_f = float(self._bin_h)
        n_rows_g = benchmark.grid_rows
        n_cols_g = benchmark.grid_cols
        hcap_f = max(1e-6, bin_h_f * float(getattr(benchmark, "hroutes_per_micron", 1.0) or 1.0))
        vcap_f = max(1e-6, bin_w_f * float(getattr(benchmark, "vroutes_per_micron", 1.0) or 1.0))
        try:
            h_alloc_f, v_alloc_f = plc.get_macro_routing_allocation() if plc is not None else (0.0, 0.0)
        except Exception:
            h_alloc_f = float(getattr(plc, "hrouting_alloc", 0.0) or 0.0)
            v_alloc_f = float(getattr(plc, "vrouting_alloc", 0.0) or 0.0)
        h_alloc_f = float(h_alloc_f)
        v_alloc_f = float(v_alloc_f)
        h_route_n = np.zeros((n_rows_g, n_cols_g), dtype=np.float32)
        v_route_n = np.zeros((n_rows_g, n_cols_g), dtype=np.float32)
        h_macro_n = np.zeros((n_rows_g, n_cols_g), dtype=np.float32)
        v_macro_n = np.zeros((n_rows_g, n_cols_g), dtype=np.float32)
        cong_forces_nesterov = None
        cong_refresh = 30
        # Weights chosen on the lower side — Nesterov gets many steps so
        # cong nudge compounds. Real-proxy guard via top-K checkpoints
        # catches regressions.
        cong_weight_hard = 0.0015
        cong_weight_soft = 0.004

        # RePlAce routability inflation: per-macro effective size scaled
        # by current routing-pressure at its bin. Updated periodically;
        # used in density grid computation to bias Nesterov gradient.
        macro_inflation = torch.ones(num_all, dtype=torch.float32)

        for step in range(self.num_steps):
            progress = step / self.num_steps
            current_lr = base_lr * (0.5 * (1 + math.cos(math.pi * progress)))
            current_lr = max(current_lr, base_lr * 0.05)

            # if step % 50 == 0:
            #     print(f"Step {step}/{self.num_steps} - Time elapsed: {time() - start_time:.1f}s", end="\r")
            if time() - start_time > nesterov_time_budget:
                print(f"Nesterov time budget reached at step {step}")
                break

            # # Periodic mid-run legalization
            if step % 200 == 0 and step > 0:
                if self._hard_overlap_count(placement, benchmark) > 0:
                    old_pos = placement[:num_hard].clone()
                    placement = self._legalize_fast(placement, benchmark, gap=0.01, max_iters=40)
                    moved = (placement[:num_hard] - old_pos).abs().sum(dim=1) > 1e-4
                    velocity[:num_hard][moved] = 0.0

            # 1. WL gradient at lookahead (+ optionally smooth-density loss
            # combined in the same backward pass so we get a structurally
            # correct gradient toward the TILOS-aligned objective).
            lookahead = placement.clone()
            lookahead[:num_hard] = placement[:num_hard] + self.momentum * velocity[:num_hard]
            lookahead.requires_grad_(True)
            wl_loss, wl = self._compute_wl_loss(
                lookahead, net_indices, net_mask, nets, canvas_norm, net_weights=net_weights
            )
            if self.use_smooth_density:
                # Replaces the Poisson-based L2-spread force with a smooth
                # top-frac mean of bin densities — matches the TILOS density
                # cost shape and lets autograd compute structurally correct
                # gradients (no more "minimize surrogate while real cost diverges").
                smooth_dens = self._compute_smooth_density_loss(
                    lookahead, benchmark,
                    frac=self.smooth_density_frac,
                    tau=self.smooth_density_tau,
                    hard_weight=1.0,
                    soft_weight=self.smooth_density_soft_scale,
                )
                total_loss = wl_loss + self.smooth_density_weight * smooth_dens
            else:
                total_loss = wl_loss
            total_loss.backward()
            wl_grad = lookahead.grad.detach().clone()
            lookahead.requires_grad_(False)

            # Build density grid with current per-macro inflation factors.
            # Inflated macros project a larger footprint into hot bins →
            # Poisson force pushes them out more aggressively.
            inflation_arg = macro_inflation if self.use_routability_inflation else 1.0
            grid = self._compute_density_grid_fast(placement, benchmark, inflation=inflation_arg)
            if step % 500 == 0:
                rudy = self._compute_rudy_map(placement, benchmark, nets, net_indices, net_mask)
                # Lower density target where congestion is high
                target_grid = grid.mean() - 0.1 * (rudy / rudy.max())
            # 2. Density forces at current position (LEGACY — only used when
            # use_smooth_density=False; otherwise the smooth-density gradient
            # is already baked into wl_grad above).
            if self.use_smooth_density:
                density_forces = torch.zeros_like(placement)
            else:
                density_forces = self._compute_density_force_fast(
                    self._solve_poisson(grid, target_grid), placement, benchmark
                )

            # Refresh inflation factors periodically.
            if (
                self.use_routability_inflation
                and plc is not None
                and step > 0
                and step % self.inflation_update_every == 0
            ):
                rudy_now = self._compute_rudy_map(
                    placement, benchmark, nets, net_indices, net_mask
                )
                rudy_norm = rudy_now / (rudy_now.mean() + 1e-6)
                macro_inflation = self._update_routability_inflation(
                    macro_inflation, placement, benchmark, rudy_norm, grid, step,
                    growth=self.inflation_growth,
                    decay=self.inflation_decay,
                    hard_cap=self.inflation_hard_cap,
                    soft_cap=self.inflation_soft_cap,
                )

            # 2b. HV-cong gradient (DISABLED — see comment above for revert
            # reason. Re-enable by uncommenting the block AND the two
            # hard_grad/soft_grad add lines below.)
            # if cong_forces_nesterov is None or step % cong_refresh == 0:
            #     pos_np = placement.detach().numpy().astype(np.float32)
            #     h_route_n[:] = 0; v_route_n[:] = 0; h_macro_n[:] = 0; v_macro_n[:] = 0
            #     _build_hv_route_grid(h_route_n, v_route_n, ni_np, nm_np, nw_np, pos_np, num_nets_np, bin_w_f, bin_h_f, n_rows_g, n_cols_g, hcap_f, vcap_f)
            #     _build_macro_route_grid(h_macro_n, v_macro_n, pos_np, sizes_np, num_hard, bl_np, br_np, bb_np, bt_np, n_rows_g, n_cols_g, bin_w_f, bin_h_f, hcap_f, vcap_f, h_alloc_f, v_alloc_f)
            #     cong_map_t = torch.from_numpy(h_route_n + v_route_n + h_macro_n + v_macro_n)
            #     cong_forces_nesterov = self._compute_congestion_force(cong_map_t, placement, benchmark, num_all)
            #     mag_c = cong_forces_nesterov.abs().max().clamp_min(1e-6)
            #     cong_forces_nesterov = cong_forces_nesterov / mag_c

            # 3. Adaptive density weight + congestion force update (every 20 steps)
            if step % 200 == 0 and plc is not None:
                sync_plc(step)
                tilos_den = plc.get_density_cost()
                if tilos_den > 0.9:
                    density_weight = min(0.005, density_weight * 1.05)
                elif tilos_den < 0.7:
                    density_weight = max(0.0001, density_weight * 0.95)

            # 4. Combined gradient
            hard_grad = wl_grad[:num_hard].clone()
            hard_grad -= density_weight * density_forces[:num_hard]
            # Cong gradient disabled — net-negative across tested benches
            # (ibm01 +0.014, ibm17 +0.026, ibm18 +0.003). Theory: cong push
            # diverts Nesterov from WL/density optimum without compounding
            # through downstream stages. Keep computation alive for
            # bench-adaptive re-enable later.
            # hard_grad += cong_weight_hard * cong_forces_nesterov[:num_hard]
            if fixed.any():
                hard_grad[fixed[:num_hard]] = 0.0

            # 5. Nesterov update
            velocity[:num_hard] = self.momentum * velocity[:num_hard] - current_lr * hard_grad
            placement[:num_hard] = (placement[:num_hard] + velocity[:num_hard]).clamp(
                min=torch.stack([hw_hard, hh_hard], dim=1),
                max=torch.stack(
                    [benchmark.canvas_width - hw_hard, benchmark.canvas_height - hh_hard], dim=1
                ),
            )

            # 6. Soft macros
            soft_density_weight = 0.01
            soft_grad = wl_grad[num_hard:num_all].clone()
            soft_grad -= soft_density_weight * density_forces[num_hard:num_all]
            # soft_grad += cong_weight_soft * cong_forces_nesterov[num_hard:num_all]
            placement[num_hard:num_all] -= self.soft_macro_lr * soft_grad
            placement[num_hard:num_all, 0].clamp_(
                min=hw_all[num_hard:], max=benchmark.canvas_width - hw_all[num_hard:]
            )
            placement[num_hard:num_all, 1].clamp_(
                min=hh_all[num_hard:], max=benchmark.canvas_height - hh_all[num_hard:]
            )

            if fixed.any():
                placement[fixed] = benchmark.macro_positions[fixed]

            # 7. Top-k tracking
            if (step + 1) % track_every == 0:
                if plc is not None:
                    sync_plc(step)
                    den_cost = plc.get_density_cost()
                    cong_cost = plc.get_congestion_cost()
                    proxy_est = wl.item() + 0.5 * den_cost + 0.5 * cong_cost
                else:
                    proxy_est = wl.item()
                top_k_candidates.append((proxy_est, step, placement.detach().clone()))
                top_k_candidates.sort(key=lambda x: x[0])
                top_k_candidates = top_k_candidates[:K]
                track_proxies.append(float(proxy_est))
                if len(track_proxies) >= 6 and all(
                    p > track_proxies[-6] for p in track_proxies[-5:]
                ):
                    print(f"Proxy diverging, stopping early at step {step}")
                    break

            # 8. Full logging
            if (step + 1) % log_every == 0 or step >= self.num_steps - 1:
                if plc is not None:
                    sync_plc(step)
                    metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
                    proxy_est = metrics["proxy_cost"]
                else:
                    metrics = None
                    proxy_est = wl.item()
                self._log_stats(
                    f"step_{step+1}",
                    benchmark,
                    placement,
                    plc,
                    wl=wl.item(),
                    density_weight=density_weight,
                    metrics=metrics,
                )
                recent_proxies.append(float(proxy_est))

            placement = placement.detach()

        # ── Post-optimization: legalize top-k and pick best ──
        _save_plot("01_nesterov", placement)
        best_valid_proxy = float("inf")
        best_valid_placement = None

        topk_deadline = start_time + max(120, total_time_budget - 5 * min_stage_budget)
        for proxy_est, ckpt_step, candidate in top_k_candidates:
            if time() > topk_deadline and best_valid_placement is not None:
                print("Top-K time cap reached, skipping remaining candidates")
                break
            c = candidate.clone()
            for i in range(8):
                if self._hard_overlap_count(c, benchmark) == 0:
                    break
                c = self._legalize_fast(c, benchmark, gap=0.01 * (i + 1), max_iters=200)
            if self._hard_overlap_count(c, benchmark) > 0:
                if time() < topk_deadline:
                    print("attempting strong legalize")
                    c = strong_legalize(c, benchmark, gap=0.01, max_iters=40)
                    print(f"strong legalize finished: {self._hard_overlap_count(c, benchmark)}")
                else:
                    print("strong legalize skipped (time cap), continuing")
                    continue
            if self._hard_overlap_count(c, benchmark) == 0:
                _set_placement(plc, c.detach(), benchmark)
                proxy = compute_proxy_cost(c, benchmark, plc)["proxy_cost"]
                print(f"legalized checkpoint from step {ckpt_step} has proxy cost {proxy}")
                if proxy < best_valid_proxy:
                    best_valid_proxy = proxy
                    best_valid_placement = c.clone()

        if best_valid_placement is None:
            if top_k_candidates:
                _, ckpt_step, candidate = min(top_k_candidates, key=lambda x: x[0])
                print(f"Top-K rescue: using unlegalized checkpoint from step {ckpt_step}")
                best_valid_placement = candidate.clone()
            else:
                best_valid_placement = placement.clone()
            best_valid_placement = self._legalize_fast(
                best_valid_placement, benchmark, gap=0.05, max_iters=500
            )

        def stage_budget(label: str, max_budget: float = 180) -> float:
            elapsed = time() - start_time
            hard_remaining = hard_time_budget - elapsed
            if hard_remaining < min_stage_budget:
                print(
                    f"{label} skipped: only {hard_remaining:.1f}s remains before hard budget "
                    f"(elapsed={elapsed:.1f}s)"
                )
                return 0.0
            return min(max_budget, hard_remaining)

        if best_valid_placement is not None and plc is not None:
            remaining = total_time_budget - (time() - start_time)
            swap_budget = stage_budget("SA swap", max_budget=240)
            print(
                f"SA swap budget: {swap_budget:.0f}s "
                f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
            )
            if swap_budget > 0:
                best_valid_placement = self._sa_refine(
                    best_valid_placement,
                    benchmark,
                    plc,
                    net_indices,
                    net_mask,
                    net_weights,
                    canvas_norm,
                    num_hard,
                    num_all,
                    fixed,
                    self._leg_sep_x_base.numpy().copy(),
                    self._leg_sep_y_base.numpy().copy(),
                    budget=swap_budget,
                    checkpoint_every=200,
                )
                _save_plot("02_sa_swap", best_valid_placement)
            else:
                print("SA swap skipped: no remaining budget")

            if benchmark.num_soft_macros > 1:
                remaining = total_time_budget - (time() - start_time)
                soft_swap_budget = stage_budget("SA soft swap", max_budget=180)
                print(
                    f"SA soft swap budget: {soft_swap_budget:.0f}s "
                    f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                )
                if soft_swap_budget > 0:
                    best_valid_placement = self._sa_soft_swap(
                        best_valid_placement,
                        benchmark,
                        plc,
                        net_indices,
                        net_mask,
                        net_weights,
                        canvas_norm,
                        num_hard,
                        num_all,
                        budget=soft_swap_budget,
                        checkpoint_every=100,
                    )
                    _save_plot("03_sa_soft_swap", best_valid_placement)
                else:
                    print("SA soft swap skipped: no remaining budget")

                remaining = total_time_budget - (time() - start_time)
                soft_spread_budget = stage_budget("Soft spread", max_budget=180)
                print(
                    f"Soft spread budget: {soft_spread_budget:.0f}s "
                    f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                )
                if soft_spread_budget > 0:
                    best_valid_placement = self._soft_spread_refine(
                        best_valid_placement,
                        benchmark,
                        plc,
                        net_indices,
                        net_mask,
                        net_weights,
                        canvas_norm,
                        num_hard,
                        num_all,
                        fixed,
                        budget=soft_spread_budget,
                        checkpoint_every=50,
                    )
                    _save_plot("04_soft_spread", best_valid_placement)
                else:
                    print("Soft spread skipped: no remaining budget")

                remaining = total_time_budget - (time() - start_time)
                soft_displace_budget = stage_budget("SA soft displace", max_budget=180)
                print(
                    f"SA soft displace budget: {soft_displace_budget:.0f}s "
                    f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                )
                if soft_displace_budget > 0:
                    best_valid_placement = self._sa_soft_displace(
                        best_valid_placement,
                        benchmark,
                        plc,
                        net_indices,
                        net_mask,
                        net_weights,
                        canvas_norm,
                        num_hard,
                        num_all,
                        fixed,
                        budget=soft_displace_budget,
                        checkpoint_every=200,
                    )
                    _save_plot("05_sa_soft_displace", best_valid_placement)
                else:
                    print("SA soft displace skipped: no remaining budget")

            remaining = total_time_budget - (time() - start_time)
            displace_budget = stage_budget("SA displace", max_budget=240)
            print(
                f"SA displace budget: {displace_budget:.0f}s "
                f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
            )
            if displace_budget > 0:
                best_valid_placement = self._sa_refine_displace(
                    best_valid_placement,
                    benchmark,
                    plc,
                    net_indices,
                    net_mask,
                    net_weights,
                    canvas_norm,
                    num_hard,
                    num_all,
                    fixed,
                    self._leg_sep_x_base.numpy().copy(),
                    self._leg_sep_y_base.numpy().copy(),
                    budget=displace_budget,
                    checkpoint_every=200,
                )
                _save_plot("06_sa_displace", best_valid_placement)
            else:
                print("SA displace skipped: no remaining budget")

            # best_valid_placement = self._cd_refine(
            #     best_valid_placement, benchmark, plc,
            #     net_indices, net_mask, num_hard, num_all, fixed,
            #     budget=180,
            # )


        # Re-settle soft macros after SA
        if best_valid_placement is not None:
            best_valid_placement = self._settle_soft_macros(
                best_valid_placement,
                net_indices,
                net_mask,
                nets,
                benchmark,
                num_hard,
                steps=80,
                lr=0.1,
            )
            _save_plot("07_final", best_valid_placement)

        final = best_valid_placement

        # Emergency legalize safety check
        if self._hard_overlap_count(final, benchmark) > 0:
            print("WARNING: final has overlaps, emergency legalize")
            final = strong_legalize(final, benchmark, gap=0.05, max_iters=200)

        self._log_stats("final", benchmark, final, plc, wl=None, density_weight=density_weight)
        # pr.disable()
        # s = io.StringIO()
        # pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
        # print(s.getvalue())
        return final

    def _compute_rudy_map(self, placement, benchmark, nets, net_indices, net_mask):
        rudy = torch.zeros(benchmark.grid_rows, benchmark.grid_cols)
        pos = placement.detach()
        for n in range(len(nets)):
            members = net_indices[n][net_mask[n]]
            xs = pos[members, 0]
            ys = pos[members, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Grid bins covered by this net's bbox
            col_lo = max(0, int(x_min / self._bin_w))
            col_hi = min(benchmark.grid_cols - 1, int(x_max / self._bin_w))
            row_lo = max(0, int(y_min / self._bin_h))
            row_hi = min(benchmark.grid_rows - 1, int(y_max / self._bin_h))

            area = (x_max - x_min + 1e-6) * (y_max - y_min + 1e-6)
            demand = 1.0 / area
            rudy[row_lo : row_hi + 1, col_lo : col_hi + 1] += demand

        return rudy

    def _sa_refine_displace(
            self,
            placement,
            benchmark,
            plc,
            net_indices,
            net_mask,
            net_weights,
            canvas_norm,
            num_hard,
            num_all,
            fixed,
            sep_x_np,
            sep_y_np,
            budget=600,
            checkpoint_every=200,
        ):
            """SA refinement using single-macro displacements with incremental proxy."""
            sa_placement = placement.clone()
            sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl = self._build_incremental_wl(
                net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
            )

            ni_np = net_indices.numpy().copy()
            nm_np = net_mask.numpy().copy()
            num_nets = ni_np.shape[0]
            sizes_np = benchmark.macro_sizes[:num_all].numpy().copy()
            hw_np = sizes_np[:, 0] / 2
            hh_np = sizes_np[:, 1] / 2

            bl = self._bin_left.numpy().copy()
            br = self._bin_right.numpy().copy()
            bb = self._bin_bottom.numpy().copy()
            bt = self._bin_top.numpy().copy()
            bin_w = float(self._bin_w)
            bin_h = float(self._bin_h)
            bin_area = bin_w * bin_h
            n_rows = benchmark.grid_rows
            n_cols = benchmark.grid_cols

            density_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_density_grid(density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h)

            # HV-separated routing congestion + macro blockage — matches TILOS
            # structurally (was RUDY before; calibration drifted because RUDY
            # is a fundamentally different metric than the real cong cost).
            nw_np = net_weights.numpy().copy()
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
            h_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            v_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            h_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            v_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )

            current_den = _density_cost_top5(density_grid)
            current_cong = _hv_congestion_cost_top5(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )

            # Calibrate fast proxy to match real proxy scale
            _set_placement(plc, sa_placement.detach(), benchmark)
            real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
            best_proxy = real_metrics["proxy_cost"]
            den_scale = real_metrics['density_cost'] / (current_den + 1e-8)
            cong_scale = real_metrics['congestion_cost'] / (current_cong + 1e-8)

            # Use scaled values for current_proxy
            current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)

            best_placement = sa_placement.clone()
            print(f"SA displace start: real={best_proxy:.4f} fast={current_proxy:.4f} wl={total_wl:.4f} den_scale={den_scale:.4f} cong_scale={cong_scale:.4f} (HV cong, smooth={smooth_range})")

            accepts = total = stalls = 0
            last_accept_step = 0
            t0 = time()
            max_displacement = canvas_norm * 0.03

            for _ in range(100_000_000):
                if time() - t0 > budget:
                    break

                if accepts > 0 and accepts % 500 == 0 and accepts % checkpoint_every != 0:
                    density_grid[:] = 0
                    _build_density_grid(density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h)
                    h_route_grid[:] = 0
                    v_route_grid[:] = 0
                    h_macro_grid[:] = 0
                    v_macro_grid[:] = 0
                    _build_hv_route_grid(h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                    _build_macro_route_grid(h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)
                    current_den = _density_cost_top5(density_grid)
                    current_cong = _hv_congestion_cost_top5(h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range)
                    current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)

                if total % 100000 == 0 and total > 0:
                    print(f"  step={total} accepts={accepts} wl={total_wl:.4f} fast={current_proxy:.4f} [{time()-t0:.0f}s]", end="\r")

                i = random.randint(0, num_hard - 1)
                if fixed[i].item():
                    continue

                total += 1

                old_x = float(sa_pos[i, 0])
                old_y = float(sa_pos[i, 1])

                dx = random.uniform(-max_displacement, max_displacement)
                dy = random.uniform(-max_displacement, max_displacement)
                sa_pos[i, 0] = np.clip(old_x + dx, hw_np[i], benchmark.canvas_width - hw_np[i])
                sa_pos[i, 1] = np.clip(old_y + dy, hh_np[i], benchmark.canvas_height - hh_np[i])

                has_overlap = False
                for k in range(num_hard):
                    if k == i:
                        continue
                    if (abs(sa_pos[i, 0] - sa_pos[k, 0]) < sep_x_np[i, k] and
                        abs(sa_pos[i, 1] - sa_pos[k, 1]) < sep_y_np[i, k]):
                        has_overlap = True
                        break

                if has_overlap:
                    sa_pos[i, 0] = old_x
                    sa_pos[i, 1] = old_y
                    continue

                aff = macro_to_nets[i]
                if len(aff) == 0:
                    sa_pos[i, 0] = old_x
                    sa_pos[i, 1] = old_y
                    continue

                old_hpwl = net_hpwl[aff].copy()
                new_hpwl = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
                wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)

                new_x = float(sa_pos[i, 0])
                new_y = float(sa_pos[i, 1])
                _update_density_incr(density_grid, old_x, old_y, new_x, new_y,
                                    hw_np[i], hh_np[i], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h)
                new_den = _density_cost_top5(density_grid)

                # Incremental HV routing update (nets touching i) +
                # macro blockage update (i moved).
                _update_hv_route_incr_single(h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, aff, i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                _update_macro_route_incr_single(h_macro_grid, v_macro_grid, old_x, old_y, new_x, new_y, hw_np[i], hh_np[i], bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)
                new_cong = _hv_congestion_cost_top5(h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range)

                new_proxy = (total_wl + wl_delta) + 0.5 * (new_den * den_scale) + .5 * (new_cong * cong_scale)

                if new_proxy < current_proxy:
                    sa_placement[i, 0] = float(sa_pos[i, 0])
                    sa_placement[i, 1] = float(sa_pos[i, 1])
                    net_hpwl[aff] = new_hpwl
                    total_wl += wl_delta
                    current_proxy = new_proxy
                    current_den = new_den
                    current_cong = new_cong
                    accepts += 1
                    last_accept_step = total

                    if accepts % checkpoint_every == 0:
                        _set_placement(plc, sa_placement.detach(), benchmark)
                        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                        real_proxy = real_metrics["proxy_cost"]
                        # Recalibrate surrogate at every checkpoint — the
                        # top-N percentile structure of density/cong means
                        # initial scales drift as the bin-leaderboard changes.
                        # Without this, the fast proxy lies more and more
                        # (saw gap widen from 0.04 → 0.08 on ibm18).
                        den_scale = real_metrics['density_cost'] / (current_den + 1e-8)
                        cong_scale = real_metrics['congestion_cost'] / (current_cong + 1e-8)
                        current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)
                        print(f"  step={total} accepts={accepts} wl={total_wl:.4f} fast={current_proxy:.4f} real={real_proxy:.4f} den_s={den_scale:.4f} cong_s={cong_scale:.4f} [{time()-t0:.0f}s]")
                        if real_proxy < best_proxy:
                            best_proxy = real_proxy
                            best_placement = sa_placement.clone()
                            stalls = 0
                        else:
                            sa_placement = best_placement.clone()
                            sa_pos[:] = best_placement.detach().numpy()
                            net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
                            total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                            density_grid[:] = 0
                            _build_density_grid(density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h)
                            h_route_grid[:] = 0
                            v_route_grid[:] = 0
                            h_macro_grid[:] = 0
                            v_macro_grid[:] = 0
                            _build_hv_route_grid(h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                            _build_macro_route_grid(h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)
                            current_den = _density_cost_top5(density_grid)
                            current_cong = _hv_congestion_cost_top5(h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range)
                            # After revert, components are now for best state.
                            # Refit scales to best so fast proxy is honest here too.
                            _set_placement(plc, sa_placement.detach(), benchmark)
                            best_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                            den_scale = best_metrics['density_cost'] / (current_den + 1e-8)
                            cong_scale = best_metrics['congestion_cost'] / (current_cong + 1e-8)
                            current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)
                            stalls += 1
                            if stalls >= 6:
                                print("SA displace stalled")
                                break
                else:
                    new_x = float(sa_pos[i, 0])
                    new_y = float(sa_pos[i, 1])
                    sa_pos[i, 0] = old_x
                    sa_pos[i, 1] = old_y
                    _update_density_incr(density_grid, new_x, new_y, old_x, old_y,
                                        hw_np[i], hh_np[i], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h)
                    # Reverse the HV updates: pass new_x/new_y as "old" so
                    # subtract uses new, add uses old (which is now in sa_pos).
                    _update_hv_route_incr_single(h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, aff, i, new_x, new_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                    _update_macro_route_incr_single(h_macro_grid, v_macro_grid, new_x, new_y, old_x, old_y, hw_np[i], hh_np[i], bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)

                if total - last_accept_step > 5_000_000:
                    print(f"SA displace: no accepts in 5M attempts, stopping")
                    break

            print(f"SA displace done: {total} attempts, {accepts} accepts, best real_proxy={best_proxy:.4f}")
            return best_placement
  
    def _cd_refine(self, placement, benchmark, plc, net_indices, net_mask,
               num_hard, num_all, fixed, budget=180):
        """
        Coordinate descent refinement using real density + congestion costs.
        For each macro, try positions along x and y axes, keep best.
        """
        placement = placement.clone()
        canvas_norm = benchmark.canvas_width + benchmark.canvas_height
        sizes = benchmark.macro_sizes[:num_hard]
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        sep_x = self._leg_sep_x_base
        sep_y = self._leg_sep_y_base

        _set_placement(plc, placement.detach(), benchmark)
        best_proxy = compute_proxy_cost(placement.detach(), benchmark, plc)["proxy_cost"]
        best_placement = placement.clone()
        print(f"CD start: proxy={best_proxy:.4f}")

        t0 = time()
        sweep = 0
        total_improved = 0
        n_positions = 8  # positions to try per axis

        while time() - t0 < budget:
            sweep += 1
            sweep_improved = 0
            order = list(range(num_hard))
            random.shuffle(order)

            for macro_idx in order:
                if time() - t0 > budget:
                    break
                if fixed[macro_idx].item():
                    continue

                orig_x = float(placement[macro_idx, 0])
                orig_y = float(placement[macro_idx, 1])
                best_x = orig_x
                best_y = orig_y
                best_cost = best_proxy

                # Generate candidate positions along x and y
                x_min = float(hw[macro_idx])
                x_max = float(benchmark.canvas_width - hw[macro_idx])
                y_min = float(hh[macro_idx])
                y_max = float(benchmark.canvas_height - hh[macro_idx])

                # Spread around current position
                spread_x = (x_max - x_min) * 0.1
                spread_y = (y_max - y_min) * 0.1

                candidates = []
                for dx in np.linspace(-spread_x, spread_x, n_positions):
                    cx = np.clip(orig_x + dx, x_min, x_max)
                    candidates.append((cx, orig_y))
                for dy in np.linspace(-spread_y, spread_y, n_positions):
                    cy = np.clip(orig_y + dy, y_min, y_max)
                    candidates.append((orig_x, cy))

                for cx, cy in candidates:
                    if cx == orig_x and cy == orig_y:
                        continue

                    # Check overlaps
                    placement[macro_idx, 0] = cx
                    placement[macro_idx, 1] = cy
                    has_overlap = False
                    for k in range(num_hard):
                        if k == macro_idx:
                            continue
                        if (abs(cx - float(placement[k, 0])) < float(sep_x[macro_idx, k]) and
                            abs(cy - float(placement[k, 1])) < float(sep_y[macro_idx, k])):
                            has_overlap = True
                            break

                    if has_overlap:
                        continue

                    # Evaluate with real costs
                    _set_placement(plc, placement.detach(), benchmark)
                    den = plc.get_density_cost()
                    cong = plc.get_congestion_cost()
                    wl_metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
                    cost = wl_metrics["proxy_cost"]

                    if cost < best_cost:
                        best_cost = cost
                        best_x = cx
                        best_y = cy

                # Apply best position
                placement[macro_idx, 0] = best_x
                placement[macro_idx, 1] = best_y

                if best_cost < best_proxy:
                    best_proxy = best_cost
                    best_placement = placement.clone()
                    sweep_improved += 1
                    total_improved += 1

            elapsed = time() - t0
            print(f"  CD sweep {sweep}: proxy={best_proxy:.4f} improved={sweep_improved} total={total_improved} [{elapsed:.0f}s]")

            if sweep_improved == 0:
                print("CD: no improvement this sweep, stopping")
                break

        print(f"CD done: {sweep} sweeps, {total_improved} improvements, proxy={best_proxy:.4f}")
        return best_placement

    def _sa_refine(
        self,
        placement,
        benchmark,
        plc,
        net_indices,
        net_mask,
        net_weights,
        canvas_norm,
        num_hard,
        num_all,
        fixed,
        sep_x_np,
        sep_y_np,
        budget=600,
        checkpoint_every=200,
    ):
        """
        SA refinement with incremental WL evaluation and checkpoint revert.
        Returns best placement found.
        """
        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        num_nets = ni_np.shape[0]

        _set_placement(plc, sa_placement.detach(), benchmark)
        best_proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
        best_placement = sa_placement.clone()
        print(f"SA start: proxy={best_proxy:.4f} wl={total_wl:.4f}")

        accepts = total = stalls = 0
        t0 = time()
        last_accept_step = 0
        for _ in range(100_000_000):
            if time() - t0 > budget:
                break
            if total % 100000 == 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )
            i = random.randint(0, num_hard - 1)
            j = random.randint(0, num_hard - 1)
            if i == j or fixed[i].item() or fixed[j].item():
                continue

            total += 1
            sa_pos[[i, j]] = sa_pos[[j, i]]

            if _swap_creates_overlap(sa_pos, i, j, sep_x_np, sep_y_np, num_hard):
                sa_pos[[i, j]] = sa_pos[[j, i]]
                continue

            delta, aff, new_vals = eval_delta(i, j)

            if delta <= 0:
                sa_placement[i], sa_placement[j] = sa_placement[j].clone(), sa_placement[i].clone()
                net_hpwl[aff] = new_vals
                total_wl += delta
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} proxy={proxy:.4f} [{time()-t0:.0f}s]"
                    )
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = sa_placement.clone()
                        stalls = 0
                    else:
                        # Revert to best known placement
                        sa_placement = best_placement.clone()
                        sa_pos[:] = best_placement.detach().numpy()
                        net_hpwl[:] = _hpwl_batch(
                            np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos
                        )
                        total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                        stalls += 1
                        if stalls >= 6:
                            print("SA stalled")
                            break

            else:
                sa_pos[[i, j]] = sa_pos[[j, i]]
            if total - last_accept_step > 5_000_000:
                print(f"SA: no accepts in 5M attempts, stopping")
                break

        print(f"SA done: {total} attempts, {accepts} accepts, best proxy={best_proxy:.4f}")
        return best_placement

    def _sa_soft_swap(
        self,
        placement,
        benchmark,
        plc,
        net_indices,
        net_mask,
        net_weights,
        canvas_norm,
        num_hard,
        num_all,
        budget=60,
        checkpoint_every=100,
    ):
        """Greedy pairwise soft macro swaps with real-proxy checkpoint/revert."""
        num_soft = num_all - num_hard
        if num_soft < 2:
            return placement

        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )
        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        num_nets = ni_np.shape[0]

        _set_placement(plc, sa_placement.detach(), benchmark)
        best_proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
        best_placement = sa_placement.clone()
        print(f"SA soft swap start: proxy={best_proxy:.4f} wl={total_wl:.4f} num_soft={num_soft}")

        accepts = total = stalls = 0
        last_accept_step = 0
        t0 = time()

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break
            if total % 100000 == 0 and total > 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            i = num_hard + random.randint(0, num_soft - 1)
            j = num_hard + random.randint(0, num_soft - 1)
            if i == j:
                continue

            total += 1
            sa_pos[[i, j]] = sa_pos[[j, i]]

            if len(macro_to_nets[i]) == 0 and len(macro_to_nets[j]) == 0:
                sa_pos[[i, j]] = sa_pos[[j, i]]
                continue

            delta, aff, new_vals = eval_delta(i, j)

            if delta <= 0:
                sa_placement[i], sa_placement[j] = sa_placement[j].clone(), sa_placement[i].clone()
                net_hpwl[aff] = new_vals
                total_wl += delta
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                        f"proxy={proxy:.4f} [{time()-t0:.0f}s]"
                    )
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = sa_placement.clone()
                        stalls = 0
                    else:
                        sa_placement = best_placement.clone()
                        sa_pos[:] = best_placement.detach().numpy()
                        net_hpwl[:] = _hpwl_batch(
                            np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos
                        )
                        total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                        stalls += 1
                        if stalls >= 10:
                            print("SA soft swap stalled")
                            break
            else:
                sa_pos[[i, j]] = sa_pos[[j, i]]

            if total - last_accept_step > 2_000_000:
                print("SA soft swap: no accepts in 2M attempts, stopping")
                break

        print(f"SA soft swap done: {total} attempts, {accepts} accepts, best proxy={best_proxy:.4f}")
        return best_placement

    def _soft_spread_refine(
        self,
        placement,
        benchmark,
        plc,
        net_indices,
        net_mask,
        net_weights,
        canvas_norm,
        num_hard,
        num_all,
        fixed,
        budget=70,
        checkpoint_every=150,
    ):
        """Move hot soft macros toward nearby low-pressure bins around their net centroid."""
        num_soft = num_all - num_hard
        if num_soft < 1:
            return placement

        spread_placement = placement.clone()
        pos, macro_to_nets, net_hpwl, _eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, spread_placement, num_all, canvas_norm
        )

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        num_nets = ni_np.shape[0]
        sizes_np = benchmark.macro_sizes[:num_all].numpy().copy()
        hw_np = sizes_np[:, 0] / 2
        hh_np = sizes_np[:, 1] / 2

        bl = self._bin_left.numpy().copy()
        br = self._bin_right.numpy().copy()
        bb = self._bin_bottom.numpy().copy()
        bt = self._bin_top.numpy().copy()
        bin_w = float(self._bin_w)
        bin_h = float(self._bin_h)
        bin_area = bin_w * bin_h
        n_rows = benchmark.grid_rows
        n_cols = benchmark.grid_cols

        def rebuild_maps():
            soft_density = np.zeros((n_rows, n_cols), dtype=np.float32)
            if num_soft > 0:
                _build_density_grid(
                    soft_density,
                    pos[num_hard:num_all],
                    sizes_np[num_hard:num_all],
                    num_soft,
                    bl,
                    br,
                    bb,
                    bt,
                    bin_area,
                    n_rows,
                    n_cols,
                    bin_w,
                    bin_h,
                )
            rudy_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_rudy_grid(rudy_grid, ni_np, nm_np, pos, num_nets, bin_w, bin_h, n_rows, n_cols)
            hot = soft_density + 0.25 * (rudy_grid / (rudy_grid.mean() + 1e-6))
            return soft_density, rudy_grid, hot

        def bin_value(grid, x, y):
            c = min(n_cols - 1, max(0, int(x / bin_w)))
            r = min(n_rows - 1, max(0, int(y / bin_h)))
            return float(grid[r, c])

        def connected_centroid(macro_idx):
            xs = []
            ys = []
            for net_idx in macro_to_nets[macro_idx]:
                for d in range(ni_np.shape[1]):
                    if not nm_np[net_idx, d]:
                        break
                    other = ni_np[net_idx, d]
                    if other == macro_idx:
                        continue
                    xs.append(pos[other, 0])
                    ys.append(pos[other, 1])
            if not xs:
                return float(pos[macro_idx, 0]), float(pos[macro_idx, 1])
            return float(np.mean(np.array(xs))), float(np.mean(np.array(ys)))

        _set_placement(plc, spread_placement.detach(), benchmark)
        best_proxy = compute_proxy_cost(spread_placement.detach(), benchmark, plc)["proxy_cost"]
        best_placement = spread_placement.clone()
        print(f"Soft spread start: proxy={best_proxy:.4f} wl={total_wl:.4f} num_soft={num_soft}")

        soft_density, rudy_grid, hot_grid = rebuild_maps()
        accepts = total = stalls = 0
        # Persists across retries: each revert widens search radius and
        # raises the uphill tolerance so retry batches generate genuinely
        # different proposals (without this the 50-tries-then-revert
        # pattern just re-explores the same neighborhood).
        stale_level = 0
        t0 = time()
        rebuild_every = 75
        checkpoint_accepts = 0
        soft_indices = np.arange(num_hard, num_all, dtype=np.int32)

        while time() - t0 < budget:
            soft_bins_r = np.clip((pos[soft_indices, 1] / bin_h).astype(np.int32), 0, n_rows - 1)
            soft_bins_c = np.clip((pos[soft_indices, 0] / bin_w).astype(np.int32), 0, n_cols - 1)
            pressure = hot_grid[soft_bins_r, soft_bins_c]
            order = soft_indices[np.argsort(-pressure)]

            moved_this_sweep = 0
            for i in order:
                if time() - t0 > budget:
                    break
                if fixed[i].item() or len(macro_to_nets[i]) == 0:
                    continue

                total += 1
                old_x = float(pos[i, 0])
                old_y = float(pos[i, 1])
                cx, cy = connected_centroid(i)
                old_hot = bin_value(hot_grid, old_x, old_y)
                old_hpwl = net_hpwl[macro_to_nets[i]].copy()

                best_score = 0.0
                best_xy = None
                best_new_hpwl = None
                elapsed_frac = min(1.0, (time() - t0) / max(budget, 1e-6))
                base_radius = canvas_norm * (0.025 * (1 - elapsed_frac) + 0.008 * elapsed_frac)
                max_wl_uphill = 0.00015 * (1 - elapsed_frac) + 0.00005 * elapsed_frac
                # Staleness widens the search and loosens uphill tolerance.
                base_radius *= 1.0 + 0.6 * stale_level
                max_wl_uphill *= 1.0 + 0.5 * stale_level

                candidates = []
                for radius_scale in (0.4, 0.8, 1.2):
                    radius = base_radius * radius_scale
                    for angle_idx in range(12):
                        angle = (2 * math.pi * angle_idx / 12) + random.uniform(-0.12, 0.12)
                        x = np.clip(
                            cx + math.cos(angle) * radius,
                            hw_np[i],
                            benchmark.canvas_width - hw_np[i],
                        )
                        y = np.clip(
                            cy + math.sin(angle) * radius,
                            hh_np[i],
                            benchmark.canvas_height - hh_np[i],
                        )
                        candidates.append((float(x), float(y)))

                for new_x, new_y in candidates:
                    new_hot = bin_value(hot_grid, new_x, new_y)
                    if new_hot >= old_hot and random.random() > 0.1:
                        continue
                    pos[i, 0] = new_x
                    pos[i, 1] = new_y
                    new_hpwl = _hpwl_batch(macro_to_nets[i], ni_np, nm_np, pos)
                    wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)
                    hot_drop = old_hot - new_hot
                    wl_safe = wl_delta <= 0.0
                    big_hot_drop = new_hot < old_hot * 0.80
                    if not (wl_safe or (big_hot_drop and wl_delta <= max_wl_uphill)):
                        continue
                    score = wl_delta - 0.0004 * hot_drop
                    if score < best_score:
                        best_score = score
                        best_xy = (new_x, new_y)
                        best_new_hpwl = new_hpwl.copy()

                if best_xy is None:
                    pos[i, 0] = old_x
                    pos[i, 1] = old_y
                    continue

                new_x, new_y = best_xy
                pos[i, 0] = new_x
                pos[i, 1] = new_y
                spread_placement[i, 0] = new_x
                spread_placement[i, 1] = new_y
                wl_delta = float((best_new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)
                net_hpwl[macro_to_nets[i]] = best_new_hpwl
                total_wl += wl_delta
                accepts += 1
                checkpoint_accepts += 1
                moved_this_sweep += 1

                if accepts % rebuild_every == 0:
                    soft_density, rudy_grid, hot_grid = rebuild_maps()

                if checkpoint_accepts >= checkpoint_every:
                    _set_placement(plc, spread_placement.detach(), benchmark)
                    proxy = compute_proxy_cost(spread_placement.detach(), benchmark, plc)["proxy_cost"]
                    print(
                        f"  spread accepts={accepts} tried={total} wl={total_wl:.4f} "
                        f"proxy={proxy:.4f} [{time()-t0:.0f}s]"
                    )
                    checkpoint_accepts = 0
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = spread_placement.clone()
                        stalls = 0
                        stale_level = 0
                    else:
                        spread_placement = best_placement.clone()
                        pos[:] = best_placement.detach().numpy()
                        net_hpwl[:] = _hpwl_batch(
                            np.arange(num_nets, dtype=np.int32), ni_np, nm_np, pos
                        )
                        total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                        soft_density, rudy_grid, hot_grid = rebuild_maps()
                        stalls += 1
                        stale_level = min(stale_level + 1, 5)
                        if stalls >= 8:
                            print("Soft spread stalled")
                            print(
                                f"Soft spread done: {total} tried, {accepts} accepts, "
                                f"best proxy={best_proxy:.4f}"
                            )
                            return best_placement

            if moved_this_sweep == 0:
                print("Soft spread: no moves accepted in sweep, stopping")
                break
            soft_density, rudy_grid, hot_grid = rebuild_maps()

        print(f"Soft spread done: {total} tried, {accepts} accepts, best proxy={best_proxy:.4f}")
        return best_placement

    def _sa_soft_displace(
        self,
        placement,
        benchmark,
        plc,
        net_indices,
        net_mask,
        net_weights,
        canvas_norm,
        num_hard,
        num_all,
        fixed,
        budget=60,
        checkpoint_every=200,
    ):
        """Greedy soft macro displacement with WL accept and real-proxy checkpoint/revert."""
        num_soft = num_all - num_hard
        if num_soft < 1:
            return placement

        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, _eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        num_nets = ni_np.shape[0]
        sizes_np = benchmark.macro_sizes[:num_all].numpy().copy()
        hw_np = sizes_np[:, 0] / 2
        hh_np = sizes_np[:, 1] / 2

        _set_placement(plc, sa_placement.detach(), benchmark)
        best_proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
        best_placement = sa_placement.clone()
        print(f"SA soft displace start: proxy={best_proxy:.4f} wl={total_wl:.4f} num_soft={num_soft}")

        accepts = total = stalls = 0
        last_accept_step = 0
        t0 = time()
        disp_start = canvas_norm * 0.01
        disp_end = canvas_norm * 0.002
        check_interval = 500_000
        last_check_accepts = 0
        min_rate = 5e-5

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break

            elapsed_frac = min(1.0, (time() - t0) / max(budget, 1e-6))
            max_displacement = disp_start * (1 - elapsed_frac) + disp_end * elapsed_frac
            if total % 100000 == 0 and total > 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                    f"disp={max_displacement/canvas_norm:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            if total % check_interval == 0 and total > 1_000_000:
                rate = (accepts - last_check_accepts) / check_interval
                if rate < min_rate:
                    print(
                        f"\nSA soft displace: accept rate {rate:.2e} < {min_rate:.2e}, stopping early"
                    )
                    break
                last_check_accepts = accepts

            i = num_hard + random.randint(0, num_soft - 1)
            if fixed[i].item() or len(macro_to_nets[i]) == 0:
                continue

            total += 1
            old_x = float(sa_pos[i, 0])
            old_y = float(sa_pos[i, 1])
            new_x = float(
                np.clip(
                    old_x + random.uniform(-max_displacement, max_displacement),
                    hw_np[i],
                    benchmark.canvas_width - hw_np[i],
                )
            )
            new_y = float(
                np.clip(
                    old_y + random.uniform(-max_displacement, max_displacement),
                    hh_np[i],
                    benchmark.canvas_height - hh_np[i],
                )
            )

            aff = macro_to_nets[i]
            old_hpwl = net_hpwl[aff].copy()
            sa_pos[i, 0] = new_x
            sa_pos[i, 1] = new_y
            new_hpwl = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
            wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)

            if wl_delta <= 0:
                sa_placement[i, 0] = new_x
                sa_placement[i, 1] = new_y
                net_hpwl[aff] = new_hpwl
                total_wl += wl_delta
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                        f"proxy={proxy:.4f} [{time()-t0:.0f}s]"
                    )
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = sa_placement.clone()
                        stalls = 0
                    else:
                        sa_placement = best_placement.clone()
                        sa_pos[:] = best_placement.detach().numpy()
                        net_hpwl[:] = _hpwl_batch(
                            np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos
                        )
                        total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                        stalls += 1
                        if stalls >= 6:
                            print("SA soft displace stalled")
                            break
            else:
                sa_pos[i, 0] = old_x
                sa_pos[i, 1] = old_y

            if total - last_accept_step > 2_000_000:
                print("SA soft displace: no accepts in 2M attempts, stopping")
                break

        print(
            f"SA soft displace done: {total} attempts, {accepts} accepts, "
            f"best proxy={best_proxy:.4f}"
        )
        return best_placement

    def _build_incremental_wl(
        self, net_indices, net_mask, net_weights, placement, num_all, canvas_norm
    ):
        """
        Build all data structures needed for incremental WL evaluation.
        Returns (sa_pos, macro_to_nets, net_hpwl, nw_np, eval_fn).
        """
        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = ni_np.shape[0]

        sa_pos = placement.detach().numpy().copy()

        # macro → nets inverted index
        macro_to_nets = [[] for _ in range(num_all)]
        for net_idx in range(num_nets):
            for d in range(ni_np.shape[1]):
                if not nm_np[net_idx, d]:
                    break
                macro_to_nets[ni_np[net_idx, d]].append(net_idx)
        macro_to_nets = [np.array(v, dtype=np.int32) for v in macro_to_nets]

        # warm up JIT
        _ = _hpwl_batch(np.array([0], dtype=np.int32), ni_np, nm_np, sa_pos)

        # initial per-net HPWL cache
        net_hpwl = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)

        def eval_delta(i, j):
            """
            Incremental WL delta assuming sa_pos already has swap(i,j) applied.
            Returns (delta_wl, affected_net_idxs, new_hpwl_vals).
            """
            aff = np.union1d(macro_to_nets[i], macro_to_nets[j])
            old_vals = net_hpwl[aff]
            new_vals = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
            # delta = float(((new_vals - old_vals) * nw_np[aff]).sum()) / (num_nets * canvas_norm)
            delta = float((new_vals - old_vals).sum()) / (num_nets * canvas_norm)
            return delta, aff, new_vals

        total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)

        return sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl

    def _compute_overlap_penalty(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        pos = placement[:num_hard]

        dx = torch.abs(pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0))
        dy = torch.abs(pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0))

        # penetration depth in each axis, 0 if no overlap
        overlap_x = torch.clamp(self._leg_sep_x_base - dx, min=0)
        overlap_y = torch.clamp(self._leg_sep_y_base - dy, min=0)

        # penalty is product of penetration depths — zero unless overlapping in both axes
        penalty = (overlap_x * overlap_y * self._leg_tri).sum()

        return penalty / (num_hard**2)  # normalize by macro count

    def _load_plc_for_logging(self, benchmark: Benchmark):
        try:
            test_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
            if test_dir.exists():
                _, plc = load_benchmark_from_dir(str(test_dir))
                return plc
            ng45_map = {
                "ariane133": "ariane133",
                "ariane136": "ariane136",
                "mempool_tile": "mempool_tile",
                "nvdla": "nvdla",
                "ariane133_ng45": "ariane133",
                "ariane136_ng45": "ariane136",
                "mempool_tile_ng45": "mempool_tile",
                "nvdla_ng45": "nvdla",
            }
            design = ng45_map.get(benchmark.name)
            if design is not None:
                base = (
                    Path("external/MacroPlacement/Flows/NanGate45")
                    / design
                    / "netlist"
                    / "output_CT_Grouping"
                )
                if (base / "netlist.pb.txt").exists():
                    _, plc = load_benchmark(
                        str(base / "netlist.pb.txt"),
                        str(base / "initial.plc"),
                        name=benchmark.name,
                    )
                    return plc
        except Exception:
            return None
        return None

    def _extract_nets(self, benchmark: Benchmark, plc) -> List[List[int]]:
        if plc is None:
            return [net.tolist() for net in benchmark.net_nodes if len(net) >= 2]
        name_to_idx = {}
        for ti, pi in enumerate(benchmark.hard_macro_indices):
            name_to_idx[plc.modules_w_pins[pi].get_name()] = ti
        for ti, pi in enumerate(benchmark.soft_macro_indices):
            name_to_idx[plc.modules_w_pins[pi].get_name()] = benchmark.num_hard_macros + ti
        seen = set()
        nets = []
        for driver, sinks in plc.nets.items():
            members = set()
            dm = driver.split("/")[0]
            if dm in name_to_idx:
                members.add(name_to_idx[dm])
            for sink in sinks:
                sm = sink.split("/")[0]
                if sm in name_to_idx:
                    members.add(name_to_idx[sm])
            if len(members) >= 2:
                key = frozenset(members)
                if key not in seen:
                    seen.add(key)
                    nets.append(list(members))
        return nets

    def _compute_congestion_force(self, cong_map, placement, benchmark, num_hard):
        # bin indices for all macros at once
        bin_x = (placement[:num_hard, 0] / self._bin_w).long().clamp(1, benchmark.grid_cols - 2)
        bin_y = (placement[:num_hard, 1] / self._bin_h).long().clamp(1, benchmark.grid_rows - 2)

        # finite difference gradient for all macros simultaneously
        dx = cong_map[bin_y, bin_x + 1] - cong_map[bin_y, bin_x - 1]
        dy = cong_map[bin_y + 1, bin_x] - cong_map[bin_y - 1, bin_x]

        forces = torch.stack([dx, dy], dim=1)
        return forces

    def _precompute_net_tensors(self, nets: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_degree = max(len(n) for n in nets)
        net_indices = torch.zeros(len(nets), max_degree, dtype=torch.long)
        net_mask = torch.zeros(len(nets), max_degree, dtype=torch.bool)
        for i, net in enumerate(nets):
            net_indices[i, : len(net)] = torch.tensor(net, dtype=torch.long)
            net_mask[i, : len(net)] = True

        degrees = torch.tensor([len(n) for n in nets], dtype=torch.float32)
        # upweight high fanout for congestion
        net_weights = torch.log2(degrees.clamp(min=2))
        net_weights /= net_weights.mean()
        return net_indices, net_mask, net_weights

    def _make_initial_placement(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        nets: List[List[int]],
        net_indices: torch.Tensor,
        net_mask: torch.Tensor,
    ) -> torch.Tensor:
        placement = placement.clone()
        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:num_hard]
        movable = ~benchmark.macro_fixed[:num_hard]
        movable_indices = torch.where(movable)[0].tolist()

        if movable_indices:
            grid_cols = max(1, int(math.sqrt(len(movable_indices))))
            spacing_x = benchmark.canvas_width / (grid_cols + 1)
            spacing_y = benchmark.canvas_height / (math.ceil(len(movable_indices) / grid_cols) + 1)
            order = sorted(
                movable_indices, key=lambda i: -float((sizes[i, 0] * sizes[i, 1]).item())
            )
            for count, idx in enumerate(order):
                row = count // grid_cols
                col = count % grid_cols
                target_x = spacing_x * (col + 1)
                target_y = spacing_y * (row + 1)
                placement[idx, 0] = 0.99 * placement[idx, 0] + 0.01 * target_x
                placement[idx, 1] = 0.99 * placement[idx, 1] + 0.01 * target_y

        placement = self._settle_soft_macros(
            placement, net_indices, net_mask, nets, benchmark, num_hard, steps=80, lr=0.1
        )
        return placement

    def _compute_wl_loss(
        self,
        placement: torch.Tensor,
        net_indices: torch.Tensor,
        net_mask: torch.Tensor,
        nets: List[List[int]],
        canvas_norm: float,
        net_weights=None,
        alpha=6.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_net = placement[net_indices]
        x = pos_net[:, :, 0]
        y = pos_net[:, :, 1]

        x_max = (1 / alpha) * torch.logsumexp(
            alpha * x.masked_fill(~net_mask, float("-inf")), dim=1
        )
        x_min = (-1 / alpha) * torch.logsumexp(
            -alpha * x.masked_fill(~net_mask, float("inf")), dim=1
        )
        y_max = (1 / alpha) * torch.logsumexp(
            alpha * y.masked_fill(~net_mask, float("-inf")), dim=1
        )
        y_min = (-1 / alpha) * torch.logsumexp(
            -alpha * y.masked_fill(~net_mask, float("inf")), dim=1
        )

        x_span = x_max - x_min
        y_span = y_max - y_min
        span = x_span + y_span

        if net_weights is not None:
            span = span * net_weights

        wl = span.sum() / (len(nets) * canvas_norm)

        return wl, wl

    def _settle_soft_macros(
        self,
        placement: torch.Tensor,
        net_indices: torch.Tensor,
        net_mask: torch.Tensor,
        nets: List[List[int]],
        benchmark: Benchmark,
        num_hard: int,
        steps: int = 80,
        lr: float = 0.1,
    ) -> torch.Tensor:
        num_all = num_hard + benchmark.num_soft_macros
        all_sizes = benchmark.macro_sizes[:num_all]
        hw = all_sizes[:, 0] / 2
        hh = all_sizes[:, 1] / 2
        canvas_norm = benchmark.canvas_width + benchmark.canvas_height
        placement = placement.clone()
        for _ in range(steps):
            placement.requires_grad_(True)
            alpha = 5.0 + 8.0 * (_ / steps)  # gradually increase alpha for sharper gradients
            loss, _ = self._compute_wl_loss(
                placement, net_indices, net_mask, nets, canvas_norm, net_weights=None, alpha=alpha
            )
            loss.backward()
            soft_grad = placement.grad.detach()[num_hard:num_all]
            placement.requires_grad_(False)
            placement.data[num_hard:num_all] -= lr * soft_grad
            placement.data[num_hard:num_all, 0].clamp_(
                min=hw[num_hard:], max=benchmark.canvas_width - hw[num_hard:]
            )
            placement.data[num_hard:num_all, 1].clamp_(
                min=hh[num_hard:], max=benchmark.canvas_height - hh[num_hard:]
            )
            placement = placement.detach()
        return placement

    def _compute_smooth_density_loss(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        frac: float = 0.10,
        tau: float = 0.05,
        hard_weight: float = 1.0,
        soft_weight: float = 10.0,
    ) -> torch.Tensor:
        """
        Differentiable smooth approximation of TILOS density cost.

        TILOS scores: 0.5 * mean(top frac × #bins). Replacing this with a
        sigmoid-gated weighted average that's differentiable in macro
        positions. Threshold is detached (gradient flows through values
        only). As tau → 0 this recovers the exact top-frac mean;
        practical tau ~ 0.05 keeps the gradient signal smooth.

        We weight HARD macro contributions to the grid differently from
        SOFT — soft macros have ~10× more headroom to spread.
        """
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        sizes = benchmark.macro_sizes[:num_macros]
        cx = placement[:num_macros, 0]
        cy = placement[:num_macros, 1]
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        left = cx - half_w
        right = cx + half_w
        bottom = cy - half_h
        top = cy + half_h
        overlap_x = torch.clamp(
            torch.min(right.unsqueeze(1), self._bin_right.unsqueeze(0))
            - torch.max(left.unsqueeze(1), self._bin_left.unsqueeze(0)),
            min=0,
        )
        overlap_y = torch.clamp(
            torch.min(top.unsqueeze(1), self._bin_top.unsqueeze(0))
            - torch.max(bottom.unsqueeze(1), self._bin_bottom.unsqueeze(0)),
            min=0,
        )
        # Apply per-macro weight to its grid contribution.
        per_macro_w = torch.ones(num_macros, dtype=placement.dtype, device=placement.device)
        per_macro_w[num_hard:] = soft_weight
        per_macro_w[:num_hard] = hard_weight
        # Weight along the macro axis: contribution = w_m * overlap_y[m] outer overlap_x[m]
        # We compute via element-wise scale before the mm.
        oy_w = overlap_y * per_macro_w.unsqueeze(1)
        grid = torch.mm(oy_w.t(), overlap_x) / (self._bin_w * self._bin_h)
        grid_flat = grid.flatten()
        threshold = torch.quantile(grid_flat.detach(), 1.0 - frac)
        weights = torch.sigmoid((grid_flat - threshold) / tau)
        soft_top_mean = (weights * grid_flat).sum() / (weights.sum() + 1e-8)
        return 0.5 * soft_top_mean

    def _compute_density_grid_fast(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        inflation=1.0,
    ) -> torch.Tensor:
        """
        Bin-overlap density grid. `inflation` may be a scalar (applied to hard
        macros only) OR a per-macro 1D tensor of size num_macros (each macro's
        effective size scaled by its own factor). The per-macro form is what
        the RePlAce-style routability inflation uses to push macros out of
        congested bins.
        """
        num_macros = benchmark.num_macros
        sizes = benchmark.macro_sizes[:num_macros]

        if torch.is_tensor(inflation):
            if inflation.numel() != num_macros:
                raise ValueError(
                    f"per-macro inflation tensor size {inflation.numel()} != num_macros {num_macros}"
                )
            sizes = sizes * inflation.unsqueeze(1)
        elif inflation != 1.0:
            inflated = sizes.clone()
            inflated[: benchmark.num_hard_macros] *= inflation
            sizes = inflated

        cx = placement[:num_macros, 0]
        cy = placement[:num_macros, 1]
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        left = cx - half_w
        right = cx + half_w
        bottom = cy - half_h
        top = cy + half_h

        overlap_x = torch.clamp(
            torch.min(right.unsqueeze(1), self._bin_right.unsqueeze(0))
            - torch.max(left.unsqueeze(1), self._bin_left.unsqueeze(0)),
            min=0,
        )
        overlap_y = torch.clamp(
            torch.min(top.unsqueeze(1), self._bin_top.unsqueeze(0))
            - torch.max(bottom.unsqueeze(1), self._bin_bottom.unsqueeze(0)),
            min=0,
        )
        return torch.mm(overlap_y.t(), overlap_x) / (self._bin_w * self._bin_h)

    def _update_routability_inflation(
        self,
        macro_inflation: torch.Tensor,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rudy_norm: torch.Tensor,
        density_grid: torch.Tensor,
        step: int,
        growth: float = 0.030,
        decay: float = 0.040,
        hard_cap: float = 1.28,
        soft_cap: float = 1.55,
        growth_until: int = 3000,
    ) -> torch.Tensor:
        """
        RePlAce-style routability inflation (Cheng et al., TCAD 2018; ported
        from hap2 where it contributed to ~1.315 overall).

        Macros sitting in high routing-pressure bins get their EFFECTIVE
        density size grown (multiplicative, with cap). This biases Nesterov's
        density gradient to push them OUT of those bins → opens routing
        channels → cong drops. Decay shrinks inflation when pressure clears
        so macros aren't artificially large forever.

        Behavior:
          - Grow phase (step <= growth_until): hot bins → inflate, cool bins → decay
          - Decay-only phase (step > growth_until): never grow, just shrink
          - Hard macros capped tighter (1.28×) than soft (1.55×) since soft is
            the routing sea — the user's observed problem
          - Fixed macros never inflate
        """
        num_all = benchmark.num_macros
        num_hard = benchmark.num_hard_macros
        if num_all == 0:
            return macro_inflation

        density_norm = density_grid / (density_grid.mean() + 1e-6)
        pressure_grid = rudy_norm.clamp(max=8.0) + 0.35 * density_norm.clamp(max=8.0)
        hot = torch.quantile(pressure_grid.flatten(), 0.82).clamp(min=1.05)
        very_hot = torch.quantile(pressure_grid.flatten(), 0.94).clamp(min=hot + 1e-4)

        cx = placement[:num_all, 0].detach()
        cy = placement[:num_all, 1].detach()
        c_bins = (cx / self._bin_w).long().clamp(0, benchmark.grid_cols - 1)
        r_bins = (cy / self._bin_h).long().clamp(0, benchmark.grid_rows - 1)
        macro_pressure = pressure_grid[r_bins, c_bins]

        new_inflation = macro_inflation.detach().clone()
        hot_mask = macro_pressure > hot
        allow_growth = step <= growth_until
        if hot_mask.any() and allow_growth:
            severity = ((macro_pressure[hot_mask] - hot) / (very_hot - hot + 1e-6)).clamp(0.0, 2.0)
            multiplier = 1.0 + growth * (0.7 + severity)
            new_inflation[hot_mask] *= multiplier
        elif hot_mask.any():
            # Decay-only phase: even hot bins shrink, just more slowly
            new_inflation[hot_mask] = 1.0 + (new_inflation[hot_mask] - 1.0) * (1.0 - decay * 0.45)

        cool_mask = ~hot_mask if allow_growth else torch.ones_like(hot_mask, dtype=torch.bool)
        if cool_mask.any():
            new_inflation[cool_mask] = 1.0 + (new_inflation[cool_mask] - 1.0) * (1.0 - decay)

        if num_hard > 0:
            new_inflation[:num_hard].clamp_(1.0, hard_cap)
        if num_all > num_hard:
            # Soft clusters are the routing sea; give them more room to spread.
            soft_hot = hot_mask[num_hard:num_all]
            soft_inflation = new_inflation[num_hard:num_all].clone()
            if soft_hot.any() and allow_growth:
                soft_inflation[soft_hot] *= 1.0 + growth * 0.35
            new_inflation[num_hard:num_all] = soft_inflation.clamp(1.0, soft_cap)

        if benchmark.macro_fixed.any():
            new_inflation[benchmark.macro_fixed[:num_all]] = 1.0

        if self.verbose and step % 600 == 0:
            print(
                f"  [inflate {step:5d}] hot={int(hot_mask.sum().item())}/{num_all} "
                f"mean={new_inflation.mean().item():.3f} "
                f"hard_max={new_inflation[:num_hard].max().item() if num_hard else 1.0:.2f} "
                f"soft_max={new_inflation[num_hard:num_all].max().item() if num_all > num_hard else 1.0:.2f} "
                f"growth={'on' if allow_growth else 'off'}"
            )
        return new_inflation.detach()

    def _solve_poisson(self, density_grid: torch.Tensor, target_grid=None) -> torch.Tensor:
        if target_grid is None:
            rho = density_grid - density_grid.mean()
        else:
            rho = density_grid - target_grid
        rho_freq = torch.tensor(dctn(rho.numpy()), dtype=torch.float32)
        potential = torch.tensor(
            idctn((rho_freq / self._poisson_eigenvalues).numpy()), dtype=torch.float32
        )
        return -potential

    def _compute_density_force_fast(
        self, potential: torch.Tensor, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        num_all = benchmark.num_macros

        grad_x = torch.zeros_like(potential)
        grad_y = torch.zeros_like(potential)
        grad_x[:, 1:-1] = (potential[:, 2:] - potential[:, :-2]) / (2 * self._bin_w)
        grad_y[1:-1, :] = (potential[2:, :] - potential[:-2, :]) / (2 * self._bin_h)

        cx = placement[:num_all, 0].detach()
        cy = placement[:num_all, 1].detach()
        c_bins = (cx / self._bin_w).long().clamp(0, benchmark.grid_cols - 1)
        r_bins = (cy / self._bin_h).long().clamp(0, benchmark.grid_rows - 1)

        forces = torch.zeros(num_all, 2)
        forces[:, 0] = -grad_x[r_bins, c_bins]
        forces[:, 1] = -grad_y[r_bins, c_bins]
        return forces

    def _legalize_fast(
        self, placement: torch.Tensor, benchmark: Benchmark, gap: float = 0.01, max_iters: int = 500
    ) -> torch.Tensor:
        placement = placement.clone()
        num_hard = benchmark.num_hard_macros
        sep_x = self._leg_sep_x_base + gap
        sep_y = self._leg_sep_y_base + gap
        tri = self._leg_tri
        half_w = self._leg_half_w
        half_h = self._leg_half_h

        for _ in range(max_iters):
            pos = placement[:num_hard]
            dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
            dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
            abs_dx = torch.abs(dx)
            abs_dy = torch.abs(dy)
            overlap_mask = (abs_dx < sep_x) & (abs_dy < sep_y) & tri
            if not overlap_mask.any():
                break
            pairs = overlap_mask.nonzero(as_tuple=False)
            pair_areas = (
                benchmark.macro_sizes[pairs[:, 0], 0] * benchmark.macro_sizes[pairs[:, 0], 1]
                + benchmark.macro_sizes[pairs[:, 1], 0] * benchmark.macro_sizes[pairs[:, 1], 1]
            )
            pairs = pairs[pair_areas.argsort(descending=True)]
            for p in range(pairs.shape[0]):
                i = pairs[p, 0].item()
                j = pairs[p, 1].item()
                dx_val = placement[i, 0] - placement[j, 0]
                dy_val = placement[i, 1] - placement[j, 1]
                sx = sep_x[i, j].item()
                sy = sep_y[i, j].item()
                if abs(dx_val) < sx and abs(dy_val) < sy:
                    if abs(dx_val) / sx > abs(dy_val) / sy:
                        push = (sx - abs(dx_val)) / 2 + gap
                        sign = 1.0 if dx_val >= 0 else -1.0
                        placement[i, 0] += push * sign
                        placement[j, 0] -= push * sign
                    else:
                        push = (sy - abs(dy_val)) / 2 + gap
                        sign = 1.0 if dy_val >= 0 else -1.0
                        placement[i, 1] += push * sign
                        placement[j, 1] -= push * sign
            placement[:num_hard, 0].clamp_(min=half_w, max=benchmark.canvas_width - half_w)
            placement[:num_hard, 1].clamp_(min=half_h, max=benchmark.canvas_height - half_h)
        return placement

    def _hard_overlap_count(self, placement: torch.Tensor, benchmark: Benchmark) -> int:
        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return 0
        p = placement[:num_hard]
        dx = torch.abs(p.unsqueeze(0)[:, :, 0] - p.unsqueeze(1)[:, :, 0])
        dy = torch.abs(p.unsqueeze(0)[:, :, 1] - p.unsqueeze(1)[:, :, 1])
        return int(
            ((dx < self._leg_sep_x_base) & (dy < self._leg_sep_y_base) & self._leg_tri).sum().item()
        )

    def _log_stats(
        self,
        label: str,
        benchmark: Benchmark,
        placement: torch.Tensor,
        plc,
        wl: Optional[float],
        density_weight: float,
        metrics: Optional[dict] = None,
    ) -> None:
        if not self.verbose:
            return
        overlaps = self._hard_overlap_count(placement, benchmark)
        if plc is not None:
            if metrics is None:
                metrics = compute_proxy_cost(placement, benchmark, plc)
            wl_value = wl if wl is not None else metrics["wirelength_cost"]
            print(
                f"[{benchmark.name}] {label:<12} "
                f"proxy={metrics['proxy_cost']:.4f} "
                f"wl={wl_value:.4f} "
                f"dens={metrics['density_cost']:.4f} "
                f"cong={metrics['congestion_cost']:.4f} "
                f"ovlp={metrics['overlap_count']} "
                f"dw={density_weight:.4f}",
                flush=True,
            )
        else:
            print(
                f"[{benchmark.name}] {label:<12} wl={float(wl) if wl is not None else float('nan'):.4f} "
                f"ovlp={overlaps} dw={density_weight:.4f}",
                flush=True,
            )
