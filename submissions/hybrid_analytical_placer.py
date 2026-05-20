"""
Hybrid analytical placer built from the faster grad_place_benches flow.

Design goals:
1. Keep the staged optimization loop that already runs fast.
2. Stay CPU-first and submission-friendly.
3. Print exact progress stats every 1/10 of the run without spamming.
"""

import math
import cProfile
import pickle
import pstats
import io
import random
import builtins
import importlib.util
import os
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

_IRE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "incremental_real_eval.py")
_ire_spec = importlib.util.spec_from_file_location("_hap_incremental_real_eval_parent", _IRE_PATH)
_ire_mod = importlib.util.module_from_spec(_ire_spec)
_ire_spec.loader.exec_module(_ire_mod)
SmoothHVCostTracker = _ire_mod.SmoothHVCostTracker
update_hv_route_incr_single_smooth = _ire_mod.update_hv_route_incr_single_smooth
update_pin_hv_route_incr_single_smooth = _ire_mod.update_pin_hv_route_incr_single_smooth


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


class WindowProgressGate:
    """
    Stop a noisy stage only after whole windows fail to improve best-so-far.

    Raw checkpoints can bounce around; this gate looks at the best value seen
    inside a window and asks whether it beat the window anchor by `epsilon`.
    """
    __slots__ = (
        "window",
        "epsilon",
        "min_time",
        "max_patience",
        "patience",
        "best",
        "anchor_best",
        "seen",
        "history",
    )

    def __init__(
        self,
        window: int = 5,
        patience_windows: int = 2,
        epsilon: float = 0.003,
        min_time: float = 0.0,
        initial_best: float = float("inf"),
    ):
        self.window = max(1, int(window))
        self.epsilon = float(epsilon)
        self.min_time = max(0.0, float(min_time))
        self.max_patience = max(1, int(patience_windows))
        self.patience = self.max_patience
        self.best = float(initial_best)
        self.anchor_best = float(initial_best)
        self.seen = 0
        self.history: List[Tuple[str, float, int]] = []

    def update(self, proxy: float, elapsed: Optional[float] = None) -> Tuple[bool, bool]:
        improved = proxy < self.best - 1e-4
        if improved:
            self.best = float(proxy)

        self.seen += 1
        if self.seen < self.window:
            self.history.append(("?", float(proxy), self.patience))
            return improved, False

        gain = self.anchor_best - self.best
        elapsed_ok = elapsed is None or elapsed >= self.min_time
        if gain >= self.epsilon:
            self.patience = self.max_patience
            mark = "+"
        else:
            mark = "."
            if elapsed_ok:
                self.patience -= 1

        self.anchor_best = self.best
        self.seen = 0
        self.history.append((mark, float(proxy), self.patience))
        return improved, elapsed_ok and self.patience <= 0

    def __repr__(self) -> str:
        return (
            f"WindowProgressGate(patience={self.patience}/{self.max_patience}, "
            f"best={self.best:.4f}, anchor={self.anchor_best:.4f}, seen={self.seen}/{self.window})"
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


@njit
def _hpwl_candidate_batch_for_macro(cand_x, cand_y, macro_i, net_idxs, ni_np, nm_np, pos):
    """Compute HPWL of affected nets for many candidate positions of one macro."""
    out = np.zeros((len(cand_x), len(net_idxs)), dtype=np.float32)
    for ci in range(len(cand_x)):
        mx = cand_x[ci]
        my = cand_y[ci]
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
                if idx == macro_i:
                    x = mx
                    y = my
                else:
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
            out[ci, k] = (xmax - xmin) + (ymax - ymin)
    return out


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
def _add_net_route_pin_hv(
    h_grid, v_grid, net_idx,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
    macro_i, override_x, override_y,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap, sign,
):
    """Pin-level star+L routing — matches TILOS's modules_w_pins iteration
    more accurately than macro-center routing. Routes from actual pin
    positions (macro_pos + pin_offset for macros, fixed_xy for I/O ports).
    On ibm01 initial: macro-center cong ratio 1.16 → pin-level 0.96."""
    max_degree = pin_owner_np.shape[1]
    rows = np.empty(max_degree, dtype=np.int32)
    cols = np.empty(max_degree, dtype=np.int32)
    count = 0
    src_r = 0
    src_c = 0
    source_seen = False
    weight = nw_np[net_idx]

    for d in range(max_degree):
        if not pin_mask_np[net_idx, d]:
            break
        owner = pin_owner_np[net_idx, d]
        if owner >= 0:
            # Macro pin: position = macro_pos + offset (override for incr updates)
            if owner == macro_i and override_x >= 0.0:
                x = override_x + pin_xoff_np[net_idx, d]
                y = override_y + pin_yoff_np[net_idx, d]
            else:
                x = pos[owner, 0] + pin_xoff_np[net_idx, d]
                y = pos[owner, 1] + pin_yoff_np[net_idx, d]
        else:
            # Port pin: fixed position
            x = pin_fixed_x_np[net_idx, d]
            y = pin_fixed_y_np[net_idx, d]
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
def _build_pin_hv_route_grid(
    h_grid, v_grid,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, num_nets,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
):
    for n in range(num_nets):
        _add_net_route_pin_hv(
            h_grid, v_grid, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
            -1, -1.0, -1.0,
            bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0,
        )


@njit
def _update_pin_hv_route_incr_single(
    h_grid, v_grid,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, affected_nets,
    macro_i, old_x, old_y,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
):
    """Pin-level incremental routing update for single macro displacement."""
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        _add_net_route_pin_hv(
            h_grid, v_grid, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, macro_i, old_x, old_y,
            bin_w, bin_h, n_rows, n_cols, hcap, vcap, -1.0,
        )
        _add_net_route_pin_hv(
            h_grid, v_grid, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, -1, -1.0, -1.0,
            bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0,
        )


def build_macro_to_pin_nets(benchmark, num_all_macros):
    """
    Inverted index: macro index → list of net indices that contain any pin
    owned by this macro. Indexed against benchmark.net_pin_nodes positionally
    (so use with `pin_owner` etc. from build_pin_route_tensors).

    Differs from the macro_to_nets built from net_indices (parent's deduped
    macro graph): pin nets include ALL nets ≥2 pins, even single-macro nets
    that the macro-level dedupe drops.
    """
    nets = benchmark.net_pin_nodes
    if not nets:
        return None
    out = [[] for _ in range(num_all_macros)]
    for ni, net_t in enumerate(nets):
        if net_t.shape[0] < 2:
            continue
        arr = net_t.numpy()
        seen = set()
        for d in range(arr.shape[0]):
            owner = int(arr[d, 0])
            if 0 <= owner < num_all_macros and owner not in seen:
                out[owner].append(ni)
                seen.add(owner)
    return [np.array(v, dtype=np.int32) for v in out]


def build_pin_route_tensors(benchmark):
    """
    Pre-process benchmark.net_pin_nodes + macro_pin_offsets + port_positions
    into padded numpy arrays for numba-friendly pin-level routing.

    Returns (pin_owner, pin_mask, pin_xoff, pin_yoff, pin_fixed_x, pin_fixed_y, nw)
    or None if pin data isn't populated.
    """
    nets = benchmark.net_pin_nodes
    if not nets:
        return None
    # Keep all nets indexed positionally — downstream code uses the same
    # net_idx → caller's macro_to_nets mapping. Nets with <2 pins just get
    # mask=False and contribute nothing to routing.
    max_pins = max((t.shape[0] for t in nets), default=0)
    if max_pins == 0:
        return None
    num_nets = len(nets)
    pin_owner = np.full((num_nets, max_pins), -1, dtype=np.int32)
    pin_xoff = np.zeros((num_nets, max_pins), dtype=np.float32)
    pin_yoff = np.zeros((num_nets, max_pins), dtype=np.float32)
    pin_fixed_x = np.zeros((num_nets, max_pins), dtype=np.float32)
    pin_fixed_y = np.zeros((num_nets, max_pins), dtype=np.float32)
    pin_mask = np.zeros((num_nets, max_pins), dtype=np.bool_)
    nw = np.ones(num_nets, dtype=np.float32)
    num_hard = benchmark.num_hard_macros
    num_macros = benchmark.num_macros
    port_positions = (
        benchmark.port_positions.numpy() if benchmark.port_positions.numel() else None
    )
    macro_pin_offsets = [t.numpy() for t in benchmark.macro_pin_offsets]
    for ni, net_t in enumerate(nets):
        if net_t.shape[0] < 2:
            continue
        arr = net_t.numpy()
        for d in range(arr.shape[0]):
            owner = int(arr[d, 0])
            pin_idx = int(arr[d, 1])
            pin_mask[ni, d] = True
            if owner < num_hard:
                pin_owner[ni, d] = owner
                off = macro_pin_offsets[owner][pin_idx]
                pin_xoff[ni, d] = float(off[0])
                pin_yoff[ni, d] = float(off[1])
            elif owner < num_macros:
                pin_owner[ni, d] = owner
            elif port_positions is not None:
                pin_owner[ni, d] = -1
                pidx = owner - num_macros
                pin_fixed_x[ni, d] = float(port_positions[pidx, 0])
                pin_fixed_y[ni, d] = float(port_positions[pidx, 1])
    return pin_owner, pin_mask, pin_xoff, pin_yoff, pin_fixed_x, pin_fixed_y, nw


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
        # Kept as an opt-in experiment. In the incremental-real-eval sweep,
        # soft spread never improved the retained checkpoint, so default
        # runtime is better spent on soft displace / hard displace.
        enable_soft_spread: bool = False,
        # Multi-macro soft local-neighborhood search. This is the coordinated
        # version of the soft-displace breakthrough: try permutations and
        # larger repacks inside congested soft neighborhoods, but only keep
        # real-proxy-improving checkpoints.
        enable_soft_lns: bool = False,
        # Hot-region soft untwist. Kept as an opt-in experiment: on the
        # incremental-real-eval sweep it repeatedly spent real checkpoints
        # without accepting moves on early IBM benches.
        enable_soft_untwist: bool = False,
        # Route-hotspot net shear. This is a congestion-first soft macro phase:
        # identify hot H/V bins, then try coordinated pushes of nearby soft
        # macros away from the crowded stripe. It targets the "straight wires
        # stacked on top of each other" case that slot untwist does not fix.
        enable_net_shear: bool = False,
        # Soft macro coordinate descent / line search. This is the cautious
        # follow-up to soft displace: one soft macro at a time, many targeted
        # candidate positions, same incremental real-ish proxy and real-proxy
        # checkpoint guard.
        enable_soft_cd: bool = True,
        # Community relocate: partition soft cells into communities by shared-net
        # weight, then SA-translate each community together. Targets the soft-
        # to-soft tangle that hard-driven moves (cluster_relocate) can't reach.
        # Disabled: on ibm18 the stage produced 15/120000 accepts in 240s for
        # zero proxy improvement. The placement is at a local optimum for
        # soft-only translations; community moves either spike density at the
        # destination or break WL on inter-community nets.
        enable_community_relocate: bool = False,
        community_target_size: int = 30,
        community_max_budget: int = 240,
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
        self.enable_soft_spread = enable_soft_spread
        self.enable_soft_lns = enable_soft_lns
        self.enable_soft_untwist = enable_soft_untwist
        self.enable_net_shear = enable_net_shear
        self.enable_soft_cd = enable_soft_cd
        self.enable_community_relocate = enable_community_relocate
        self.community_target_size = int(community_target_size)
        self.community_max_budget = int(community_max_budget)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        log_dir = Path(os.environ.get("HAP_LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{benchmark.name}.log"
        original_print = builtins.print

        with log_path.open("w", encoding="utf-8") as log_file:
            def tee_print(*args, **kwargs):
                original_print(*args, **kwargs)
                if kwargs.get("end") == "\r":
                    return
                log_kwargs = dict(kwargs)
                log_kwargs["file"] = log_file
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
        _vis_dir = Path(os.environ.get("HAP_VIS_DIR", "vis")) / benchmark.name

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
        K = 3
        recent_proxies = []
        top_k_candidates = []
        initial_proxy_for_gate = None
        nesterov_bad_tracks = 0

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
            initial_metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
            initial_proxy = initial_metrics["proxy_cost"]
            self._initial_proxy_cost = float(initial_metrics["proxy_cost"])
            self._initial_density_cost = float(initial_metrics["density_cost"])
            self._initial_congestion_cost = float(initial_metrics["congestion_cost"])
            initial_proxy_for_gate = float(initial_metrics["proxy_cost"])
            top_k_candidates.append((initial_proxy, -1, placement.detach().clone()))

        start_time = time()
        total_time_budget = float(os.environ.get("HAP_TOTAL_TIME_BUDGET", "3000"))
        hard_time_budget = float(os.environ.get("HAP_HARD_TIME_BUDGET", str(total_time_budget)))
        min_stage_budget = 120
        nesterov_time_budget = 300
        skip_nesterov = os.environ.get("HAP_SKIP_NESTEROV", "").strip().lower() not in (
            "", "0", "false", "no", "off"
        )
        if skip_nesterov:
            print("Nesterov skipped by HAP_SKIP_NESTEROV=1")
            nesterov_time_budget = -1.0
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
                if initial_proxy_for_gate is not None:
                    # Alternate starts can be outright bad. If Nesterov is
                    # clearly making real proxy worse for consecutive early
                    # checks, keep the initial checkpoint and move on.
                    margin = max(0.025, 0.015 * initial_proxy_for_gate)
                    best_tracked = float(top_k_candidates[0][0]) if top_k_candidates else float(proxy_est)
                    if float(proxy_est) > initial_proxy_for_gate + margin and best_tracked >= initial_proxy_for_gate - 1e-4:
                        nesterov_bad_tracks += 1
                    else:
                        nesterov_bad_tracks = 0
                    if nesterov_bad_tracks >= 2:
                        print(
                            f"Nesterov worsening early, stopping at step {step} "
                            f"(start={initial_proxy_for_gate:.4f}, current={float(proxy_est):.4f})"
                        )
                        break
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

        policy_initial_cong = float(getattr(self, "_initial_congestion_cost", 0.0))
        policy_initial_density = float(getattr(self, "_initial_density_cost", 0.0))
        policy_num_soft = int(getattr(benchmark, "num_soft_macros", max(0, num_all - num_hard)))
        high_cong_bench = policy_initial_cong >= 2.10
        very_high_cong_bench = policy_initial_cong >= 2.30
        if very_high_cong_bench:
            hard_swap_max_budget = 60
            soft_swap_max_budget = 240
            soft_displace_max_budget = 2000
            soft_cd_max_budget = 720
            soft_cd_base_budget = 300
            soft_cd_extension_budget = 120
            hard_displace_max_budget = 240
            budget_policy = "very_high_cong"
        elif high_cong_bench:
            hard_swap_max_budget = 75
            soft_swap_max_budget = 270
            soft_displace_max_budget = 1700
            soft_cd_max_budget = 720
            soft_cd_base_budget = 270
            soft_cd_extension_budget = 120
            hard_displace_max_budget = 300
            budget_policy = "high_cong"
        else:
            hard_swap_max_budget = 90
            soft_swap_max_budget = 240
            soft_displace_max_budget = 1300
            soft_cd_max_budget = 780
            soft_cd_base_budget = 240
            soft_cd_extension_budget = 120
            hard_displace_max_budget = 450
            budget_policy = "default"

        if policy_num_soft >= 1500:
            soft_macro_bonus = 150
        elif policy_num_soft >= 1000:
            soft_macro_bonus = 100
        elif policy_num_soft >= 700:
            soft_macro_bonus = 50
        else:
            soft_macro_bonus = 0
        soft_displace_max_budget += soft_macro_bonus
        if soft_macro_bonus:
            soft_cd_max_budget = max(480, soft_cd_max_budget - soft_macro_bonus // 3)
            hard_displace_max_budget = max(180, hard_displace_max_budget - soft_macro_bonus // 4)

        soft_displace_max_budget = int(
            os.environ.get("HAP_SOFT_DISPLACE_MAX", soft_displace_max_budget)
        )
        soft_cd_max_budget = int(os.environ.get("HAP_SOFT_CD_MAX", soft_cd_max_budget))
        soft_cd_base_budget = int(os.environ.get("HAP_SOFT_CD_BASE", soft_cd_base_budget))
        soft_cd_extension_budget = int(
            os.environ.get("HAP_SOFT_CD_EXTENSION", soft_cd_extension_budget)
        )
        hard_displace_max_budget = int(
            os.environ.get("HAP_HARD_DISPLACE_MAX", hard_displace_max_budget)
        )
        print(
            f"Budget policy: {budget_policy} "
            f"init_den={policy_initial_density:.4f} init_cong={policy_initial_cong:.4f} "
            f"num_soft={policy_num_soft} soft_bonus={soft_macro_bonus}s "
            f"soft_displace_max={soft_displace_max_budget}s "
            f"soft_cd_max={soft_cd_max_budget}s hard_displace_max={hard_displace_max_budget}s"
        )
        sparse_checkpoints = os.environ.get("HAP_SPARSE_CHECKPOINTS", "1").strip().lower() not in (
            "", "0", "false", "no", "off"
        )
        soft_swap_checkpoint_every = int(
            os.environ.get("HAP_SOFT_SWAP_CKPT", "800" if sparse_checkpoints else "300")
        )
        soft_displace_checkpoint_every = int(
            os.environ.get("HAP_SOFT_DISPLACE_CKPT", "800" if sparse_checkpoints else "200")
        )
        soft_cd_checkpoint_every = int(
            os.environ.get("HAP_SOFT_CD_CKPT", "180" if sparse_checkpoints else "60")
        )
        hard_displace_checkpoint_every = int(
            os.environ.get("HAP_HARD_DISPLACE_CKPT", "200")
        )
        print(
            "Checkpoint policy: "
            f"soft_swap={soft_swap_checkpoint_every} "
            f"soft_displace={soft_displace_checkpoint_every} "
            f"soft_cd={soft_cd_checkpoint_every} "
            f"hard_displace={hard_displace_checkpoint_every} "
            f"sparse={'on' if sparse_checkpoints else 'off'}"
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
            swap_budget = stage_budget("Hard swap", max_budget=hard_swap_max_budget)
            print(
                f"Hard swap budget: {swap_budget:.0f}s "
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
                    checkpoint_every=400,
                )
                _save_plot("02_hard_swap", best_valid_placement)
            else:
                print("Hard swap skipped: no remaining budget")

            if benchmark.num_soft_macros > 1:
                remaining = total_time_budget - (time() - start_time)
                soft_swap_budget = stage_budget("Soft swap", max_budget=soft_swap_max_budget)
                print(
                    f"Soft swap budget: {soft_swap_budget:.0f}s "
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
                        base_budget=90,
                        extension_budget=60,
                        extension_gain=0.002,
                        checkpoint_every=soft_swap_checkpoint_every,
                    )
                    _save_plot("03_soft_swap", best_valid_placement)
                else:
                    print("Soft swap skipped: no remaining budget")

                if self.enable_soft_spread:
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
                else:
                    print("Soft spread disabled")

                remaining = total_time_budget - (time() - start_time)
                soft_displace_budget = stage_budget("Soft displace", max_budget=soft_displace_max_budget)
                print(
                    f"Soft displace budget: {soft_displace_budget:.0f}s "
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
                        checkpoint_every=soft_displace_checkpoint_every,
                    )
                    _save_plot("05_soft_displace", best_valid_placement)
                else:
                    print("Soft displace skipped: no remaining budget")

                # Mid-pipeline emergency checkpoint: if we're already
                # >80% through the hard time budget when Soft displace
                # ends, write a pkl with the current best so that a
                # SIGTERM during the remaining stages (Soft CD / Hard
                # displace / Tail / final logging) still leaves a
                # usable result on disk.
                elapsed_after_soft_disp = time() - start_time
                if (
                    elapsed_after_soft_disp > 0.80 * hard_time_budget
                    and plc is not None
                ):
                    out_path_mid = getattr(self, "_out_path", None)
                    if out_path_mid:
                        try:
                            mid_metrics = compute_proxy_cost(
                                best_valid_placement, benchmark, plc
                            )
                            mid_out = {
                                "placement": best_valid_placement.detach().cpu(),
                                "proxy": float(mid_metrics["proxy_cost"]),
                                "wl": float(mid_metrics["wirelength_cost"]),
                                "den": float(mid_metrics["density_cost"]),
                                "cong": float(mid_metrics["congestion_cost"]),
                                "overlap_count": int(mid_metrics["overlap_count"]),
                            }
                            tmp = str(out_path_mid) + ".tmp"
                            with open(tmp, "wb") as f:
                                pickle.dump(mid_out, f)
                            os.replace(tmp, out_path_mid)
                            print(
                                f"[checkpoint] mid-pipeline pkl written "
                                f"(elapsed={elapsed_after_soft_disp:.0f}s, "
                                f"proxy={mid_out['proxy']:.4f})"
                            )
                        except Exception as exc:
                            print(f"[warn] mid checkpoint write failed: {exc}")

                if self.enable_soft_cd:
                    remaining = total_time_budget - (time() - start_time)
                    soft_cd_budget = stage_budget("Soft CD", max_budget=soft_cd_max_budget)
                    print(
                        f"Soft CD budget: {soft_cd_budget:.0f}s "
                        f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                    )
                    if soft_cd_budget > 0:
                        best_valid_placement = self._sa_soft_cd_refine(
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
                            budget=soft_cd_budget,
                            base_budget=soft_cd_base_budget,
                            extension_budget=soft_cd_extension_budget,
                            extension_gain=2e-3,
                            checkpoint_every=soft_cd_checkpoint_every,
                        )
                        _save_plot("06_soft_cd", best_valid_placement)
                    else:
                        print("Soft CD skipped: no remaining budget")
                else:
                    print("Soft CD disabled")

                if self.enable_net_shear:
                    remaining = total_time_budget - (time() - start_time)
                    net_shear_budget = stage_budget("Net shear", max_budget=240)
                    print(
                        f"Net shear budget: {net_shear_budget:.0f}s "
                        f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                    )
                    if net_shear_budget > 0:
                        best_valid_placement = self._soft_net_shear_refine(
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
                            budget=net_shear_budget,
                        )
                        _save_plot("07_net_shear", best_valid_placement)
                    else:
                        print("Net shear skipped: no remaining budget")
                else:
                    print("Net shear disabled")

                if self.enable_soft_untwist:
                    remaining = total_time_budget - (time() - start_time)
                    soft_untwist_budget = stage_budget("Soft untwist", max_budget=180)
                    print(
                        f"Soft untwist budget: {soft_untwist_budget:.0f}s "
                        f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                    )
                    if soft_untwist_budget > 0:
                        best_valid_placement = self._soft_hot_untwist_refine(
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
                            budget=soft_untwist_budget,
                        )
                        _save_plot("07_soft_untwist", best_valid_placement)
                    else:
                        print("Soft untwist skipped: no remaining budget")
                else:
                    print("Soft untwist disabled")

                if self.enable_soft_lns:
                    remaining = total_time_budget - (time() - start_time)
                    soft_lns_budget = stage_budget("Soft LNS", max_budget=180)
                    print(
                        f"Soft LNS budget: {soft_lns_budget:.0f}s "
                        f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                    )
                    if soft_lns_budget > 0:
                        best_valid_placement = self._sa_soft_lns_repack(
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
                            budget=soft_lns_budget,
                            checkpoint_every=16,
                        )
                        _save_plot("07_soft_lns", best_valid_placement)
                    else:
                        print("Soft LNS skipped: no remaining budget")
                else:
                    print("Soft LNS disabled")

                escape_enabled = os.environ.get("HAP_ESCAPE_PERTURB", "").strip().lower() not in (
                    "", "0", "false", "no", "off"
                )
                if escape_enabled:
                    remaining = total_time_budget - (time() - start_time)
                    escape_max = float(os.environ.get("HAP_ESCAPE_BUDGET", "300"))
                    escape_budget = stage_budget("Perturb escape", max_budget=escape_max)
                    print(
                        f"Perturb escape budget: {escape_budget:.0f}s "
                        f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                    )
                    if escape_budget > 0:
                        best_valid_placement = self._soft_perturb_escape(
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
                            budget=escape_budget,
                        )
                        _save_plot("07_perturb_escape", best_valid_placement)
                    else:
                        print("Perturb escape skipped: no remaining budget")
                else:
                    print("Perturb escape disabled")

            remaining = total_time_budget - (time() - start_time)
            displace_budget = stage_budget("Hard displace", max_budget=hard_displace_max_budget)
            print(
                f"Hard displace budget: {displace_budget:.0f}s "
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
                    checkpoint_every=hard_displace_checkpoint_every,
                )
                _save_plot("08_hard_displace", best_valid_placement)
            else:
                print("Hard displace skipped: no remaining budget")

            if getattr(self, "enable_community_relocate", False):
                remaining = total_time_budget - (time() - start_time)
                comm_budget = stage_budget(
                    "Community relocate",
                    max_budget=int(getattr(self, "community_max_budget", 240)),
                )
                print(
                    f"Community relocate budget: {comm_budget:.0f}s "
                    f"(elapsed={time()-start_time:.1f}s remaining={remaining:.1f}s)"
                )
                if comm_budget > 0:
                    best_valid_placement = self._sa_community_relocate(
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
                        budget=comm_budget,
                        target_size=int(getattr(self, "community_target_size", 30)),
                        checkpoint_every=80,
                    )
                    _save_plot("08b_community_relocate", best_valid_placement)
                else:
                    print("Community relocate skipped: no remaining budget")

            # best_valid_placement = self._cd_refine(
            #     best_valid_placement, benchmark, plc,
            #     net_indices, net_mask, num_hard, num_all, fixed,
            #     budget=180,
            # )

            tail_soft_swap_enabled = os.environ.get("HAP_TAIL_SOFT_SWAP", "1").strip().lower() not in (
                "", "0", "false", "no", "off"
            )
            if tail_soft_swap_enabled and benchmark.num_soft_macros > 1:
                elapsed = time() - start_time
                hard_remaining = hard_time_budget - elapsed
                tail_swap_min = float(os.environ.get("HAP_TAIL_SOFT_SWAP_MIN", "90"))
                tail_swap_reserve = float(os.environ.get("HAP_TAIL_SOFT_SWAP_RESERVE", "260"))
                tail_swap_max = float(os.environ.get("HAP_TAIL_SOFT_SWAP_MAX", "150"))
                tail_swap_budget = min(
                    tail_swap_max,
                    max(0.0, hard_remaining - tail_swap_reserve),
                )
                print(
                    f"Tail soft swap budget: {tail_swap_budget:.0f}s "
                    f"(elapsed={elapsed:.1f}s hard_remaining={hard_remaining:.1f}s "
                    f"reserve={tail_swap_reserve:.0f}s)"
                )
                if tail_swap_budget >= tail_swap_min:
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
                        budget=tail_swap_budget,
                        base_budget=min(90, tail_swap_budget),
                        extension_budget=30,
                        extension_gain=0.0015,
                        checkpoint_every=soft_swap_checkpoint_every,
                    )
                    _save_plot("08c_tail_soft_swap", best_valid_placement)
                else:
                    print(
                        f"Tail soft swap skipped: budget {tail_swap_budget:.1f}s "
                        f"< min {tail_swap_min:.1f}s"
                    )
            elif not tail_soft_swap_enabled:
                print("Tail soft swap disabled")

            tail_soft_enabled = os.environ.get("HAP_TAIL_SOFT_DISPLACE", "1").strip().lower() not in (
                "", "0", "false", "no", "off"
            )
            if tail_soft_enabled and benchmark.num_soft_macros > 1:
                elapsed = time() - start_time
                hard_remaining = hard_time_budget - elapsed
                tail_min = float(os.environ.get("HAP_TAIL_SOFT_DISPLACE_MIN", "180"))
                tail_reserve = float(os.environ.get("HAP_TAIL_SOFT_DISPLACE_RESERVE", "75"))
                tail_max = float(os.environ.get("HAP_TAIL_SOFT_DISPLACE_MAX", "600"))
                tail_budget = min(tail_max, max(0.0, hard_remaining - tail_reserve))
                print(
                    f"Tail soft displace budget: {tail_budget:.0f}s "
                    f"(elapsed={elapsed:.1f}s hard_remaining={hard_remaining:.1f}s "
                    f"reserve={tail_reserve:.0f}s)"
                )
                if tail_budget >= tail_min:
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
                        budget=tail_budget,
                        checkpoint_every=soft_displace_checkpoint_every,
                    )
                    _save_plot("08d_tail_soft_displace", best_valid_placement)
                else:
                    print(
                        f"Tail soft displace skipped: budget {tail_budget:.1f}s "
                        f"< min {tail_min:.1f}s"
                    )
            elif not tail_soft_enabled:
                print("Tail soft displace disabled")


        # Soft settle was score-neutral to slightly harmful in current logs, so
        # keep the post-SA placement exactly as selected by real-proxy stages.
        if best_valid_placement is not None:
            _save_plot("09_parent_final", best_valid_placement)

        final = best_valid_placement

        # Emergency legalize safety check
        if self._hard_overlap_count(final, benchmark) > 0:
            print("WARNING: final has overlaps, emergency legalize")
            final = strong_legalize(final, benchmark, gap=0.05, max_iters=200)

        # Compute metrics once. We were going to compute them inside
        # _log_stats anyway; doing it here lets us write the worker pkl
        # BEFORE the final log step, so a SIGTERM during _log_stats or
        # after _place_impl returns still leaves a valid pkl behind.
        final_metrics = None
        if plc is not None:
            final_metrics = compute_proxy_cost(final, benchmark, plc)
            out_path = getattr(self, "_out_path", None)
            if out_path:
                try:
                    out = {
                        "placement": final.detach().cpu(),
                        "proxy": float(final_metrics["proxy_cost"]),
                        "wl": float(final_metrics["wirelength_cost"]),
                        "den": float(final_metrics["density_cost"]),
                        "cong": float(final_metrics["congestion_cost"]),
                        "overlap_count": int(final_metrics["overlap_count"]),
                    }
                    tmp = str(out_path) + ".tmp"
                    with open(tmp, "wb") as f:
                        pickle.dump(out, f)
                    os.replace(tmp, out_path)
                except Exception as exc:
                    print(f"[warn] early pkl write failed: {exc}")

        self._log_stats(
            "final", benchmark, final, plc,
            wl=None, density_weight=density_weight, metrics=final_metrics,
        )
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
            # Pin-level routing model (matches TILOS modules_w_pins iteration).
            # Falls back to macro-center if pin data missing.
            _pin_tensors = build_pin_route_tensors(benchmark)
            _use_pin_routing = _pin_tensors is not None
            if _use_pin_routing:
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
                macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
                num_pin_nets = pin_owner_p.shape[0]
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )

            current_den = _density_cost_top5(density_grid)
            current_cong = cong_tracker.cost()

            # Calibrate fast proxy to match real proxy scale
            _set_placement(plc, sa_placement.detach(), benchmark)
            real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
            best_proxy = real_metrics["proxy_cost"]
            den_scale = real_metrics['density_cost'] / (current_den + 1e-8)
            cong_scale = real_metrics['congestion_cost'] / (current_cong + 1e-8)

            # Use scaled values for current_proxy
            current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)

            best_placement = sa_placement.clone()
            print(f"Hard displace start: real={best_proxy:.4f} fast={current_proxy:.4f} wl={total_wl:.4f} den_scale={den_scale:.4f} cong_scale={cong_scale:.4f} (HV cong, smooth={smooth_range})")

            accepts = total = stalls = 0
            last_accept_step = 0
            t0 = time()
            progress_gate = WindowProgressGate(
                window=3,
                patience_windows=1,
                epsilon=0.0010,
                min_time=min(180.0, max(0.0, float(budget))),
                initial_best=best_proxy,
            )
            max_displacement = canvas_norm * 0.03
            no_accept_limit = max(
                100_000,
                int(os.environ.get("HAP_HARD_NO_ACCEPT_LIMIT", "250000")),
            )

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
                    if _use_pin_routing:
                        _build_pin_hv_route_grid(
                            h_route_grid, v_route_grid,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
                    else:
                        _build_hv_route_grid(h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                    _build_macro_route_grid(h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)
                    current_den = _density_cost_top5(density_grid)
                    cong_tracker = SmoothHVCostTracker(
                        h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
                    )
                    current_cong = cong_tracker.cost()
                    current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)

                if total % 100000 == 0 and total > 0:
                    print(f"  step={total} accepts={accepts} wl={total_wl:.4f} fast={current_proxy:.4f} [{time()-t0:.0f}s]", end="\r")

                i = random.randint(0, num_hard - 1)
                if fixed[i].item():
                    continue

                total += 1
                if total - last_accept_step > no_accept_limit:
                    print(
                        f"Hard displace: no accepts in {no_accept_limit} attempts, stopping"
                    )
                    break

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
                if _use_pin_routing:
                    pin_aff = macro_to_pin_nets[i]
                    if len(pin_aff) > 0:
                        update_pin_hv_route_incr_single_smooth(
                            h_route_grid, v_route_grid, cong_tracker,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            sa_pos, pin_aff, i, old_x, old_y,
                            bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
                else:
                    update_hv_route_incr_single_smooth(h_route_grid, v_route_grid, cong_tracker, ni_np, nm_np, nw_np, sa_pos, aff, i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                _update_macro_route_incr_single(h_macro_grid, v_macro_grid, old_x, old_y, new_x, new_y, hw_np[i], hh_np[i], bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)
                new_cong = cong_tracker.cost()

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
                        _, gate_stop = progress_gate.update(real_proxy, time() - t0)
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
                            if _use_pin_routing:
                                _build_pin_hv_route_grid(
                                    h_route_grid, v_route_grid,
                                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                                )
                            else:
                                _build_hv_route_grid(h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                            _build_macro_route_grid(h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)
                            current_den = _density_cost_top5(density_grid)
                            cong_tracker = SmoothHVCostTracker(
                                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
                            )
                            current_cong = cong_tracker.cost()
                            # After revert, components are now for best state.
                            # Refit scales to best so fast proxy is honest here too.
                            _set_placement(plc, sa_placement.detach(), benchmark)
                            best_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                            den_scale = best_metrics['density_cost'] / (current_den + 1e-8)
                            cong_scale = best_metrics['congestion_cost'] / (current_cong + 1e-8)
                            current_proxy = total_wl + 0.5 * (current_den * den_scale) + .5 * (current_cong * cong_scale)
                            stalls += 1
                        if gate_stop:
                            print(
                                f"Hard displace stalled "
                                f"(window gate, best={progress_gate.best:.4f}, "
                                f"patience={progress_gate.patience}/{progress_gate.max_patience})"
                            )
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
                    if _use_pin_routing:
                        pin_aff = macro_to_pin_nets[i]
                        if len(pin_aff) > 0:
                            update_pin_hv_route_incr_single_smooth(
                                h_route_grid, v_route_grid, cong_tracker,
                                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                                sa_pos, pin_aff, i, new_x, new_y,
                                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                            )
                    else:
                        update_hv_route_incr_single_smooth(h_route_grid, v_route_grid, cong_tracker, ni_np, nm_np, nw_np, sa_pos, aff, i, new_x, new_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap)
                    _update_macro_route_incr_single(h_macro_grid, v_macro_grid, new_x, new_y, old_x, old_y, hw_np[i], hh_np[i], bl, br, bb, bt, n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc)

                if total - last_accept_step > no_accept_limit:
                    print(
                        f"Hard displace: no accepts in {no_accept_limit} attempts, stopping"
                    )
                    break

            print(f"Hard displace done: {total} attempts, {accepts} accepts, best real_proxy={best_proxy:.4f}")
            return best_placement

    def _build_hard_to_soft_clusters(
        self,
        num_hard: int,
        num_all: int,
        net_indices,
        net_mask,
        net_weights,
        threshold: float = 0.30,
    ):
        """Assign each soft cell to its highest-weighted hard partner if affinity >= threshold.

        Returns list of length num_hard, where cluster[h] is an np.int64 array of GLOBAL
        soft macro indices belonging to h. A soft cell appears in at most one cluster.
        Softs with no qualifying hard partner are unassigned (not in any cluster).
        """
        num_soft = num_all - num_hard
        empty = [np.array([], dtype=np.int64) for _ in range(num_hard)]
        if num_soft <= 0 or num_hard <= 0:
            return empty

        ni = net_indices.numpy().copy() if hasattr(net_indices, "numpy") else np.asarray(net_indices)
        nm = net_mask.numpy().copy() if hasattr(net_mask, "numpy") else np.asarray(net_mask)
        nw = net_weights.numpy().copy() if hasattr(net_weights, "numpy") else np.asarray(net_weights)
        num_nets = ni.shape[0]

        affinity = np.zeros((num_soft, num_hard), dtype=np.float32)
        soft_total = np.zeros(num_soft, dtype=np.float32)

        for n in range(num_nets):
            members = ni[n][nm[n]]
            if members.size < 2:
                continue
            w = float(nw[n])
            is_hard = members < num_hard
            if not is_hard.any() or is_hard.all():
                continue
            hard_in_net = members[is_hard]
            soft_in_net = members[~is_hard] - num_hard
            soft_total[soft_in_net] += w
            affinity[np.ix_(soft_in_net, hard_in_net)] += w

        best_hard = np.argmax(affinity, axis=1)
        best_affinity = affinity[np.arange(num_soft), best_hard]
        mask = (soft_total > 1e-8) & (best_affinity / np.maximum(soft_total, 1e-8) >= threshold)

        cluster_lists = [[] for _ in range(num_hard)]
        for s_idx in np.where(mask)[0]:
            cluster_lists[int(best_hard[s_idx])].append(int(s_idx) + num_hard)
        return [np.array(c, dtype=np.int64) for c in cluster_lists]

    def _sa_cluster_relocate(
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
        cluster_threshold=0.0,
    ):
        """SA stage: move a hard macro AND its >threshold-affine soft cluster together.

        Drop-in replacement for `_sa_refine_displace`. When a hard macro's cluster is
        empty, the move is equivalent to a single-macro hard displace. With non-empty
        clusters, both the hard and its dragged softs translate by the same delta —
        intra-cluster nets keep their HPWL, external nets see only the hard's typical
        WL change, but routing topology can shift because the hard's blockage moves to
        a new bin and its connected soft routing demand moves with it.
        """
        # Build cluster mapping once for this placement.
        clusters = self._build_hard_to_soft_clusters(
            num_hard, num_all, net_indices, net_mask, net_weights, threshold=cluster_threshold,
        )
        cluster_sizes = np.array([len(c) for c in clusters], dtype=np.int32)
        nonzero = cluster_sizes[cluster_sizes > 0]
        print(
            f"Cluster map: {(cluster_sizes > 0).sum()}/{num_hard} hards have clusters "
            f"(threshold={cluster_threshold:.2f}); cluster_size "
            f"mean={(nonzero.mean() if nonzero.size else 0):.1f} "
            f"max={(int(nonzero.max()) if nonzero.size else 0)} "
            f"total_softs_assigned={int(cluster_sizes.sum())}/{num_all - num_hard}"
        )

        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, _eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
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
        _build_density_grid(
            density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )

        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()

        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        den_scale = real_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = real_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
        best_placement = sa_placement.clone()
        print(
            f"Cluster relocate start: real={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} den_scale={den_scale:.4f} cong_scale={cong_scale:.4f} "
            f"(HV cong, smooth={smooth_range})"
        )

        accepts = total = 0
        last_accept_step = 0
        t0 = time()
        max_displacement = canvas_norm * 0.05
        progress_gate = WindowProgressGate(
            window=2,
            patience_windows=1,
            epsilon=0.0005,
            min_time=min(90.0, max(0.0, float(budget))),
            initial_best=best_proxy,
        )

        movable_hards = np.array(
            [h for h in range(num_hard) if not bool(fixed[h].item())],
            dtype=np.int64,
        )
        if movable_hards.size == 0:
            print("Cluster relocate: no movable hards, returning input")
            return placement

        def apply_member_move(idx: int, ox: float, oy: float, nx: float, ny: float) -> None:
            sa_pos[idx, 0] = nx
            sa_pos[idx, 1] = ny
            _update_density_incr(
                density_grid, ox, oy, nx, ny,
                hw_np[idx], hh_np[idx], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[idx]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        sa_pos, pin_aff, idx, ox, oy,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                aff_route = macro_to_nets[idx]
                if len(aff_route) > 0:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        ni_np, nm_np, nw_np, sa_pos, aff_route, idx, ox, oy,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        def rebuild_all():
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            density_grid[:] = 0
            _build_density_grid(
                density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            h_macro_grid[:] = 0
            v_macro_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )
            current_den = _density_cost_top5(density_grid)
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_cong = cong_tracker.cost()
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale

        no_accept_limit = max(
            100_000,
            int(os.environ.get("HAP_HARD_NO_ACCEPT_LIMIT", "250000")),
        )

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break
            if accepts > 0 and accepts % 500 == 0 and accepts % checkpoint_every != 0:
                rebuild_all()

            if total % 100000 == 0 and total > 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                    f"fast={current_proxy:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            h = int(movable_hards[random.randint(0, movable_hards.size - 1)])
            total += 1
            if total - last_accept_step > no_accept_limit:
                print(f"Cluster relocate: no accepts in {no_accept_limit} attempts, stopping")
                break

            cluster = clusters[h]
            n_members = 1 + cluster.size
            members = np.empty(n_members, dtype=np.int64)
            members[0] = h
            if cluster.size > 0:
                members[1:] = cluster

            # Propose displacement for the hard, then translate cluster by the same delta.
            old_hx = float(sa_pos[h, 0])
            old_hy = float(sa_pos[h, 1])
            dx = random.uniform(-max_displacement, max_displacement)
            dy = random.uniform(-max_displacement, max_displacement)
            new_hx = float(np.clip(old_hx + dx, hw_np[h], benchmark.canvas_width - hw_np[h]))
            new_hy = float(np.clip(old_hy + dy, hh_np[h], benchmark.canvas_height - hh_np[h]))

            # Hard–hard overlap check at hard's destination.
            has_overlap = False
            for k in range(num_hard):
                if k == h:
                    continue
                if (abs(new_hx - sa_pos[k, 0]) < sep_x_np[h, k] and
                        abs(new_hy - sa_pos[k, 1]) < sep_y_np[h, k]):
                    has_overlap = True
                    break
            if has_overlap:
                continue

            actual_dx = new_hx - old_hx
            actual_dy = new_hy - old_hy

            old_positions = np.empty((n_members, 2), dtype=np.float32)
            new_positions = np.empty((n_members, 2), dtype=np.float32)
            for idx, m in enumerate(members):
                ox = float(sa_pos[m, 0])
                oy = float(sa_pos[m, 1])
                old_positions[idx, 0] = ox
                old_positions[idx, 1] = oy
                new_positions[idx, 0] = float(np.clip(ox + actual_dx, hw_np[m], benchmark.canvas_width - hw_np[m]))
                new_positions[idx, 1] = float(np.clip(oy + actual_dy, hh_np[m], benchmark.canvas_height - hh_np[m]))

            # Apply every member's move.
            for idx, m in enumerate(members):
                apply_member_move(int(m), float(old_positions[idx, 0]), float(old_positions[idx, 1]),
                                  float(new_positions[idx, 0]), float(new_positions[idx, 1]))
            # Macro blockage update only for the hard.
            _update_macro_route_incr_single(
                h_macro_grid, v_macro_grid,
                float(old_positions[0, 0]), float(old_positions[0, 1]),
                float(new_positions[0, 0]), float(new_positions[0, 1]),
                hw_np[h], hh_np[h], bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )

            new_den = _density_cost_top5(density_grid)
            new_cong = cong_tracker.cost()

            aff_lists = [macro_to_nets[int(m)] for m in members if len(macro_to_nets[int(m)]) > 0]
            if aff_lists:
                aff = np.unique(np.concatenate(aff_lists))
                old_hpwl = net_hpwl[aff].copy()
                new_hpwl = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
                wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)
            else:
                aff = np.array([], dtype=np.int64)
                old_hpwl = np.empty(0, dtype=np.float32)
                new_hpwl = np.empty(0, dtype=np.float32)
                wl_delta = 0.0

            new_proxy = (total_wl + wl_delta) + 0.5 * new_den * den_scale + 0.5 * new_cong * cong_scale

            if new_proxy < current_proxy:
                for idx, m in enumerate(members):
                    sa_placement[int(m), 0] = float(new_positions[idx, 0])
                    sa_placement[int(m), 1] = float(new_positions[idx, 1])
                if aff.size > 0:
                    net_hpwl[aff] = new_hpwl
                total_wl += wl_delta
                current_den = new_den
                current_cong = new_cong
                current_proxy = new_proxy
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                    real_proxy = metrics["proxy_cost"]
                    den_scale = metrics["density_cost"] / (current_den + 1e-8)
                    cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                    current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                        f"fast={current_proxy:.4f} real={real_proxy:.4f} "
                        f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                        f"[{time()-t0:.0f}s]"
                    )
                    _, gate_stop = progress_gate.update(real_proxy, time() - t0)
                    if real_proxy < best_proxy:
                        best_proxy = real_proxy
                        best_placement = sa_placement.clone()
                    if gate_stop:
                        print(
                            f"Cluster relocate stalled "
                            f"(window gate, best={best_proxy:.4f}, "
                            f"patience={progress_gate.patience}/{progress_gate.max_patience})"
                        )
                        break
            else:
                # Revert in reverse order. Incremental updates compose, so reverse
                # also works, but reversing matches the apply order pairwise.
                for idx in range(n_members - 1, -1, -1):
                    m = int(members[idx])
                    nx = float(new_positions[idx, 0])
                    ny = float(new_positions[idx, 1])
                    ox = float(old_positions[idx, 0])
                    oy = float(old_positions[idx, 1])
                    apply_member_move(m, nx, ny, ox, oy)
                _update_macro_route_incr_single(
                    h_macro_grid, v_macro_grid,
                    float(new_positions[0, 0]), float(new_positions[0, 1]),
                    float(old_positions[0, 0]), float(old_positions[0, 1]),
                    hw_np[h], hh_np[h], bl, br, bb, bt,
                    n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
                )

        print(
            f"Cluster relocate done: {total} attempts, {accepts} accepts, "
            f"best real_proxy={best_proxy:.4f}"
        )
        return best_placement

    def _build_soft_communities(
        self,
        num_hard: int,
        num_all: int,
        net_indices,
        net_mask,
        net_weights,
        target_size: int = 30,
    ):
        """Greedy soft-soft community detection by shared-net weight.

        Builds a soft-soft adjacency matrix where edge weight = sum of net
        weights for nets touching both softs. Then greedily grows communities:
        seed from highest-degree unassigned soft, repeatedly add the
        most-connected unassigned soft to the current community, until size
        reaches `target_size` or no positive-weight neighbor remains.

        Returns list of np.int64 arrays of GLOBAL soft macro indices.
        Every soft (with at least one shared-net) ends up in exactly one community.
        Isolated softs get their own singleton communities.
        """
        from scipy.sparse import coo_matrix

        num_soft = num_all - num_hard
        if num_soft <= 1:
            return [np.array([num_hard + i for i in range(num_soft)], dtype=np.int64)]

        ni = net_indices.numpy().copy() if hasattr(net_indices, "numpy") else np.asarray(net_indices)
        nm = net_mask.numpy().copy() if hasattr(net_mask, "numpy") else np.asarray(net_mask)
        nw = net_weights.numpy().copy() if hasattr(net_weights, "numpy") else np.asarray(net_weights)
        num_nets = ni.shape[0]

        rows = []
        cols = []
        vals = []
        for n in range(num_nets):
            members = ni[n][nm[n]]
            if members.size < 2:
                continue
            w = float(nw[n])
            softs = members[members >= num_hard] - num_hard
            k = softs.size
            if k < 2:
                continue
            # All-pairs within softs of this net.
            for i in range(k):
                si = int(softs[i])
                for j in range(i + 1, k):
                    sj = int(softs[j])
                    rows.append(si); cols.append(sj); vals.append(w)
                    rows.append(sj); cols.append(si); vals.append(w)

        if not rows:
            return [np.array([num_hard + i], dtype=np.int64) for i in range(num_soft)]

        adj = coo_matrix(
            (np.asarray(vals, dtype=np.float32),
             (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(num_soft, num_soft),
        ).tocsr()
        adj.sum_duplicates()

        degree = np.asarray(adj.sum(axis=1)).flatten()
        assigned = np.full(num_soft, -1, dtype=np.int32)
        communities: List[np.ndarray] = []
        order = np.argsort(-degree)

        target_size = max(2, int(target_size))

        for seed in order:
            seed = int(seed)
            if assigned[seed] >= 0:
                continue
            cid = len(communities)
            members_set = [seed]
            assigned[seed] = cid

            neighbor_weights = adj.getrow(seed).toarray().flatten().astype(np.float32)
            neighbor_weights[seed] = -1.0

            while len(members_set) < target_size:
                # Block already-assigned neighbors.
                mask_assigned = assigned >= 0
                if mask_assigned.any():
                    neighbor_weights[mask_assigned] = -1.0

                best = int(np.argmax(neighbor_weights))
                if neighbor_weights[best] <= 0.0:
                    break
                members_set.append(best)
                assigned[best] = cid
                row = adj.getrow(best).toarray().flatten().astype(np.float32)
                neighbor_weights += row
                neighbor_weights[best] = -1.0

            communities.append(
                np.array([num_hard + m for m in members_set], dtype=np.int64)
            )

        # Any softs unassigned (no shared-net edges at all) → singletons.
        for s in range(num_soft):
            if assigned[s] < 0:
                communities.append(np.array([num_hard + s], dtype=np.int64))

        return communities

    def _sa_community_relocate(
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
        budget: float = 240.0,
        target_size: int = 30,
        checkpoint_every: int = 80,
    ):
        """SA stage: translate soft-cell communities together to untangle long nets.

        Communities are sets of softs that strongly net-connect to each other.
        Translating one as a unit preserves intra-community WL while exploring
        better positions for inter-community routing.
        """
        num_soft = num_all - num_hard
        if num_soft < 2:
            return placement

        communities = self._build_soft_communities(
            num_hard, num_all, net_indices, net_mask, net_weights, target_size=target_size,
        )
        sizes = np.array([len(c) for c in communities], dtype=np.int32)
        multi = sizes[sizes > 1]
        print(
            f"Community map: {len(communities)} communities (target_size={target_size}); "
            f"multi-member={(sizes > 1).sum()} singletons={(sizes == 1).sum()} "
            f"mean_multi={(multi.mean() if multi.size else 0):.1f} "
            f"max_size={int(sizes.max()) if sizes.size else 0}"
        )

        # Restrict moves to multi-member communities. Singletons are already
        # what _sa_soft_displace does at the cell level.
        movable = [i for i, c in enumerate(communities) if len(c) >= 2]
        if not movable:
            print("Community relocate: no multi-member communities, skipping")
            return placement
        movable_arr = np.array(movable, dtype=np.int64)

        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, _eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
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
        _build_density_grid(
            density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        # Hard macro blockage stays fixed (we only move softs).
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )

        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()

        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        den_scale = real_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = real_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
        best_placement = sa_placement.clone()
        print(
            f"Community relocate start: real={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} den_scale={den_scale:.4f} cong_scale={cong_scale:.4f} "
            f"(HV cong, smooth={smooth_range})"
        )

        accepts = total = 0
        last_accept_step = 0
        t0 = time()
        max_displacement = canvas_norm * 0.04
        progress_gate = WindowProgressGate(
            window=2,
            patience_windows=1,
            epsilon=0.0005,
            min_time=min(90.0, max(0.0, float(budget))),
            initial_best=best_proxy,
        )

        def apply_member_move(idx: int, ox: float, oy: float, nx: float, ny: float) -> None:
            sa_pos[idx, 0] = nx
            sa_pos[idx, 1] = ny
            _update_density_incr(
                density_grid, ox, oy, nx, ny,
                hw_np[idx], hh_np[idx], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[idx]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        sa_pos, pin_aff, idx, ox, oy,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                aff_route = macro_to_nets[idx]
                if len(aff_route) > 0:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        ni_np, nm_np, nw_np, sa_pos, aff_route, idx, ox, oy,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        def rebuild_all():
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            density_grid[:] = 0
            _build_density_grid(
                density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            current_den = _density_cost_top5(density_grid)
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_cong = cong_tracker.cost()
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale

        no_accept_limit = max(
            50_000,
            int(os.environ.get("HAP_COMMUNITY_NO_ACCEPT_LIMIT", "500000")),
        )

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break
            if accepts > 0 and accepts % 500 == 0 and accepts % checkpoint_every != 0:
                rebuild_all()

            if total % 50000 == 0 and total > 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                    f"fast={current_proxy:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            total += 1
            if total - last_accept_step > no_accept_limit:
                print(
                    f"Community relocate: no accepts in {no_accept_limit} attempts, stopping"
                )
                break

            cid = int(movable_arr[random.randint(0, movable_arr.size - 1)])
            members = communities[cid]
            n_members = members.size
            if n_members < 2:
                continue

            dx = random.uniform(-max_displacement, max_displacement)
            dy = random.uniform(-max_displacement, max_displacement)

            old_positions = np.empty((n_members, 2), dtype=np.float32)
            new_positions = np.empty((n_members, 2), dtype=np.float32)
            for idx, m in enumerate(members):
                m_int = int(m)
                ox = float(sa_pos[m_int, 0])
                oy = float(sa_pos[m_int, 1])
                old_positions[idx, 0] = ox
                old_positions[idx, 1] = oy
                new_positions[idx, 0] = float(
                    np.clip(ox + dx, hw_np[m_int], benchmark.canvas_width - hw_np[m_int])
                )
                new_positions[idx, 1] = float(
                    np.clip(oy + dy, hh_np[m_int], benchmark.canvas_height - hh_np[m_int])
                )

            for idx, m in enumerate(members):
                apply_member_move(
                    int(m),
                    float(old_positions[idx, 0]), float(old_positions[idx, 1]),
                    float(new_positions[idx, 0]), float(new_positions[idx, 1]),
                )

            new_den = _density_cost_top5(density_grid)
            new_cong = cong_tracker.cost()

            aff_lists = [macro_to_nets[int(m)] for m in members if len(macro_to_nets[int(m)]) > 0]
            if aff_lists:
                aff = np.unique(np.concatenate(aff_lists))
                old_hpwl = net_hpwl[aff].copy()
                new_hpwl = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
                wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)
            else:
                aff = np.array([], dtype=np.int64)
                old_hpwl = np.empty(0, dtype=np.float32)
                new_hpwl = np.empty(0, dtype=np.float32)
                wl_delta = 0.0

            new_proxy = (total_wl + wl_delta) + 0.5 * new_den * den_scale + 0.5 * new_cong * cong_scale

            if new_proxy < current_proxy:
                for idx, m in enumerate(members):
                    sa_placement[int(m), 0] = float(new_positions[idx, 0])
                    sa_placement[int(m), 1] = float(new_positions[idx, 1])
                if aff.size > 0:
                    net_hpwl[aff] = new_hpwl
                total_wl += wl_delta
                current_den = new_den
                current_cong = new_cong
                current_proxy = new_proxy
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                    real_proxy = metrics["proxy_cost"]
                    den_scale = metrics["density_cost"] / (current_den + 1e-8)
                    cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                    current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                        f"fast={current_proxy:.4f} real={real_proxy:.4f} "
                        f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                        f"[{time()-t0:.0f}s]"
                    )
                    _, gate_stop = progress_gate.update(real_proxy, time() - t0)
                    if real_proxy < best_proxy:
                        best_proxy = real_proxy
                        best_placement = sa_placement.clone()
                    if gate_stop:
                        print(
                            f"Community relocate stalled "
                            f"(window gate, best={best_proxy:.4f})"
                        )
                        break
            else:
                for idx in range(n_members - 1, -1, -1):
                    m = int(members[idx])
                    nx = float(new_positions[idx, 0])
                    ny = float(new_positions[idx, 1])
                    ox = float(old_positions[idx, 0])
                    oy = float(old_positions[idx, 1])
                    apply_member_move(m, nx, ny, ox, oy)

        print(
            f"Community relocate done: {total} attempts, {accepts} accepts, "
            f"best real_proxy={best_proxy:.4f}"
        )
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
        checkpoint_every=400,
    ):
        """
        Hard-macro swap refinement with incremental WL+density+HV proxy.
        Returns best placement found.
        """
        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
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
        _build_density_grid(
            density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )
        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()

        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        best_den_real = real_metrics["density_cost"]
        best_cong_real = real_metrics["congestion_cost"]
        den_scale = best_den_real / (current_den + 1e-8)
        cong_scale = best_cong_real / (current_cong + 1e-8)
        current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
        best_placement = sa_placement.clone()
        print(
            f"Hard swap start: proxy={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} (HV cong, smooth={smooth_range})"
        )

        accepts = total = stalls = 0
        t0 = time()
        last_accept_step = 0
        last_checkpoint_accepts = 0
        last_rebuild_accepts = 0

        def rebuild_fast_state(from_tensor):
            nonlocal sa_placement, sa_pos, net_hpwl, total_wl
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            sa_placement = from_tensor.clone()
            sa_pos[:] = sa_placement.detach().numpy()
            net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
            total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            density_grid[:] = 0
            _build_density_grid(
                density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            h_macro_grid[:] = 0
            v_macro_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_den = _density_cost_top5(density_grid)
            current_cong = cong_tracker.cost()
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale

        def apply_move(idx, old_x, old_y, new_x, new_y):
            sa_pos[idx, 0] = new_x
            sa_pos[idx, 1] = new_y
            _update_density_incr(
                density_grid, old_x, old_y, new_x, new_y,
                hw_np[idx], hh_np[idx], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[idx]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        sa_pos, pin_aff, idx, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                aff_route = macro_to_nets[idx]
                if len(aff_route) > 0:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        ni_np, nm_np, nw_np, sa_pos, aff_route, idx, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            _update_macro_route_incr_single(
                h_macro_grid, v_macro_grid, old_x, old_y, new_x, new_y,
                hw_np[idx], hh_np[idx], bl, br, bb, bt, n_rows, n_cols,
                bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )

        def checkpoint(label=""):
            nonlocal best_proxy, best_placement, best_den_real, best_cong_real
            nonlocal den_scale, cong_scale, current_proxy, stalls, last_checkpoint_accepts
            _set_placement(plc, sa_placement.detach(), benchmark)
            metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
            proxy = metrics["proxy_cost"]
            den_scale = metrics["density_cost"] / (current_den + 1e-8)
            cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
            last_checkpoint_accepts = accepts
            print(
                f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                f"fast={current_proxy:.4f} proxy={proxy:.4f} "
                f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                f"{label}[{time()-t0:.0f}s]"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_den_real = metrics["density_cost"]
                best_cong_real = metrics["congestion_cost"]
                best_placement = sa_placement.clone()
                stalls = 0
                return False
            rebuild_fast_state(best_placement)
            den_scale = best_den_real / (current_den + 1e-8)
            cong_scale = best_cong_real / (current_cong + 1e-8)
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
            stalls += 1
            return stalls >= 6

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break
            if (
                accepts > 0
                and accepts % 500 == 0
                and accepts != last_rebuild_accepts
                and accepts % checkpoint_every != 0
            ):
                rebuild_fast_state(sa_placement)
                last_rebuild_accepts = accepts
            if total % 100000 == 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                    f"fast={current_proxy:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )
            i = random.randint(0, num_hard - 1)
            j = random.randint(0, num_hard - 1)
            if i == j or fixed[i].item() or fixed[j].item():
                continue

            total += 1
            old_i_x = float(sa_pos[i, 0])
            old_i_y = float(sa_pos[i, 1])
            old_j_x = float(sa_pos[j, 0])
            old_j_y = float(sa_pos[j, 1])

            sa_pos[[i, j]] = sa_pos[[j, i]]
            if _swap_creates_overlap(sa_pos, i, j, sep_x_np, sep_y_np, num_hard):
                sa_pos[[i, j]] = sa_pos[[j, i]]
                continue
            sa_pos[[i, j]] = sa_pos[[j, i]]

            aff = np.union1d(macro_to_nets[i], macro_to_nets[j])
            old_vals = net_hpwl[aff].copy()

            apply_move(i, old_i_x, old_i_y, old_j_x, old_j_y)
            apply_move(j, old_j_x, old_j_y, old_i_x, old_i_y)

            new_vals = _hpwl_batch(aff, ni_np, nm_np, sa_pos) if aff.size else old_vals
            delta = float((new_vals - old_vals).sum()) / (num_nets * canvas_norm)
            new_den = _density_cost_top5(density_grid)
            new_cong = cong_tracker.cost()
            new_proxy = total_wl + delta + 0.5 * new_den * den_scale + 0.5 * new_cong * cong_scale

            if new_proxy <= current_proxy:
                sa_placement[i, 0] = float(sa_pos[i, 0])
                sa_placement[i, 1] = float(sa_pos[i, 1])
                sa_placement[j, 0] = float(sa_pos[j, 0])
                sa_placement[j, 1] = float(sa_pos[j, 1])
                net_hpwl[aff] = new_vals
                total_wl += delta
                current_den = new_den
                current_cong = new_cong
                current_proxy = new_proxy
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    if checkpoint():
                        print("Hard swap stalled")
                        break

            else:
                apply_move(i, old_j_x, old_j_y, old_i_x, old_i_y)
                apply_move(j, old_i_x, old_i_y, old_j_x, old_j_y)
            if total - last_accept_step > 5_000_000:
                print(f"Hard swap: no accepts in 5M attempts, stopping")
                break

        if accepts > last_checkpoint_accepts:
            if checkpoint(label="final_chk "):
                print("Hard swap stalled")

        print(f"Hard swap done: {total} attempts, {accepts} accepts, best proxy={best_proxy:.4f}")
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
        base_budget=90,
        extension_budget=60,
        extension_gain=0.002,
        checkpoint_every=300,
    ):
        """Greedy pairwise soft macro swaps with incremental WL+density+HV proxy."""
        num_soft = num_all - num_hard
        if num_soft < 2:
            return placement

        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )
        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
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
        _build_density_grid(
            density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        # Soft macros do not block routing; hard macro blockage is fixed.
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )
        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()

        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        best_den_real = real_metrics["density_cost"]
        best_cong_real = real_metrics["congestion_cost"]
        den_scale = best_den_real / (current_den + 1e-8)
        cong_scale = best_cong_real / (current_cong + 1e-8)
        current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
        best_placement = sa_placement.clone()
        start_wl = float(total_wl)
        wl_cap = start_wl + max(0.025, 0.35 * start_wl)
        initial_density = float(getattr(self, "_initial_density_cost", real_metrics["density_cost"]))
        initial_cong = float(getattr(self, "_initial_congestion_cost", real_metrics["congestion_cost"]))
        mode_override = os.environ.get("HAP_SOFT_SWAP_MODE", "").strip().lower()
        if mode_override in ("proxy", "fast", "cong", "real"):
            use_proxy_swap = True
            mode_reason = f"env:{mode_override}"
        elif mode_override in ("wl", "hpwl", "wl_greedy", "wirelength"):
            use_proxy_swap = False
            mode_reason = f"env:{mode_override}"
        else:
            use_proxy_swap = (
                (initial_cong >= 2.10 and initial_density >= 0.836)
                or (
                    initial_density < 0.836
                    and (290 <= num_hard < 700 or initial_cong > 2.30)
                )
                or num_nets >= 36_000
            )
            mode_reason = "feature_gate"
        soft_swap_mode = "proxy" if use_proxy_swap else "wl_greedy"
        print(
            f"Soft swap start: proxy={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} wl_cap={wl_cap:.4f} "
            f"num_soft={num_soft} mode={soft_swap_mode} reason={mode_reason} "
            f"hard={num_hard} nets={num_nets} init_den={initial_density:.4f} "
            f"init_cong={initial_cong:.4f} (HV cong, smooth={smooth_range})"
        )
        timing_enabled = os.environ.get("HAP_STAGE_TIMERS", "").strip().lower() not in (
            "", "0", "false", "no", "off"
        )
        ss_timers = {
            "proposal": 0.0,
            "hpwl": 0.0,
            "density_cong": 0.0,
            "move_apply": 0.0,
            "rollback": 0.0,
            "rebuild": 0.0,
            "real": 0.0,
        }

        accepts = total = stalls = 0
        last_accept_step = 0
        last_checkpoint_accepts = 0
        last_rebuild_accepts = 0
        t0 = time()
        max_budget = max(0.0, float(budget))
        active_budget = min(max_budget, max(0.0, float(base_budget)))
        extension_budget = max(0.0, float(extension_budget))
        extension_anchor = best_proxy
        if active_budget <= 0.0:
            print("Soft swap skipped: no active budget")
            return placement

        def rebuild_fast_state(from_tensor):
            nonlocal sa_placement, sa_pos, net_hpwl, total_wl
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            if timing_enabled:
                _timer_t = time()
            sa_placement = from_tensor.clone()
            sa_pos[:] = sa_placement.detach().numpy()
            net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
            total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            density_grid[:] = 0
            _build_density_grid(
                density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_den = _density_cost_top5(density_grid)
            current_cong = cong_tracker.cost()
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
            if timing_enabled:
                ss_timers["rebuild"] += time() - _timer_t

        def apply_move(idx, old_x, old_y, new_x, new_y):
            sa_pos[idx, 0] = new_x
            sa_pos[idx, 1] = new_y
            _update_density_incr(
                density_grid, old_x, old_y, new_x, new_y,
                hw_np[idx], hh_np[idx], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[idx]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        sa_pos, pin_aff, idx, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                aff_route = macro_to_nets[idx]
                if len(aff_route) > 0:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        ni_np, nm_np, nw_np, sa_pos, aff_route, idx, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        def checkpoint(label=""):
            nonlocal best_proxy, best_placement, best_den_real, best_cong_real
            nonlocal den_scale, cong_scale, current_proxy, stalls, extension_anchor
            nonlocal active_budget, last_checkpoint_accepts
            if timing_enabled:
                _timer_t = time()
            _set_placement(plc, sa_placement.detach(), benchmark)
            metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
            if timing_enabled:
                ss_timers["real"] += time() - _timer_t
            proxy = metrics["proxy_cost"]
            den_scale = metrics["density_cost"] / (current_den + 1e-8)
            cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
            last_checkpoint_accepts = accepts
            print(
                f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                f"fast={current_proxy:.4f} proxy={proxy:.4f} "
                f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                f"{label}[{time()-t0:.0f}s]"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_den_real = metrics["density_cost"]
                best_cong_real = metrics["congestion_cost"]
                best_placement = sa_placement.clone()
                stalls = 0
                if (
                    extension_anchor - best_proxy >= extension_gain
                    and extension_budget > 0.0
                    and active_budget < max_budget - 1e-6
                ):
                    old_budget = active_budget
                    active_budget = min(max_budget, active_budget + extension_budget)
                    extension_anchor = best_proxy
                    print(
                        f"  soft_swap extend: {old_budget:.0f}s->{active_budget:.0f}s "
                        f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
                    )
                return False
            rebuild_fast_state(best_placement)
            den_scale = best_den_real / (current_den + 1e-8)
            cong_scale = best_cong_real / (current_cong + 1e-8)
            current_proxy = total_wl + 0.5 * current_den * den_scale + 0.5 * current_cong * cong_scale
            stalls += 1
            return stalls >= 10

        for _ in range(100_000_000):
            if time() - t0 > active_budget:
                if accepts > last_checkpoint_accepts:
                    old_budget = active_budget
                    if checkpoint(label="time_chk "):
                        print("Soft swap stalled")
                        break
                    if active_budget > old_budget and time() - t0 <= active_budget:
                        continue
                break
            if (
                accepts > 0
                and accepts % 500 == 0
                and accepts != last_rebuild_accepts
                and accepts % checkpoint_every != 0
            ):
                rebuild_fast_state(sa_placement)
                last_rebuild_accepts = accepts
            if total % 100000 == 0 and total > 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                    f"fast={current_proxy:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            i = num_hard + random.randint(0, num_soft - 1)
            j = num_hard + random.randint(0, num_soft - 1)
            if i == j:
                continue

            total += 1
            if timing_enabled:
                _timer_t = time()
            old_i_x = float(sa_pos[i, 0])
            old_i_y = float(sa_pos[i, 1])
            old_j_x = float(sa_pos[j, 0])
            old_j_y = float(sa_pos[j, 1])

            aff = np.union1d(macro_to_nets[i], macro_to_nets[j])
            old_vals = net_hpwl[aff].copy()
            if timing_enabled:
                ss_timers["proposal"] += time() - _timer_t

            if timing_enabled:
                _timer_t = time()
            apply_move(i, old_i_x, old_i_y, old_j_x, old_j_y)
            apply_move(j, old_j_x, old_j_y, old_i_x, old_i_y)
            if timing_enabled:
                ss_timers["move_apply"] += time() - _timer_t

            if timing_enabled:
                _timer_t = time()
            new_vals = _hpwl_batch(aff, ni_np, nm_np, sa_pos) if aff.size else old_vals
            delta = float((new_vals - old_vals).sum()) / (num_nets * canvas_norm)
            if timing_enabled:
                ss_timers["hpwl"] += time() - _timer_t

            if timing_enabled:
                _timer_t = time()
            new_den = _density_cost_top5(density_grid)
            new_cong = cong_tracker.cost()
            if timing_enabled:
                ss_timers["density_cong"] += time() - _timer_t
            candidate_wl = total_wl + delta
            new_proxy = total_wl + delta + 0.5 * new_den * den_scale + 0.5 * new_cong * cong_scale
            wl_ok = candidate_wl <= wl_cap
            if total_wl > wl_cap:
                # If a rebuild/checkpoint exposes that we have drifted over the
                # soft-swap WL cap, require recovery rather than an immediate
                # return under cap. Otherwise the stage can freeze permanently.
                wl_ok = candidate_wl < total_wl

            if (
                (use_proxy_swap and new_proxy <= current_proxy and wl_ok)
                or ((not use_proxy_swap) and delta <= 0.0)
            ):
                sa_placement[i, 0] = float(sa_pos[i, 0])
                sa_placement[i, 1] = float(sa_pos[i, 1])
                sa_placement[j, 0] = float(sa_pos[j, 0])
                sa_placement[j, 1] = float(sa_pos[j, 1])
                net_hpwl[aff] = new_vals
                total_wl += delta
                current_den = new_den
                current_cong = new_cong
                current_proxy = new_proxy
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    if checkpoint():
                        print("Soft swap stalled")
                        break
            else:
                if timing_enabled:
                    _timer_t = time()
                apply_move(i, old_j_x, old_j_y, old_i_x, old_i_y)
                apply_move(j, old_i_x, old_i_y, old_j_x, old_j_y)
                if timing_enabled:
                    ss_timers["rollback"] += time() - _timer_t

            if total - last_accept_step > 2_000_000:
                print("Soft swap: no accepts in 2M attempts, stopping")
                break

        if accepts > last_checkpoint_accepts:
            if checkpoint(label="final_chk "):
                print("Soft swap stalled")

        print(f"Soft swap done: {total} attempts, {accepts} accepts, best proxy={best_proxy:.4f}")
        if timing_enabled:
            timed_total = max(1e-9, sum(ss_timers.values()))
            print(
                "Soft swap timing: "
                + " ".join(
                    f"{k}={v:.2f}s({100.0*v/timed_total:.0f}%)"
                    for k, v in ss_timers.items()
                    if v > 0.0
                )
                + f" timed={timed_total:.2f}s wall={time()-t0:.2f}s"
            )
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

    def _sa_soft_cd_refine(
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
        budget=240,
        base_budget=240,
        extension_budget=120,
        extension_gain=2e-3,
        checkpoint_every=60,
    ):
        """Soft-macro coordinate descent with incremental WL+density+HV proxy."""
        num_soft = num_all - num_hard
        if num_soft < 1:
            return placement

        cd_placement = placement.clone()
        cd_pos, macro_to_nets, net_hpwl, _eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, cd_placement, num_all, canvas_norm
        )

        soft_candidates = np.array(
            [
                i for i in range(num_hard, num_all)
                if (not fixed[i].item()) and len(macro_to_nets[i]) > 0
            ],
            dtype=np.int32,
        )
        if soft_candidates.size == 0:
            return placement

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = ni_np.shape[0]
        max_degree = ni_np.shape[1]
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
        _build_density_grid(
            density_grid, cd_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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

        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                cd_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, cd_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        # Soft macros do not block routing; hard blockage is fixed in this stage.
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, cd_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )

        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()
        _set_placement(plc, cd_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        den_scale = real_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = real_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = (
            total_wl
            + 0.5 * current_den * den_scale
            + 0.5 * current_cong * cong_scale
        )
        best_placement = cd_placement.clone()
        print(
            f"Soft CD start: proxy={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} candidates={soft_candidates.size} "
            f"(HV cong, smooth={smooth_range})"
        )
        route_aware_soft_cd = os.environ.get("HAP_ROUTE_AWARE_SOFT_CD", "1").strip().lower() not in (
            "0", "false", "no", "off"
        )
        if not route_aware_soft_cd:
            print("Soft CD route-aware candidates disabled by HAP_ROUTE_AWARE_SOFT_CD=0")
        timing_enabled = os.environ.get("HAP_STAGE_TIMERS", "").strip().lower() not in (
            "", "0", "false", "no", "off"
        )
        cd_timers = {
            "order": 0.0,
            "candidate_gen": 0.0,
            "hpwl_batch": 0.0,
            "incr_eval": 0.0,
            "commit": 0.0,
            "rebuild": 0.0,
            "real": 0.0,
        }

        def rebuild_fast_state(from_tensor):
            nonlocal cd_placement, cd_pos, net_hpwl, total_wl
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            if timing_enabled:
                _timer_t = time()
            cd_placement = from_tensor.clone()
            cd_pos[:] = cd_placement.detach().numpy()
            net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, cd_pos)
            total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            density_grid[:] = 0
            _build_density_grid(
                density_grid, cd_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    cd_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, cd_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_den = _density_cost_top5(density_grid)
            current_cong = cong_tracker.cost()
            current_proxy = (
                total_wl
                + 0.5 * current_den * den_scale
                + 0.5 * current_cong * cong_scale
            )
            if timing_enabled:
                cd_timers["rebuild"] += time() - _timer_t

        def apply_one(i, old_x, old_y, new_x, new_y):
            cd_pos[i, 0] = new_x
            cd_pos[i, 1] = new_y
            _update_density_incr(
                density_grid, old_x, old_y, new_x, new_y,
                hw_np[i], hh_np[i], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[i]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        cd_pos, pin_aff, i, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                aff_i = macro_to_nets[i]
                if len(aff_i) > 0:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        ni_np, nm_np, nw_np, cd_pos, aff_i, i, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        def macro_barycenter(i):
            sx = sy = sw = 0.0
            for n in macro_to_nets[i]:
                w = float(nw_np[n])
                for d in range(max_degree):
                    if not nm_np[n, d]:
                        break
                    j = int(ni_np[n, d])
                    if j == i:
                        continue
                    sx += float(cd_pos[j, 0]) * w
                    sy += float(cd_pos[j, 1]) * w
                    sw += w
            if sw <= 0.0:
                return float(cd_pos[i, 0]), float(cd_pos[i, 1])
            return sx / sw, sy / sw

        def local_pressure_grid():
            return density_grid + 0.5 * (cong_tracker.h_smooth + cong_tracker.v_smooth)

        def route_pressure_grids():
            h_total = cong_tracker.h_smooth + h_macro_grid
            v_total = cong_tracker.v_smooth + v_macro_grid
            return h_total, v_total, np.maximum(h_total, v_total)

        def soft_order():
            rows = np.clip((cd_pos[soft_candidates, 1] / bin_h).astype(np.int64), 0, n_rows - 1)
            cols = np.clip((cd_pos[soft_candidates, 0] / bin_w).astype(np.int64), 0, n_cols - 1)
            pressure = local_pressure_grid()[rows, cols]
            if route_aware_soft_cd:
                h_total, v_total, route_pressure = route_pressure_grids()
                pressure = pressure + 0.35 * route_pressure[rows, cols]
            degree = np.array([len(macro_to_nets[int(i)]) for i in soft_candidates], dtype=np.float32)
            score = pressure + 0.01 * np.sqrt(degree) + 0.002 * np.random.random(size=pressure.shape)
            visit_n = min(soft_candidates.size, 320)
            if soft_candidates.size <= visit_n:
                order = np.arange(soft_candidates.size)
            else:
                order = np.argpartition(score, -visit_n)[-visit_n:]
            order = order[np.argsort(-score[order])]
            return soft_candidates[order]

        def candidate_positions(i, step_scale):
            x = float(cd_pos[i, 0])
            y = float(cd_pos[i, 1])
            out = []
            for s in (1.0, 0.5, 0.25):
                d = step_scale * s
                out.extend(((x + d, y), (x - d, y), (x, y + d), (x, y - d)))

            bx, by = macro_barycenter(i)
            for frac in (0.35, 0.65, 1.0):
                out.append((x + frac * (bx - x), y + frac * (by - y)))

            pressure = local_pressure_grid()
            r = int(np.clip(y / bin_h, 1, n_rows - 2))
            c = int(np.clip(x / bin_w, 1, n_cols - 2))
            gx = float(pressure[r, c + 1] - pressure[r, c - 1])
            gy = float(pressure[r + 1, c] - pressure[r - 1, c])
            gnorm = math.hypot(gx, gy)
            if gnorm > 1e-8:
                out.append((x - step_scale * gx / gnorm, y - step_scale * gy / gnorm))
                out.append((x - 0.5 * step_scale * gx / gnorm, y - 0.5 * step_scale * gy / gnorm))

            if route_aware_soft_cd:
                # Congestion-specific candidates: if this macro is sitting on a hot
                # horizontal/vertical route stripe, try moving perpendicular to that
                # stripe into nearby cooler rows/columns while staying biased toward
                # the macro's net barycenter. This is a gentler version of net shear:
                # single-macro moves, real-proxy guarded by the existing CD loop.
                h_total, v_total, _route_pressure = route_pressure_grids()
                h_here = float(h_total[r, c])
                v_here = float(v_total[r, c])
                h_thresh = float(h_total.mean() + 0.25 * h_total.std())
                v_thresh = float(v_total.mean() + 0.25 * v_total.std())
                x_pull = x + 0.45 * (bx - x)
                y_pull = y + 0.45 * (by - y)

                if h_here >= h_thresh and n_rows > 2:
                    radius = max(2, min(n_rows - 1, int(math.ceil(2.5 * step_scale / max(bin_h, 1e-8)))))
                    r0h = max(0, r - radius)
                    r1h = min(n_rows - 1, r + radius)
                    c0h = max(0, c - 1)
                    c1h = min(n_cols - 1, c + 1)
                    row_pressure = h_total[r0h : r1h + 1, c0h : c1h + 1].mean(axis=1)
                    low_n = min(3, row_pressure.size)
                    if low_n > 0:
                        low_rows = np.argpartition(row_pressure, low_n - 1)[:low_n]
                        for rr_local in low_rows:
                            rr = r0h + int(rr_local)
                            yy = (rr + 0.5) * bin_h
                            out.append((x, yy))
                            out.append((x_pull, yy))

                if v_here >= v_thresh and n_cols > 2:
                    radius = max(2, min(n_cols - 1, int(math.ceil(2.5 * step_scale / max(bin_w, 1e-8)))))
                    c0v = max(0, c - radius)
                    c1v = min(n_cols - 1, c + radius)
                    r0v = max(0, r - 1)
                    r1v = min(n_rows - 1, r + 1)
                    col_pressure = v_total[r0v : r1v + 1, c0v : c1v + 1].mean(axis=0)
                    low_n = min(3, col_pressure.size)
                    if low_n > 0:
                        low_cols = np.argpartition(col_pressure, low_n - 1)[:low_n]
                        for cc_local in low_cols:
                            cc = c0v + int(cc_local)
                            xx = (cc + 0.5) * bin_w
                            out.append((xx, y))
                            out.append((xx, y_pull))

            win = max(step_scale * 1.5, max(bin_w, bin_h) * 2.0)
            c0 = max(0, int((x - win) / bin_w))
            c1 = min(n_cols - 1, int((x + win) / bin_w))
            r0 = max(0, int((y - win) / bin_h))
            r1 = min(n_rows - 1, int((y + win) / bin_h))
            sub = pressure[r0 : r1 + 1, c0 : c1 + 1]
            if sub.size > 0:
                low_n = min(6, sub.size)
                low = np.argpartition(sub.reshape(-1), low_n - 1)[:low_n]
                for flat_idx in low:
                    rr = r0 + int(flat_idx) // sub.shape[1]
                    cc = c0 + int(flat_idx) % sub.shape[1]
                    out.append(((cc + 0.5) * bin_w, (rr + 0.5) * bin_h))

            clipped = []
            seen = set()
            for cx, cy in out:
                cx = float(np.clip(cx, hw_np[i], benchmark.canvas_width - hw_np[i]))
                cy = float(np.clip(cy, hh_np[i], benchmark.canvas_height - hh_np[i]))
                key = (round(cx, 4), round(cy, 4))
                if key != (round(x, 4), round(y, 4)) and key not in seen:
                    clipped.append((cx, cy))
                    seen.add(key)
            return clipped

        accepts = evals = stalls = sweeps = 0
        real_misses = 0
        last_checkpoint_accepts = 0
        t0 = time()
        last_real_check_time = 0.0
        max_budget = max(0.0, float(budget))
        active_budget = min(max_budget, max(0.0, float(base_budget)))
        extension_budget = max(0.0, float(extension_budget))
        extension_anchor = best_proxy
        real_miss_limit = max(1, int(os.environ.get("HAP_SOFT_CD_REAL_MISS_LIMIT", "2")))
        real_checkpoint_seconds = max(
            20.0, float(os.environ.get("HAP_SOFT_CD_REAL_CKPT_SECONDS", "90"))
        )
        progress_gate = WindowProgressGate(
            window=5,
            patience_windows=2,
            epsilon=0.002,
            min_time=min(240.0, active_budget),
            initial_best=best_proxy,
        )
        if active_budget <= 0.0:
            print("Soft CD skipped: no active budget")
            return placement

        stop_cd = False
        while time() - t0 < active_budget:
            sweeps += 1
            sweep_accepts = 0
            if timing_enabled:
                _timer_t = time()
            order = soft_order()
            if timing_enabled:
                cd_timers["order"] += time() - _timer_t
            elapsed_frac = min(1.0, (time() - t0) / max(active_budget, 1e-6))
            step_scale = canvas_norm * (0.025 * (1.0 - elapsed_frac) + 0.004 * elapsed_frac)

            for i_raw in order:
                if time() - t0 > active_budget:
                    break
                i = int(i_raw)
                aff = macro_to_nets[i]
                if len(aff) == 0:
                    continue

                old_x = float(cd_pos[i, 0])
                old_y = float(cd_pos[i, 1])
                old_hpwl = net_hpwl[aff].copy()

                best_move = None
                best_move_proxy = current_proxy
                best_move_hpwl = None
                best_move_delta = 0.0
                best_move_den = current_den
                best_move_cong = current_cong

                if timing_enabled:
                    _timer_t = time()
                candidates = candidate_positions(i, step_scale)
                if timing_enabled:
                    cd_timers["candidate_gen"] += time() - _timer_t
                if not candidates:
                    continue
                if timing_enabled:
                    _timer_t = time()
                cand_xy = np.asarray(candidates, dtype=np.float32)
                cand_hpwl = _hpwl_candidate_batch_for_macro(
                    cand_xy[:, 0], cand_xy[:, 1], i, aff, ni_np, nm_np, cd_pos
                )
                if timing_enabled:
                    cd_timers["hpwl_batch"] += time() - _timer_t

                for cand_idx, (new_x, new_y) in enumerate(candidates):
                    evals += 1
                    if timing_enabled:
                        _timer_t = time()
                    apply_one(i, old_x, old_y, new_x, new_y)
                    new_hpwl = cand_hpwl[cand_idx]
                    wl_delta = float(((new_hpwl - old_hpwl) * nw_np[aff]).sum()) / (
                        num_nets * canvas_norm
                    )
                    new_den = _density_cost_top5(density_grid)
                    new_cong = cong_tracker.cost()
                    cand_proxy = (
                        total_wl + wl_delta
                        + 0.5 * new_den * den_scale
                        + 0.5 * new_cong * cong_scale
                    )
                    apply_one(i, new_x, new_y, old_x, old_y)
                    if timing_enabled:
                        cd_timers["incr_eval"] += time() - _timer_t

                    if cand_proxy < best_move_proxy - 1e-6:
                        best_move = (new_x, new_y)
                        best_move_proxy = cand_proxy
                        best_move_hpwl = new_hpwl.copy()
                        best_move_delta = wl_delta
                        best_move_den = new_den
                        best_move_cong = new_cong

                if best_move is None:
                    continue

                new_x, new_y = best_move
                if timing_enabled:
                    _timer_t = time()
                apply_one(i, old_x, old_y, new_x, new_y)
                cd_placement[i, 0] = new_x
                cd_placement[i, 1] = new_y
                net_hpwl[aff] = best_move_hpwl
                total_wl += best_move_delta
                current_den = best_move_den
                current_cong = best_move_cong
                current_proxy = best_move_proxy
                accepts += 1
                sweep_accepts += 1
                if timing_enabled:
                    cd_timers["commit"] += time() - _timer_t

                if accepts % checkpoint_every == 0:
                    if timing_enabled:
                        _timer_t = time()
                    _set_placement(plc, cd_placement.detach(), benchmark)
                    metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
                    if timing_enabled:
                        cd_timers["real"] += time() - _timer_t
                    proxy = metrics["proxy_cost"]
                    print(
                        f"  soft_cd sweep={sweeps} accepts={accepts} evals={evals} "
                        f"fast={current_proxy:.4f} proxy={proxy:.4f} "
                        f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                        f"[{time()-t0:.0f}s]"
                    )
                    last_checkpoint_accepts = accepts
                    last_real_check_time = time() - t0
                    den_scale = metrics["density_cost"] / (current_den + 1e-8)
                    cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                    current_proxy = (
                        total_wl
                        + 0.5 * current_den * den_scale
                        + 0.5 * current_cong * cong_scale
                    )
                    _, gate_stop = progress_gate.update(proxy, time() - t0)
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = cd_placement.clone()
                        stalls = 0
                        real_misses = 0
                        if (
                            extension_anchor - best_proxy >= extension_gain
                            and extension_budget > 0.0
                            and active_budget < max_budget - 1e-6
                        ):
                            old_budget = active_budget
                            active_budget = min(max_budget, active_budget + extension_budget)
                            extension_anchor = best_proxy
                            print(
                                f"  soft_cd extend: {old_budget:.0f}s->{active_budget:.0f}s "
                                f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
                            )
                    else:
                        real_misses += 1
                        rebuild_fast_state(best_placement)
                        if timing_enabled:
                            _timer_t = time()
                        _set_placement(plc, cd_placement.detach(), benchmark)
                        best_metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
                        if timing_enabled:
                            cd_timers["real"] += time() - _timer_t
                        den_scale = best_metrics["density_cost"] / (current_den + 1e-8)
                        cong_scale = best_metrics["congestion_cost"] / (current_cong + 1e-8)
                        current_proxy = (
                            total_wl
                            + 0.5 * current_den * den_scale
                            + 0.5 * current_cong * cong_scale
                        )
                    if gate_stop:
                        print(
                            f"Soft CD stalled "
                            f"(window gate, best={progress_gate.best:.4f}, "
                            f"patience={progress_gate.patience}/{progress_gate.max_patience})"
                        )
                        stop_cd = True
                        break
                    if real_misses >= real_miss_limit:
                        print(
                            f"Soft CD stalled "
                            f"(real miss limit {real_misses}/{real_miss_limit}, "
                            f"best={best_proxy:.4f})"
                        )
                        stop_cd = True
                        break

            if stop_cd or stalls >= 6:
                break
            print(
                f"  soft_cd sweep={sweeps} sweep_accepts={sweep_accepts} "
                f"accepts={accepts} evals={evals} fast={current_proxy:.4f} "
                f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
            )
            if (
                sweep_accepts > 0
                and accepts > last_checkpoint_accepts
                and (time() - t0 - last_real_check_time) >= real_checkpoint_seconds
            ):
                if timing_enabled:
                    _timer_t = time()
                _set_placement(plc, cd_placement.detach(), benchmark)
                metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
                if timing_enabled:
                    cd_timers["real"] += time() - _timer_t
                proxy = metrics["proxy_cost"]
                print(
                    f"  soft_cd time_chk accepts={accepts} evals={evals} "
                    f"proxy={proxy:.4f} best={best_proxy:.4f} [{time()-t0:.0f}s]"
                )
                last_checkpoint_accepts = accepts
                last_real_check_time = time() - t0
                den_scale = metrics["density_cost"] / (current_den + 1e-8)
                cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                current_proxy = (
                    total_wl
                    + 0.5 * current_den * den_scale
                    + 0.5 * current_cong * cong_scale
                )
                _, gate_stop = progress_gate.update(proxy, time() - t0)
                if proxy < best_proxy:
                    best_proxy = proxy
                    best_placement = cd_placement.clone()
                    stalls = 0
                    real_misses = 0
                    if (
                        extension_anchor - best_proxy >= extension_gain
                        and extension_budget > 0.0
                        and active_budget < max_budget - 1e-6
                    ):
                        old_budget = active_budget
                        active_budget = min(max_budget, active_budget + extension_budget)
                        extension_anchor = best_proxy
                        print(
                            f"  soft_cd extend: {old_budget:.0f}s->{active_budget:.0f}s "
                            f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
                        )
                    continue
                real_misses += 1
                rebuild_fast_state(best_placement)
                if timing_enabled:
                    _timer_t = time()
                _set_placement(plc, cd_placement.detach(), benchmark)
                best_metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
                if timing_enabled:
                    cd_timers["real"] += time() - _timer_t
                den_scale = best_metrics["density_cost"] / (current_den + 1e-8)
                cong_scale = best_metrics["congestion_cost"] / (current_cong + 1e-8)
                current_proxy = (
                    total_wl
                    + 0.5 * current_den * den_scale
                    + 0.5 * current_cong * cong_scale
                )
                if gate_stop:
                    print(
                        f"Soft CD stalled "
                        f"(window gate, best={progress_gate.best:.4f}, "
                        f"patience={progress_gate.patience}/{progress_gate.max_patience})"
                    )
                    break
                if real_misses >= real_miss_limit:
                    print(
                        f"Soft CD stalled "
                        f"(real miss limit {real_misses}/{real_miss_limit}, "
                        f"best={best_proxy:.4f})"
                    )
                    break
            if sweep_accepts == 0:
                if accepts > last_checkpoint_accepts:
                    if timing_enabled:
                        _timer_t = time()
                    _set_placement(plc, cd_placement.detach(), benchmark)
                    metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
                    if timing_enabled:
                        cd_timers["real"] += time() - _timer_t
                    proxy = metrics["proxy_cost"]
                    print(
                        f"  soft_cd no_move_chk accepts={accepts} evals={evals} "
                        f"proxy={proxy:.4f} best={best_proxy:.4f} [{time()-t0:.0f}s]"
                    )
                    last_checkpoint_accepts = accepts
                    last_real_check_time = time() - t0
                    den_scale = metrics["density_cost"] / (current_den + 1e-8)
                    cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                    current_proxy = (
                        total_wl
                        + 0.5 * current_den * den_scale
                        + 0.5 * current_cong * cong_scale
                    )
                    _, gate_stop = progress_gate.update(proxy, time() - t0)
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = cd_placement.clone()
                        stalls = 0
                        real_misses = 0
                        if (
                            extension_anchor - best_proxy >= extension_gain
                            and extension_budget > 0.0
                            and active_budget < max_budget - 1e-6
                        ):
                            old_budget = active_budget
                            active_budget = min(max_budget, active_budget + extension_budget)
                            extension_anchor = best_proxy
                            print(
                                f"  soft_cd extend: {old_budget:.0f}s->{active_budget:.0f}s "
                                f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
                        )
                        continue
                    real_misses += 1
                    if gate_stop:
                        print(
                            f"Soft CD stalled "
                            f"(window gate, best={progress_gate.best:.4f}, "
                            f"patience={progress_gate.patience}/{progress_gate.max_patience})"
                        )
                        break
                    if real_misses >= real_miss_limit:
                        print(
                            f"Soft CD stalled "
                            f"(real miss limit {real_misses}/{real_miss_limit}, "
                            f"best={best_proxy:.4f})"
                        )
                        break
                stalls += 1
                if stalls >= 8:
                    print("Soft CD: no sweep improvements, stopping")
                    break

        if accepts > last_checkpoint_accepts:
            if timing_enabled:
                _timer_t = time()
            _set_placement(plc, cd_placement.detach(), benchmark)
            metrics = compute_proxy_cost(cd_placement.detach(), benchmark, plc)
            if timing_enabled:
                cd_timers["real"] += time() - _timer_t
            proxy = metrics["proxy_cost"]
            print(
                f"  soft_cd final_chk accepts={accepts} evals={evals} "
                f"proxy={proxy:.4f} best={best_proxy:.4f} [{time()-t0:.0f}s]"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_placement = cd_placement.clone()

        print(
            f"Soft CD done: sweeps={sweeps} accepts={accepts} "
            f"evals={evals} best proxy={best_proxy:.4f}"
        )
        if timing_enabled:
            timed_total = max(1e-9, sum(cd_timers.values()))
            print(
                "Soft CD timing: "
                + " ".join(
                    f"{k}={v:.2f}s({100.0*v/timed_total:.0f}%)"
                    for k, v in cd_timers.items()
                    if v > 0.0
                )
                + f" timed={timed_total:.2f}s wall={time()-t0:.2f}s"
        )
        return best_placement

    def _soft_perturb_escape(
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
        budget=300,
    ):
        """Soft-only basin hop: shake a large soft subset, repair, keep if real proxy improves."""
        num_soft = num_all - num_hard
        if num_soft < 1 or budget <= 0:
            return placement

        trials = max(1, int(os.environ.get("HAP_ESCAPE_TRIALS", "2")))
        frac = min(1.0, max(0.02, float(os.environ.get("HAP_ESCAPE_FRAC", "0.45"))))
        sigma = max(0.001, float(os.environ.get("HAP_ESCAPE_SIGMA", "0.06")))
        repair_frac = min(0.90, max(0.20, float(os.environ.get("HAP_ESCAPE_REPAIR_FRAC", "0.70"))))
        trial_budget = max(20.0, float(budget) * repair_frac / trials)

        _set_placement(plc, placement.detach(), benchmark)
        best_metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
        best_proxy = float(best_metrics["proxy_cost"])
        best_placement = placement.clone()
        rng = np.random.default_rng(self.seed + 91073 + int(num_soft))
        hw = benchmark.macro_sizes[:, 0] / 2
        hh = benchmark.macro_sizes[:, 1] / 2
        soft_indices = np.array(
            [i for i in range(num_hard, num_all) if not fixed[i].item()],
            dtype=np.int32,
        )
        if soft_indices.size == 0:
            return placement

        print(
            f"Perturb escape start: proxy={best_proxy:.4f} trials={trials} "
            f"frac={frac:.2f} sigma={sigma:.3f} repair_budget={trial_budget:.0f}s"
        )
        t0 = time()
        for trial in range(1, trials + 1):
            if time() - t0 > budget:
                break
            cand = best_placement.clone()
            k = max(1, int(round(frac * soft_indices.size)))
            chosen = rng.choice(soft_indices, size=min(k, soft_indices.size), replace=False)
            chosen_t = torch.as_tensor(chosen, dtype=torch.long)
            dx = rng.normal(0.0, sigma * benchmark.canvas_width, size=chosen.size)
            dy = rng.normal(0.0, sigma * benchmark.canvas_height, size=chosen.size)
            cand[chosen_t, 0] = torch.clamp(
                cand[chosen_t, 0] + torch.tensor(dx, dtype=cand.dtype),
                min=hw[chosen_t],
                max=benchmark.canvas_width - hw[chosen_t],
            )
            cand[chosen_t, 1] = torch.clamp(
                cand[chosen_t, 1] + torch.tensor(dy, dtype=cand.dtype),
                min=hh[chosen_t],
                max=benchmark.canvas_height - hh[chosen_t],
            )

            _set_placement(plc, cand.detach(), benchmark)
            shaken_proxy = float(compute_proxy_cost(cand.detach(), benchmark, plc)["proxy_cost"])
            print(
                f"  escape trial={trial} shaken={shaken_proxy:.4f} "
                f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
            )
            repair_budget = min(trial_budget, max(1.0, budget - (time() - t0)))
            repaired = self._sa_soft_displace(
                cand,
                benchmark,
                plc,
                net_indices,
                net_mask,
                net_weights,
                canvas_norm,
                num_hard,
                num_all,
                fixed,
                budget=repair_budget,
                checkpoint_every=200,
            )
            _set_placement(plc, repaired.detach(), benchmark)
            metrics = compute_proxy_cost(repaired.detach(), benchmark, plc)
            proxy = float(metrics["proxy_cost"])
            print(
                f"  escape trial={trial} repaired={proxy:.4f} "
                f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                f"best={best_proxy:.4f} [{time()-t0:.0f}s]"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_placement = repaired.clone()

        print(f"Perturb escape done: best proxy={best_proxy:.4f}")
        return best_placement

    def _soft_net_shear_refine(
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
        budget=240,
    ):
        """
        Open hot routing rows/columns with coordinated soft-macro shears.

        Soft untwist permutes macros inside existing slots, which helps crossed
        local spokes but cannot open a straight bundle of over-used routing bins.
        Net shear treats the hot bin as the object: find soft macros whose local
        nets are likely crossing the hot stripe, push them away from that stripe
        as a group, and keep only real-proxy-improving candidates.
        """
        num_soft = num_all - num_hard
        if num_soft < 4 or plc is None or budget <= 0:
            return placement

        cur = placement.clone()
        pos = cur.detach().numpy().astype(np.float32).copy()
        _, macro_to_nets, _net_hpwl, _eval_delta, _total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, cur, num_all, canvas_norm
        )
        soft_candidates = np.array(
            [
                i for i in range(num_hard, num_all)
                if (not fixed[i].item()) and len(macro_to_nets[i]) > 0
            ],
            dtype=np.int32,
        )
        if soft_candidates.size < 4:
            return placement

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = ni_np.shape[0]
        max_degree = ni_np.shape[1]
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

        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            num_pin_nets = pin_owner_p.shape[0]
        else:
            pin_owner_p = pin_mask_p = pin_xoff_p = pin_yoff_p = pin_fx_p = pin_fy_p = nw_p = None
            num_pin_nets = 0

        all_net_ids = np.arange(num_nets, dtype=np.int32)

        def build_fast_state(pos_np):
            density_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_density_grid(
                density_grid, pos_np, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            v_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    pos_np, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos_np, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            h_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            v_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, pos_np, sizes_np, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            hpwl = _hpwl_batch(all_net_ids, ni_np, nm_np, pos_np)
            wl = float((hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            den = _density_cost_top5(density_grid)
            cong = cong_tracker.cost()
            h_total = cong_tracker.h_smooth + h_macro_grid
            v_total = cong_tracker.v_smooth + v_macro_grid
            route_pressure = np.maximum(h_total, v_total)
            return wl, den, cong, density_grid, h_total, v_total, route_pressure

        wl0, den0, cong0, _density0, h_total0, v_total0, pressure0 = build_fast_state(pos)
        _set_placement(plc, cur.detach(), benchmark)
        metrics0 = compute_proxy_cost(cur.detach(), benchmark, plc)
        best_proxy = float(metrics0["proxy_cost"])
        den_scale = float(metrics0["density_cost"]) / (den0 + 1e-8)
        cong_scale = float(metrics0["congestion_cost"]) / (cong0 + 1e-8)
        current_fast = wl0 + 0.5 * den0 * den_scale + 0.5 * cong0 * cong_scale
        best_placement = cur.clone()
        best_pos = pos.copy()
        print(
            f"Net shear start: proxy={best_proxy:.4f} fast={current_fast:.4f} "
            f"wl={wl0:.4f} den={metrics0['density_cost']:.4f} "
            f"cong={metrics0['congestion_cost']:.4f} candidates={soft_candidates.size}"
        )

        bary_cache = {}

        def macro_barycenter(i, pos_np):
            key = (int(i), id(pos_np))
            if key in bary_cache:
                return bary_cache[key]
            sx = sy = sw = 0.0
            for n in macro_to_nets[int(i)]:
                w = float(nw_np[n])
                for d in range(max_degree):
                    if not nm_np[n, d]:
                        break
                    j = int(ni_np[n, d])
                    if j == i:
                        continue
                    sx += float(pos_np[j, 0]) * w
                    sy += float(pos_np[j, 1]) * w
                    sw += w
            if sw <= 1e-9:
                out = (float(pos_np[i, 0]), float(pos_np[i, 1]))
            else:
                out = (sx / sw, sy / sw)
            bary_cache[key] = out
            return out

        def hot_bins(h_total, v_total, pressure):
            flat = pressure.reshape(-1)
            hot_n = min(24, flat.size)
            if hot_n <= 0:
                return []
            idx = np.argpartition(flat, -hot_n)[-hot_n:]
            idx = idx[np.argsort(-flat[idx])]
            out = []
            seen = set()
            for flat_idx in idx:
                r = int(flat_idx) // n_cols
                c = int(flat_idx) % n_cols
                axis = "V" if float(v_total[r, c]) >= float(h_total[r, c]) else "H"
                # Merge adjacent bins so one stripe does not consume all probes.
                key = (axis, r // 2 if axis == "H" else r, c // 2 if axis == "V" else c)
                if key in seen:
                    continue
                seen.add(key)
                out.append((r, c, axis, float(flat[flat_idx])))
                if len(out) >= 14:
                    break
            return out

        def group_for_hot_bin(pos_np, r, c, axis):
            cx = (c + 0.5) * bin_w
            cy = (r + 0.5) * bin_h
            scores = []
            for i_raw in soft_candidates:
                i = int(i_raw)
                x = float(pos_np[i, 0])
                y = float(pos_np[i, 1])
                bx, by = macro_barycenter(i, pos_np)
                if axis == "V":
                    cross = (x - cx) * (bx - cx) <= 0.0
                    near = abs(x - cx) <= 3.0 * bin_w
                    dist = abs(x - cx) / max(bin_w, 1e-8) + 0.22 * abs(y - cy) / max(bin_h, 1e-8)
                else:
                    cross = (y - cy) * (by - cy) <= 0.0
                    near = abs(y - cy) <= 3.0 * bin_h
                    dist = abs(y - cy) / max(bin_h, 1e-8) + 0.22 * abs(x - cx) / max(bin_w, 1e-8)
                if not cross and not near:
                    continue
                degree_bonus = 0.02 * math.sqrt(max(1, len(macro_to_nets[i])))
                scores.append((dist - degree_bonus - (0.7 if cross else 0.0), i))
            if len(scores) < 4:
                return np.empty(0, dtype=np.int32)
            scores.sort(key=lambda x: x[0])
            k = min(22, max(6, min(len(scores), 14)))
            return np.array([i for _score, i in scores[:k]], dtype=np.int32)

        def shear_targets(group, pos_np, r, c, axis, strength):
            targets = pos_np[group].copy()
            cx = (c + 0.5) * bin_w
            cy = (r + 0.5) * bin_h
            for local, i_raw in enumerate(group):
                i = int(i_raw)
                x = float(pos_np[i, 0])
                y = float(pos_np[i, 1])
                bx, by = macro_barycenter(i, pos_np)
                if axis == "V":
                    side = x - cx
                    if abs(side) < 0.35 * bin_w and abs(bx - cx) > 0.35 * bin_w:
                        side = bx - cx
                    sign = -1.0 if side < 0.0 else 1.0
                    dx = sign * strength * bin_w
                    # Light y alignment reduces pin-spoke overlap after opening the stripe.
                    dy = 0.12 * (by - y)
                else:
                    side = y - cy
                    if abs(side) < 0.35 * bin_h and abs(by - cy) > 0.35 * bin_h:
                        side = by - cy
                    sign = -1.0 if side < 0.0 else 1.0
                    dx = 0.12 * (bx - x)
                    dy = sign * strength * bin_h
                nx = float(np.clip(x + dx, hw_np[i], benchmark.canvas_width - hw_np[i]))
                ny = float(np.clip(y + dy, hh_np[i], benchmark.canvas_height - hh_np[i]))
                targets[local, 0] = nx
                targets[local, 1] = ny
            return targets

        t0 = time()
        passes = 0
        accepts = 0
        fast_evals = 0
        real_evals = 1
        real_eval_cap = max(8, min(30, int(max(1.0, budget) // 8)))

        while time() - t0 < budget and passes < 3 and real_evals < real_eval_cap:
            passes += 1
            bary_cache.clear()
            wl, den, cong, _density, h_total, v_total, pressure = build_fast_state(best_pos)
            current_fast = wl + 0.5 * den * den_scale + 0.5 * cong * cong_scale
            bins = hot_bins(h_total, v_total, pressure)
            proposals = []

            for r, c, axis, hot in bins:
                if time() - t0 > budget:
                    break
                group = group_for_hot_bin(best_pos, r, c, axis)
                if group.size < 4:
                    continue
                base_hot = hot
                for strength in (0.55, 1.0, 1.6, 2.3):
                    cand_pos = best_pos.copy()
                    targets = shear_targets(group, best_pos, r, c, axis, strength)
                    cand_pos[group] = targets
                    fast_wl, fast_den, fast_cong, _d, cand_h, cand_v, cand_pressure = build_fast_state(cand_pos)
                    fast_evals += 1
                    fast_proxy = fast_wl + 0.5 * fast_den * den_scale + 0.5 * fast_cong * cong_scale
                    new_hot = float(max(cand_h[r, c], cand_v[r, c]))
                    hot_drop = max(0.0, base_hot - new_hot)
                    cong_drop = max(0.0, cong - fast_cong)
                    den_drop = max(0.0, den - fast_den)
                    wl_rise = max(0.0, fast_wl - wl)
                    allow = 0.012 + 0.18 * hot_drop + 0.08 * cong_drop
                    if fast_proxy <= current_fast + allow and wl_rise <= 0.018:
                        score = (
                            fast_proxy
                            - 0.20 * hot_drop * cong_scale
                            - 0.10 * cong_drop * cong_scale
                            - 0.03 * den_drop * den_scale
                        )
                        proposals.append((
                            score, fast_proxy, group.copy(), targets.copy(), axis,
                            r, c, strength, fast_wl, fast_den, fast_cong,
                            hot_drop, cong_drop,
                        ))

            if not proposals:
                print(
                    f"  net_shear pass={passes} no plausible shears "
                    f"bins={len(bins)} fast_evals={fast_evals} [{time()-t0:.0f}s]"
                )
                break

            proposals.sort(key=lambda x: x[0])
            accepted = False
            probe_n = min(len(proposals), max(5, min(14, int(max(1.0, budget) // 18))))
            for (
                _score, fast_proxy, group, targets, axis, r, c, strength,
                fast_wl, fast_den, fast_cong, hot_drop, cong_drop,
            ) in proposals[:probe_n]:
                if time() - t0 > budget or real_evals >= real_eval_cap:
                    break
                cand_pos = best_pos.copy()
                cand_pos[group] = targets
                cand = best_placement.clone()
                cand[:num_all] = torch.from_numpy(cand_pos).to(dtype=cand.dtype, device=cand.device)
                _set_placement(plc, cand.detach(), benchmark)
                metrics = compute_proxy_cost(cand.detach(), benchmark, plc)
                real_evals += 1
                proxy = float(metrics["proxy_cost"])
                print(
                    f"  net_shear pass={passes} axis={axis} bin=({r},{c}) "
                    f"size={group.size:2d} str={strength:.2f} fast={fast_proxy:.4f} "
                    f"proxy={proxy:.4f} best={best_proxy:.4f} hot_drop={hot_drop:.3f} "
                    f"cong={metrics['congestion_cost']:.4f} [{time()-t0:.0f}s]"
                )
                if proxy < best_proxy - 1e-4:
                    best_proxy = proxy
                    best_placement = cand.clone()
                    best_pos = cand_pos.astype(np.float32).copy()
                    den_scale = float(metrics["density_cost"]) / (fast_den + 1e-8)
                    cong_scale = float(metrics["congestion_cost"]) / (fast_cong + 1e-8)
                    accepts += 1
                    accepted = True
                    break

            if not accepted:
                break

        print(
            f"Net shear done: passes={passes} accepts={accepts} "
            f"fast_evals={fast_evals} real_evals={real_evals} best proxy={best_proxy:.4f}"
        )
        return best_placement

    def _soft_hot_untwist_refine(
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
        budget=180,
    ):
        """
        Reorder soft macros inside hot congestion neighborhoods.

        The move keeps the neighborhood's existing soft-macro coordinates as
        "slots" and only permutes which soft macro occupies each slot. That
        makes it a topology untangler rather than another spreading force:
        density changes only because soft sizes differ, hard legality is
        untouched, and every accepted move is guarded by the real proxy.
        """
        num_soft = num_all - num_hard
        if num_soft < 4 or plc is None:
            return placement

        cur = placement.clone()
        pos = cur.detach().numpy().astype(np.float32).copy()
        _, macro_to_nets, _net_hpwl, _eval_delta, _total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, cur, num_all, canvas_norm
        )
        soft_candidates = np.array(
            [
                i for i in range(num_hard, num_all)
                if (not fixed[i].item()) and len(macro_to_nets[i]) > 0
            ],
            dtype=np.int32,
        )
        if soft_candidates.size < 4:
            return placement

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = ni_np.shape[0]
        max_degree = ni_np.shape[1]
        sizes_np = benchmark.macro_sizes[:num_all].numpy().copy()
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

        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            num_pin_nets = pin_owner_p.shape[0]
        else:
            pin_owner_p = pin_mask_p = pin_xoff_p = pin_yoff_p = pin_fx_p = pin_fy_p = nw_p = None
            num_pin_nets = 0

        def build_fast_state(pos_np):
            density_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_density_grid(
                density_grid, pos_np, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            v_route_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    pos_np, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, pos_np, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            h_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            v_macro_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
            _build_macro_route_grid(
                h_macro_grid, v_macro_grid, pos_np, sizes_np, num_hard, bl, br, bb, bt,
                n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
            )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            hpwl = _hpwl_batch(
                np.arange(num_nets, dtype=np.int32), ni_np, nm_np, pos_np
            )
            wl = float((hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            den = _density_cost_top5(density_grid)
            cong = cong_tracker.cost()
            pressure = np.maximum(
                cong_tracker.v_smooth + v_macro_grid,
                cong_tracker.h_smooth + h_macro_grid,
            )
            pressure = pressure + 0.25 * density_grid
            return wl, den, cong, pressure

        wl0, den0, cong0, pressure0 = build_fast_state(pos)
        _set_placement(plc, cur.detach(), benchmark)
        metrics0 = compute_proxy_cost(cur.detach(), benchmark, plc)
        best_proxy = float(metrics0["proxy_cost"])
        den_scale = float(metrics0["density_cost"]) / (den0 + 1e-8)
        cong_scale = float(metrics0["congestion_cost"]) / (cong0 + 1e-8)

        def fast_proxy_for(pos_np):
            wl, den, cong, pressure = build_fast_state(pos_np)
            proxy = wl + 0.5 * den * den_scale + 0.5 * cong * cong_scale
            return proxy, wl, den, cong, pressure

        current_fast = wl0 + 0.5 * den0 * den_scale + 0.5 * cong0 * cong_scale
        best_placement = cur.clone()
        best_pos = pos.copy()
        print(
            f"Soft untwist start: proxy={best_proxy:.4f} fast={current_fast:.4f} "
            f"wl={wl0:.4f} den={metrics0['density_cost']:.4f} "
            f"cong={metrics0['congestion_cost']:.4f} candidates={soft_candidates.size}"
        )

        def external_barycenters(group, pos_np):
            group_set = {int(i) for i in group}
            out = np.zeros((len(group), 2), dtype=np.float32)
            for gi, i_raw in enumerate(group):
                i = int(i_raw)
                sx = sy = sw = 0.0
                fallback_sx = fallback_sy = fallback_sw = 0.0
                for n in macro_to_nets[i]:
                    w = float(nw_np[n])
                    for d in range(max_degree):
                        if not nm_np[n, d]:
                            break
                        j = int(ni_np[n, d])
                        if j == i:
                            continue
                        fallback_sx += float(pos_np[j, 0]) * w
                        fallback_sy += float(pos_np[j, 1]) * w
                        fallback_sw += w
                        if j not in group_set:
                            sx += float(pos_np[j, 0]) * w
                            sy += float(pos_np[j, 1]) * w
                            sw += w
                if sw <= 1e-9:
                    sx, sy, sw = fallback_sx, fallback_sy, fallback_sw
                if sw <= 1e-9:
                    out[gi] = pos_np[i]
                else:
                    out[gi, 0] = sx / sw
                    out[gi, 1] = sy / sw
            return out

        def make_groups(pos_np, pressure):
            rows = np.clip((pos_np[soft_candidates, 1] / bin_h).astype(np.int64), 0, n_rows - 1)
            cols = np.clip((pos_np[soft_candidates, 0] / bin_w).astype(np.int64), 0, n_cols - 1)
            hot = pressure[rows, cols]
            degree = np.array([len(macro_to_nets[int(i)]) for i in soft_candidates], dtype=np.float32)
            score = hot + 0.006 * np.sqrt(np.maximum(degree, 1.0))
            seed_n = min(28, soft_candidates.size)
            seed_idx = np.argpartition(score, -seed_n)[-seed_n:]
            seed_idx = seed_idx[np.argsort(-score[seed_idx])]
            groups = []
            seen = set()
            min_group = min(6, soft_candidates.size)
            max_group = min(24, soft_candidates.size)
            for si in seed_idx:
                seed = int(soft_candidates[int(si)])
                dx = pos_np[soft_candidates, 0] - pos_np[seed, 0]
                dy = pos_np[soft_candidates, 1] - pos_np[seed, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                pool_n = min(soft_candidates.size, max(max_group * 4, min_group))
                pool_idx = np.argpartition(dist, pool_n - 1)[:pool_n]
                pool_scores = score[pool_idx] / (float(score[pool_idx].max()) + 1e-8)
                rank = dist[pool_idx] / (canvas_norm + 1e-8) - 0.015 * pool_scores
                k = min(max_group, max(min_group, int(10 + 0.18 * pool_n)))
                chosen_idx = pool_idx[np.argsort(rank)[:k]]
                group = np.unique(soft_candidates[chosen_idx]).astype(np.int32)
                if group.size < 4:
                    continue
                key = tuple(sorted(int(x) for x in group[: min(group.size, 16)]))
                if key in seen:
                    continue
                seen.add(key)
                groups.append(group)
                if len(groups) >= 14:
                    break
            return groups

        def candidate_targets(group, pos_np):
            slots = pos_np[group].copy()
            bary = external_barycenters(group, pos_np)
            center_slots = slots.mean(axis=0)
            center_bary = bary.mean(axis=0)
            candidates = []
            axes = (
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([1.0, -1.0], dtype=np.float32),
            )
            for axis in axes:
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                macro_order = np.argsort(bary @ axis)
                slot_order = np.argsort(slots @ axis)
                targets = slots.copy()
                for rank, macro_local in enumerate(macro_order):
                    targets[int(macro_local)] = slots[int(slot_order[rank])]
                candidates.append(("axis", targets))

            macro_angles = np.arctan2(bary[:, 1] - center_bary[1], bary[:, 0] - center_bary[0])
            slot_angles = np.arctan2(slots[:, 1] - center_slots[1], slots[:, 0] - center_slots[0])
            macro_order = np.argsort(macro_angles)
            slot_order = np.argsort(slot_angles)
            targets = slots.copy()
            for rank, macro_local in enumerate(macro_order):
                targets[int(macro_local)] = slots[int(slot_order[rank])]
            candidates.append(("angle", targets))

            # A mild radial variant catches spoke-wheel groups where angular
            # order is right but near/far ordering is inverted.
            macro_rad = np.linalg.norm(bary - center_bary, axis=1)
            slot_rad = np.linalg.norm(slots - center_slots, axis=1)
            macro_order = np.argsort(macro_rad)
            slot_order = np.argsort(slot_rad)
            targets = slots.copy()
            for rank, macro_local in enumerate(macro_order):
                targets[int(macro_local)] = slots[int(slot_order[rank])]
            candidates.append(("radial", targets))

            deduped = []
            seen = set()
            original_key = tuple(np.round(slots.reshape(-1), 4))
            for label, targets in candidates:
                key = tuple(np.round(targets.reshape(-1), 4))
                if key == original_key or key in seen:
                    continue
                if float(np.max(np.abs(targets - slots))) < 1e-5:
                    continue
                seen.add(key)
                deduped.append((label, targets))
            return deduped

        t0 = time()
        passes = 0
        fast_evals = 0
        real_evals = 1
        accepts = 0
        real_eval_cap = max(6, min(28, int(max(1.0, budget) // 7)))

        while time() - t0 < budget and passes < 3 and real_evals < real_eval_cap:
            passes += 1
            current_fast, _wl, current_den_est, current_cong_est, pressure = fast_proxy_for(best_pos)
            groups = make_groups(best_pos, pressure)
            if not groups:
                break

            proposals = []
            for group in groups:
                for label, targets in candidate_targets(group, best_pos):
                    if time() - t0 > budget:
                        break
                    cand_pos = best_pos.copy()
                    cand_pos[group] = targets
                    fast_proxy, wl, den, cong, _pressure = fast_proxy_for(cand_pos)
                    fast_evals += 1
                    fast_delta = fast_proxy - current_fast
                    den_drop = max(0.0, current_den_est - den)
                    cong_drop = max(0.0, current_cong_est - cong)
                    # The fast proxy is still a calibrated approximation; for
                    # pure slot reorders it can reject the exact moves we want.
                    # Probe mildly uphill candidates if they relieve hot-route
                    # pressure, but keep the real-eval cap small.
                    fast_allow = max(0.020, 0.015 * max(1.0, current_fast))
                    if fast_delta <= fast_allow or cong_drop > 0.004 or den_drop > 0.004:
                        selection_score = (
                            fast_proxy
                            - 0.15 * cong_drop * cong_scale
                            - 0.04 * den_drop * den_scale
                        )
                        proposals.append((
                            selection_score, fast_proxy, label, group.copy(),
                            targets.copy(), wl, den, cong,
                        ))
                if time() - t0 > budget:
                    break

            if not proposals:
                print(
                    f"  untwist pass={passes} no plausible reorder "
                    f"groups={len(groups)} fast_evals={fast_evals} [{time()-t0:.0f}s]"
                )
                break

            proposals.sort(key=lambda x: x[0])
            accepted_this_pass = False
            probe_n = min(len(proposals), max(4, min(12, int(max(1.0, budget) // 20))))
            for _score, fast_proxy, label, group, targets, wl, den, cong in proposals[:probe_n]:
                if time() - t0 > budget or real_evals >= real_eval_cap:
                    break
                cand = best_placement.clone()
                group_t = torch.as_tensor(group, dtype=torch.long, device=cand.device)
                cand[group_t] = torch.from_numpy(targets).to(dtype=cand.dtype, device=cand.device)
                _set_placement(plc, cand.detach(), benchmark)
                metrics = compute_proxy_cost(cand.detach(), benchmark, plc)
                real_evals += 1
                proxy = float(metrics["proxy_cost"])
                print(
                    f"  untwist pass={passes} {label:<6} size={group.size:2d} "
                    f"fast={fast_proxy:.4f} proxy={proxy:.4f} best={best_proxy:.4f} "
                    f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                    f"[{time()-t0:.0f}s]"
                )
                if proxy < best_proxy - 1e-4:
                    best_proxy = proxy
                    best_placement = cand.clone()
                    best_pos = best_placement.detach().numpy().astype(np.float32).copy()
                    den_scale = float(metrics["density_cost"]) / (den + 1e-8)
                    cong_scale = float(metrics["congestion_cost"]) / (cong + 1e-8)
                    accepts += 1
                    accepted_this_pass = True
                    break

            if not accepted_this_pass:
                break

        print(
            f"Soft untwist done: passes={passes} accepts={accepts} "
            f"fast_evals={fast_evals} real_evals={real_evals} best proxy={best_proxy:.4f}"
        )
        return best_placement

    def _sa_soft_lns_repack(
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
        budget=300,
        checkpoint_every=16,
    ):
        """Coordinated soft-macro LNS using incremental WL+density+HV proxy."""
        num_soft = num_all - num_hard
        if num_soft < 4:
            return placement

        sa_placement = placement.clone()
        sa_pos, macro_to_nets, net_hpwl, _eval_delta, total_wl = self._build_incremental_wl(
            net_indices, net_mask, net_weights, sa_placement, num_all, canvas_norm
        )

        soft_candidates = np.array(
            [
                i for i in range(num_hard, num_all)
                if (not fixed[i].item()) and len(macro_to_nets[i]) > 0
            ],
            dtype=np.int32,
        )
        if soft_candidates.size < 4:
            return placement

        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        num_nets = ni_np.shape[0]
        nw_np = net_weights.numpy().copy()
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
        _build_density_grid(
            density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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

        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )

        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()
        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        den_scale = real_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = real_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = (
            total_wl
            + 0.5 * current_den * den_scale
            + 0.5 * current_cong * cong_scale
        )
        best_placement = sa_placement.clone()
        print(
            f"Soft LNS start: proxy={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} num_soft={num_soft} candidates={soft_candidates.size} "
            f"(HV cong, smooth={smooth_range})"
        )

        def rebuild_fast_state(from_tensor):
            nonlocal sa_placement, sa_pos, net_hpwl, total_wl
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            sa_placement = from_tensor.clone()
            sa_pos[:] = sa_placement.detach().numpy()
            net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
            total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
            density_grid[:] = 0
            _build_density_grid(
                density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_den = _density_cost_top5(density_grid)
            current_cong = cong_tracker.cost()
            current_proxy = (
                total_wl
                + 0.5 * current_den * den_scale
                + 0.5 * current_cong * cong_scale
            )

        def pressure_for(indices):
            rows = np.clip((sa_pos[indices, 1] / bin_h).astype(np.int64), 0, n_rows - 1)
            cols = np.clip((sa_pos[indices, 0] / bin_w).astype(np.int64), 0, n_cols - 1)
            cong_pressure = 0.5 * (cong_tracker.h_smooth[rows, cols] + cong_tracker.v_smooth[rows, cols])
            return density_grid[rows, cols] + cong_pressure

        def macro_barycenter(i):
            sx = sy = sw = 0.0
            for n in macro_to_nets[i]:
                w = float(nw_np[n])
                for d in range(ni_np.shape[1]):
                    if not nm_np[n, d]:
                        break
                    j = int(ni_np[n, d])
                    if j == i:
                        continue
                    sx += float(sa_pos[j, 0]) * w
                    sy += float(sa_pos[j, 1]) * w
                    sw += w
            if sw <= 0.0:
                return sa_pos[i].copy()
            return np.array([sx / sw, sy / sw], dtype=np.float32)

        def choose_group():
            sample_n = min(96, soft_candidates.size)
            sample = np.random.choice(soft_candidates, size=sample_n, replace=False)
            p = pressure_for(sample)
            center = int(sample[int(np.argmax(p + 0.01 * np.random.random(size=p.shape)))])
            k_min = min(8, soft_candidates.size)
            k_max = min(28, soft_candidates.size)
            if k_max <= k_min:
                k = k_min
            else:
                k = random.randint(k_min, k_max)
            dx = sa_pos[soft_candidates, 0] - sa_pos[center, 0]
            dy = sa_pos[soft_candidates, 1] - sa_pos[center, 1]
            dist2 = dx * dx + dy * dy
            pool_n = min(soft_candidates.size, max(k * 4, k + 1))
            pool_idx = np.argpartition(dist2, pool_n - 1)[:pool_n]
            pool = soft_candidates[pool_idx]
            group = np.random.choice(pool, size=k, replace=False).astype(np.int32)
            if center not in group:
                group[0] = center
            return np.unique(group).astype(np.int32)

        def propose_group(group, move_scale):
            old_xy = sa_pos[group].copy()
            new_xy = old_xy.copy()
            center_xy = old_xy.mean(axis=0)
            mode = random.random()

            if mode < 0.35 and group.size >= 3:
                slots = old_xy.copy()
                bvec = np.array([macro_barycenter(int(i)) for i in group], dtype=np.float32)
                macro_angles = np.arctan2(bvec[:, 1] - center_xy[1], bvec[:, 0] - center_xy[0])
                slot_angles = np.arctan2(slots[:, 1] - center_xy[1], slots[:, 0] - center_xy[0])
                macro_order = np.argsort(macro_angles)
                slot_order = np.argsort(slot_angles)
                new_xy[macro_order] = slots[slot_order]
            elif mode < 0.55 and group.size >= 2:
                new_xy = old_xy[np.random.permutation(group.size)].copy()
            elif mode < 0.75:
                vec = old_xy - center_xy
                norm = np.linalg.norm(vec, axis=1)
                tiny = norm < 1e-6
                if tiny.any():
                    vec[tiny] = np.random.uniform(-1.0, 1.0, size=(int(tiny.sum()), 2))
                scale = random.uniform(1.15, 1.75)
                jitter = np.random.uniform(-0.35 * move_scale, 0.35 * move_scale, size=old_xy.shape)
                new_xy = center_xy + scale * vec + jitter
            elif mode < 0.90:
                for idx, i in enumerate(group):
                    bary = macro_barycenter(int(i))
                    step = random.uniform(0.25, 0.65)
                    jitter = np.random.uniform(-0.25 * move_scale, 0.25 * move_scale, size=2)
                    new_xy[idx] = old_xy[idx] + step * (bary - old_xy[idx]) + jitter
            else:
                pressure_grid = density_grid + 0.5 * (cong_tracker.h_smooth + cong_tracker.v_smooth)
                x0 = max(0, int((old_xy[:, 0].min() - 2.0 * move_scale) / bin_w))
                x1 = min(n_cols - 1, int((old_xy[:, 0].max() + 2.0 * move_scale) / bin_w))
                y0 = max(0, int((old_xy[:, 1].min() - 2.0 * move_scale) / bin_h))
                y1 = min(n_rows - 1, int((old_xy[:, 1].max() + 2.0 * move_scale) / bin_h))
                sub = pressure_grid[y0 : y1 + 1, x0 : x1 + 1]
                if sub.size > 0:
                    low_n = min(max(4, group.size * 3), sub.size)
                    low_flat = np.argpartition(sub.reshape(-1), low_n - 1)[:low_n]
                    for idx in range(group.size):
                        pick = int(random.choice(low_flat))
                        rr = y0 + pick // sub.shape[1]
                        cc = x0 + pick % sub.shape[1]
                        new_xy[idx, 0] = (cc + random.uniform(0.2, 0.8)) * bin_w
                        new_xy[idx, 1] = (rr + random.uniform(0.2, 0.8)) * bin_h

            for idx, i in enumerate(group):
                ii = int(i)
                new_xy[idx, 0] = np.clip(new_xy[idx, 0], hw_np[ii], benchmark.canvas_width - hw_np[ii])
                new_xy[idx, 1] = np.clip(new_xy[idx, 1], hh_np[ii], benchmark.canvas_height - hh_np[ii])
            return old_xy, new_xy

        def apply_one(i, old_x, old_y, new_x, new_y):
            sa_pos[i, 0] = new_x
            sa_pos[i, 1] = new_y
            _update_density_incr(
                density_grid, old_x, old_y, new_x, new_y,
                hw_np[i], hh_np[i], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[i]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        sa_pos, pin_aff, i, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                aff_i = macro_to_nets[i]
                if len(aff_i) > 0:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        ni_np, nm_np, nw_np, sa_pos, aff_i, i, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )

        accepts = trials = stalls = 0
        last_accept_trial = 0
        last_checkpoint_accepts = 0
        t0 = time()
        meaningful_gain = 1e-3
        progress_anchor = best_proxy

        while time() - t0 < budget:
            elapsed_frac = min(1.0, (time() - t0) / max(budget, 1e-6))
            move_scale = canvas_norm * (0.045 * (1.0 - elapsed_frac) + 0.010 * elapsed_frac)
            group = choose_group()
            if group.size < 2:
                continue

            aff_parts = [macro_to_nets[int(i)] for i in group if len(macro_to_nets[int(i)]) > 0]
            if not aff_parts:
                continue
            aff = np.unique(np.concatenate(aff_parts)).astype(np.int32)
            if aff.size == 0:
                continue

            trials += 1
            old_xy, new_xy = propose_group(group, move_scale)
            if np.max(np.abs(new_xy - old_xy)) < 1e-5:
                continue

            old_hpwl = net_hpwl[aff].copy()
            for idx, i in enumerate(group):
                apply_one(
                    int(i),
                    float(old_xy[idx, 0]),
                    float(old_xy[idx, 1]),
                    float(new_xy[idx, 0]),
                    float(new_xy[idx, 1]),
                )

            new_hpwl = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
            wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)
            new_den = _density_cost_top5(density_grid)
            new_cong = cong_tracker.cost()
            new_proxy = (
                total_wl + wl_delta
                + 0.5 * new_den * den_scale
                + 0.5 * new_cong * cong_scale
            )

            if new_proxy < current_proxy - 1e-6:
                for idx, i in enumerate(group):
                    sa_placement[int(i), 0] = float(new_xy[idx, 0])
                    sa_placement[int(i), 1] = float(new_xy[idx, 1])
                net_hpwl[aff] = new_hpwl
                total_wl += wl_delta
                current_den = new_den
                current_cong = new_cong
                current_proxy = new_proxy
                accepts += 1
                last_accept_trial = trials

                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                    proxy = metrics["proxy_cost"]
                    print(
                        f"  lns trials={trials} accepts={accepts} size={group.size} "
                        f"fast={current_proxy:.4f} proxy={proxy:.4f} "
                        f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                        f"[{time()-t0:.0f}s]"
                    )
                    last_checkpoint_accepts = accepts
                    den_scale = metrics["density_cost"] / (current_den + 1e-8)
                    cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                    current_proxy = (
                        total_wl
                        + 0.5 * current_den * den_scale
                        + 0.5 * current_cong * cong_scale
                    )
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = sa_placement.clone()
                        if progress_anchor - best_proxy >= meaningful_gain:
                            progress_anchor = best_proxy
                            stalls = 0
                        else:
                            stalls += 1
                    else:
                        rebuild_fast_state(best_placement)
                        _set_placement(plc, sa_placement.detach(), benchmark)
                        best_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                        den_scale = best_metrics["density_cost"] / (current_den + 1e-8)
                        cong_scale = best_metrics["congestion_cost"] / (current_cong + 1e-8)
                        current_proxy = (
                            total_wl
                            + 0.5 * current_den * den_scale
                            + 0.5 * current_cong * cong_scale
                        )
                        stalls += 1
                    if stalls >= 6:
                        print("Soft LNS stalled")
                        break
            else:
                for idx in range(group.size - 1, -1, -1):
                    i = int(group[idx])
                    apply_one(
                        i,
                        float(new_xy[idx, 0]),
                        float(new_xy[idx, 1]),
                        float(old_xy[idx, 0]),
                        float(old_xy[idx, 1]),
                    )

            if trials % 2000 == 0:
                print(
                    f"  lns trials={trials} accepts={accepts} fast={current_proxy:.4f} "
                    f"best={best_proxy:.4f} move={move_scale/canvas_norm:.4f} "
                    f"[{time()-t0:.0f}s]",
                    end="\r",
                )
            if trials - last_accept_trial > 100_000:
                print("Soft LNS: no accepts in 100k trials, stopping")
                break

        if accepts > last_checkpoint_accepts:
            _set_placement(plc, sa_placement.detach(), benchmark)
            metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
            proxy = metrics["proxy_cost"]
            print(
                f"  lns final_chk accepts={accepts} trials={trials} "
                f"proxy={proxy:.4f} best={best_proxy:.4f} [{time()-t0:.0f}s]"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_placement = sa_placement.clone()

        print(
            f"Soft LNS done: {trials} trials, {accepts} accepts, "
            f"best proxy={best_proxy:.4f}"
        )
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
        """Greedy soft macro displacement with incremental WL+density+HV congestion."""
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
        _build_density_grid(
            density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
            bin_area, n_rows, n_cols, bin_w, bin_h,
        )

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
        _pin_tensors = build_pin_route_tensors(benchmark)
        _use_pin_routing = _pin_tensors is not None
        if _use_pin_routing:
            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p = _pin_tensors
            macro_to_pin_nets = build_macro_to_pin_nets(benchmark, num_all)
            num_pin_nets = pin_owner_p.shape[0]
            _build_pin_hv_route_grid(
                h_route_grid, v_route_grid,
                pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        else:
            _build_hv_route_grid(
                h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                bin_w, bin_h, n_rows, n_cols, hcap, vcap,
            )
        # Soft macros do not block macro routing; hard macro blockage is fixed
        # throughout this stage.
        _build_macro_route_grid(
            h_macro_grid, v_macro_grid, sa_pos, sizes_np, num_hard, bl, br, bb, bt,
            n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc,
        )
        cong_tracker = SmoothHVCostTracker(
            h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
        )
        current_den = _density_cost_top5(density_grid)
        current_cong = cong_tracker.cost()

        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        den_scale = real_metrics["density_cost"] / (current_den + 1e-8)
        cong_scale = real_metrics["congestion_cost"] / (current_cong + 1e-8)
        current_proxy = (
            total_wl
            + 0.5 * current_den * den_scale
            + 0.5 * current_cong * cong_scale
        )
        best_placement = sa_placement.clone()
        print(
            f"Soft displace start: proxy={best_proxy:.4f} fast={current_proxy:.4f} "
            f"wl={total_wl:.4f} num_soft={num_soft} "
            f"(HV cong, smooth={smooth_range})"
        )
        timing_enabled = os.environ.get("HAP_STAGE_TIMERS", "").strip().lower() not in (
            "", "0", "false", "no", "off"
        )
        sd_timers = {
            "proposal": 0.0,
            "hpwl": 0.0,
            "density": 0.0,
            "route": 0.0,
            "rollback": 0.0,
            "rebuild": 0.0,
            "real": 0.0,
        }

        accepts = total = stalls = 0
        last_accept_step = 0
        last_checkpoint_accepts = 0
        last_rebuild_accepts = 0
        t0 = time()
        disp_start = canvas_norm * 0.01
        disp_end = canvas_norm * 0.002
        check_interval = 500_000
        last_check_accepts = 0
        min_rate = 5e-5
        long_soft_displace = float(budget) >= 900.0
        gate_window = int(os.environ.get(
            "HAP_SOFT_DISPLACE_GATE_WINDOW",
            "3" if long_soft_displace else "4",
        ))
        gate_patience = int(os.environ.get(
            "HAP_SOFT_DISPLACE_GATE_PATIENCE",
            "1" if long_soft_displace else "2",
        ))
        gate_epsilon = float(os.environ.get(
            "HAP_SOFT_DISPLACE_GATE_EPS",
            "0.0020" if long_soft_displace else "0.0010",
        ))
        gate_min_time = float(os.environ.get("HAP_SOFT_DISPLACE_GATE_MIN_TIME", "300"))
        progress_gate = WindowProgressGate(
            window=gate_window,
            patience_windows=gate_patience,
            epsilon=gate_epsilon,
            min_time=min(gate_min_time, max(0.0, float(budget))),
            initial_best=best_proxy,
        )
        print(
            f"Soft displace gate: window={gate_window} patience={gate_patience} "
            f"epsilon={gate_epsilon:.4f} min_time={min(gate_min_time, max(0.0, float(budget))):.0f}s"
        )

        def rebuild_fast_state(from_tensor):
            nonlocal sa_placement, sa_pos, net_hpwl, total_wl
            nonlocal current_den, current_cong, current_proxy, cong_tracker
            if timing_enabled:
                _timer_t = time()
            sa_placement = from_tensor.clone()
            sa_pos[:] = sa_placement.detach().numpy()
            net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
            total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
            density_grid[:] = 0
            _build_density_grid(
                density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt,
                bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            h_route_grid[:] = 0
            v_route_grid[:] = 0
            if _use_pin_routing:
                _build_pin_hv_route_grid(
                    h_route_grid, v_route_grid,
                    pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                    sa_pos, num_pin_nets, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            else:
                _build_hv_route_grid(
                    h_route_grid, v_route_grid, ni_np, nm_np, nw_np, sa_pos, num_nets,
                    bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            cong_tracker = SmoothHVCostTracker(
                h_route_grid, v_route_grid, h_macro_grid, v_macro_grid, smooth_range
            )
            current_den = _density_cost_top5(density_grid)
            current_cong = cong_tracker.cost()
            current_proxy = (
                total_wl
                + 0.5 * current_den * den_scale
                + 0.5 * current_cong * cong_scale
            )
            if timing_enabled:
                sd_timers["rebuild"] += time() - _timer_t

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break

            if (
                accepts > 0
                and accepts % 500 == 0
                and accepts != last_rebuild_accepts
                and accepts % checkpoint_every != 0
            ):
                rebuild_fast_state(sa_placement)
                last_rebuild_accepts = accepts

            elapsed_frac = min(1.0, (time() - t0) / max(budget, 1e-6))
            max_displacement = disp_start * (1 - elapsed_frac) + disp_end * elapsed_frac
            if total % 100000 == 0 and total > 0:
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} fast={current_proxy:.4f} "
                    f"disp={max_displacement/canvas_norm:.4f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            if total % check_interval == 0 and total > 1_000_000:
                rate = (accepts - last_check_accepts) / check_interval
                if rate < min_rate:
                    print(
                        f"\nSoft displace: accept rate {rate:.2e} < {min_rate:.2e}, stopping early"
                    )
                    break
                last_check_accepts = accepts

            if timing_enabled:
                _timer_t = time()
            i = num_hard + random.randint(0, num_soft - 1)
            if fixed[i].item() or len(macro_to_nets[i]) == 0:
                if timing_enabled:
                    sd_timers["proposal"] += time() - _timer_t
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
            if timing_enabled:
                sd_timers["proposal"] += time() - _timer_t

            aff = macro_to_nets[i]
            if timing_enabled:
                _timer_t = time()
            old_hpwl = net_hpwl[aff].copy()
            sa_pos[i, 0] = new_x
            sa_pos[i, 1] = new_y
            new_hpwl = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
            wl_delta = float((new_hpwl - old_hpwl).sum()) / (num_nets * canvas_norm)
            if timing_enabled:
                sd_timers["hpwl"] += time() - _timer_t

            if timing_enabled:
                _timer_t = time()
            _update_density_incr(
                density_grid, old_x, old_y, new_x, new_y,
                hw_np[i], hh_np[i], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
            )
            new_den = _density_cost_top5(density_grid)
            if timing_enabled:
                sd_timers["density"] += time() - _timer_t

            if timing_enabled:
                _timer_t = time()
            if _use_pin_routing:
                pin_aff = macro_to_pin_nets[i]
                if len(pin_aff) > 0:
                    update_pin_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker,
                        pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                        sa_pos, pin_aff, i, old_x, old_y,
                        bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
            else:
                update_hv_route_incr_single_smooth(
                    h_route_grid, v_route_grid, cong_tracker, ni_np, nm_np, nw_np, sa_pos, aff,
                    i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                )
            new_cong = cong_tracker.cost()
            if timing_enabled:
                sd_timers["route"] += time() - _timer_t
            new_proxy = (
                total_wl + wl_delta
                + 0.5 * new_den * den_scale
                + 0.5 * new_cong * cong_scale
            )

            if new_proxy <= current_proxy:
                sa_placement[i, 0] = new_x
                sa_placement[i, 1] = new_y
                net_hpwl[aff] = new_hpwl
                total_wl += wl_delta
                current_den = new_den
                current_cong = new_cong
                current_proxy = new_proxy
                accepts += 1
                last_accept_step = total

                if accepts % checkpoint_every == 0:
                    if timing_enabled:
                        _timer_t = time()
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                    if timing_enabled:
                        sd_timers["real"] += time() - _timer_t
                    proxy = metrics["proxy_cost"]
                    last_checkpoint_accepts = accepts
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                        f"fast={current_proxy:.4f} proxy={proxy:.4f} "
                        f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                        f"[{time()-t0:.0f}s]"
                    )
                    den_scale = metrics["density_cost"] / (current_den + 1e-8)
                    cong_scale = metrics["congestion_cost"] / (current_cong + 1e-8)
                    current_proxy = (
                        total_wl
                        + 0.5 * current_den * den_scale
                        + 0.5 * current_cong * cong_scale
                    )
                    _, gate_stop = progress_gate.update(proxy, time() - t0)
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = sa_placement.clone()
                        stalls = 0
                    else:
                        rebuild_fast_state(best_placement)
                        if timing_enabled:
                            _timer_t = time()
                        _set_placement(plc, sa_placement.detach(), benchmark)
                        best_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
                        if timing_enabled:
                            sd_timers["real"] += time() - _timer_t
                        den_scale = best_metrics["density_cost"] / (current_den + 1e-8)
                        cong_scale = best_metrics["congestion_cost"] / (current_cong + 1e-8)
                        current_proxy = (
                            total_wl
                            + 0.5 * current_den * den_scale
                            + 0.5 * current_cong * cong_scale
                        )
                        stalls += 1
                    if gate_stop:
                        print(
                            f"Soft displace stalled "
                            f"(window gate, best={progress_gate.best:.4f}, "
                            f"patience={progress_gate.patience}/{progress_gate.max_patience})"
                        )
                        break
            else:
                if timing_enabled:
                    _timer_t = time()
                sa_pos[i, 0] = old_x
                sa_pos[i, 1] = old_y
                _update_density_incr(
                    density_grid, new_x, new_y, old_x, old_y,
                    hw_np[i], hh_np[i], bl, br, bb, bt, bin_area, n_rows, n_cols, bin_w, bin_h,
                )
                if _use_pin_routing:
                    pin_aff = macro_to_pin_nets[i]
                    if len(pin_aff) > 0:
                        update_pin_hv_route_incr_single_smooth(
                            h_route_grid, v_route_grid, cong_tracker,
                            pin_owner_p, pin_mask_p, pin_xoff_p, pin_yoff_p, pin_fx_p, pin_fy_p, nw_p,
                            sa_pos, pin_aff, i, new_x, new_y,
                            bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                        )
                else:
                    update_hv_route_incr_single_smooth(
                        h_route_grid, v_route_grid, cong_tracker, ni_np, nm_np, nw_np, sa_pos, aff,
                        i, new_x, new_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
                    )
                if timing_enabled:
                    sd_timers["rollback"] += time() - _timer_t

            if total - last_accept_step > 2_000_000:
                print("Soft displace: no accepts in 2M attempts, stopping")
                break

        if accepts > last_checkpoint_accepts:
            if timing_enabled:
                _timer_t = time()
            _set_placement(plc, sa_placement.detach(), benchmark)
            metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
            if timing_enabled:
                sd_timers["real"] += time() - _timer_t
            proxy = metrics["proxy_cost"]
            print(
                f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                f"fast={current_proxy:.4f} proxy={proxy:.4f} "
                f"den={metrics['density_cost']:.4f} cong={metrics['congestion_cost']:.4f} "
                f"final_chk [{time()-t0:.0f}s]"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_placement = sa_placement.clone()

        print(
            f"Soft displace done: {total} attempts, {accepts} accepts, "
            f"best proxy={best_proxy:.4f}"
        )
        if timing_enabled:
            timed_total = max(1e-9, sum(sd_timers.values()))
            print(
                "Soft displace timing: "
                + " ".join(
                    f"{k}={v:.2f}s({100.0*v/timed_total:.0f}%)"
                    for k, v in sd_timers.items()
                    if v > 0.0
                )
                + f" timed={timed_total:.2f}s wall={time()-t0:.2f}s"
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
