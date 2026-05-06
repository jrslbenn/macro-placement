"""
HAP v2 — TILOS-matched fast proxy.

Key improvements over hybrid_analytical_placer.py:
- Density: top-10%, ×0.5 (matches plc_client_os.get_density_cost())
- Congestion: L-shape net routing + macro blockage + box-filter smoothing
  normalized by routes_per_micron, top-5% of combined H+V
  (matches plc_client_os.get_congestion_cost())
- SA displace proxy uses TILOS values directly — no den_scale/cong_scale needed
"""

import math
import random
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import numpy as np
from numba import njit
import torch
from scipy.fft import dctn, idctn

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost


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

    sep_x = (sizes[:, 0].unsqueeze(1) + sizes[:, 0].unsqueeze(0)) / 2 + gap
    sep_y = (sizes[:, 1].unsqueeze(1) + sizes[:, 1].unsqueeze(0)) / 2 + gap
    tri_mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)

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

        push_amount_x = torch.where(overlap & push_x_axis, push_amount_x, torch.zeros_like(push_amount_x))
        push_amount_y = torch.where(overlap & ~push_x_axis, push_amount_y, torch.zeros_like(push_amount_y))
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


# ── Density ──────────────────────────────────────────────────────────────────

@njit
def _density_cost_tilos(grid):
    """Top-10% mean × 0.5. Matches plc_client_os.get_density_cost()."""
    flat = grid.flatten()
    n = len(flat)
    k = max(1, int(n * 0.1))
    idx = np.argpartition(flat, -k)[-k:]
    return flat[idx].mean() * np.float32(0.5)


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
def _update_density_incr(
    grid, old_cx, old_cy, new_cx, new_cy, hw, hh, bl, br, bb, bt,
    bin_area, n_rows, n_cols, bin_w, bin_h,
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


# ── TILOS congestion ──────────────────────────────────────────────────────────

@njit
def _build_tilos_routing_raw(
    H_raw, V_raw, H_mac, V_mac,
    ni_np, nm_np, pos,
    num_nets, num_hard, macro_sizes,
    bin_w, bin_h, n_rows, n_cols,
    hrouting_alloc, vrouting_alloc,
    bl, br, bb, bt,
):
    """
    Build raw (unnormalized, unsmoothed) TILOS H/V routing demand grids.

    Net routing: 2-pin L-shape; 3-pin L/T-routing (matches TILOS __three_pin_net_routing);
    4+-pin star decomposition. Macro blockage: x_overlap*vrouting_alloc for V, y_overlap*hrouting_alloc for H.
    """
    for r in range(n_rows):
        for c in range(n_cols):
            H_raw[r, c] = np.float32(0.0)
            V_raw[r, c] = np.float32(0.0)
            H_mac[r, c] = np.float32(0.0)
            V_mac[r, c] = np.float32(0.0)

    max_deg = ni_np.shape[1]
    gcells_r = np.empty(max_deg, dtype=np.int32)
    gcells_c = np.empty(max_deg, dtype=np.int32)

    # Net routing
    for n in range(num_nets):
        count = np.int32(0)
        for d in range(max_deg):
            if not nm_np[n, d]:
                break
            idx = ni_np[n, d]
            r = int(pos[idx, 1] / bin_h)
            c = int(pos[idx, 0] / bin_w)
            if r < 0:
                r = 0
            if r >= n_rows:
                r = n_rows - 1
            if c < 0:
                c = 0
            if c >= n_cols:
                c = n_cols - 1
            gcells_r[count] = r
            gcells_c[count] = c
            count += 1

        if count < 2:
            continue

        if count == 2:
            # 2-pin L-shape: H at pin-0 row, V at pin-1 col
            _r0 = gcells_r[0]; _c0 = gcells_c[0]
            _r1 = gcells_r[1]; _c1 = gcells_c[1]
            for _c in range(min(_c0, _c1), max(_c0, _c1)):
                H_raw[_r0, _c] += np.float32(1.0)
            for _r in range(min(_r0, _r1), max(_r0, _r1)):
                V_raw[_r, _c1] += np.float32(1.0)

        elif count == 3:
            # 3-pin routing matching TILOS __three_pin_net_routing.
            # Sort pins by (col, row).
            _yr0 = gcells_r[0]; _xc0 = gcells_c[0]
            _yr1 = gcells_r[1]; _xc1 = gcells_c[1]
            _yr2 = gcells_r[2]; _xc2 = gcells_c[2]
            # Bubble sort 3 elements by (col, row)
            if (_xc0 > _xc1) or (_xc0 == _xc1 and _yr0 > _yr1):
                _t = _yr0; _yr0 = _yr1; _yr1 = _t
                _t = _xc0; _xc0 = _xc1; _xc1 = _t
            if (_xc1 > _xc2) or (_xc1 == _xc2 and _yr1 > _yr2):
                _t = _yr1; _yr1 = _yr2; _yr2 = _t
                _t = _xc1; _xc1 = _xc2; _xc2 = _t
            if (_xc0 > _xc1) or (_xc0 == _xc1 and _yr0 > _yr1):
                _t = _yr0; _yr0 = _yr1; _yr1 = _t
                _t = _xc0; _xc0 = _xc1; _xc1 = _t
            _y1 = _yr0; _x1 = _xc0
            _y2 = _yr1; _x2 = _xc1
            _y3 = _yr2; _x3 = _xc2

            if _x1 < _x2 and _x2 < _x3 and min(_y1, _y3) < _y2 and _y2 < max(_y1, _y3):
                # L-routing
                for _c in range(_x1, _x2):
                    H_raw[_y1, _c] += np.float32(1.0)
                for _c in range(_x2, _x3):
                    H_raw[_y2, _c] += np.float32(1.0)
                for _r in range(min(_y1, _y2), max(_y1, _y2)):
                    V_raw[_r, _x2] += np.float32(1.0)
                for _r in range(min(_y2, _y3), max(_y2, _y3)):
                    V_raw[_r, _x3] += np.float32(1.0)
            elif _x2 == _x3 and _x1 < _x2 and _y1 < min(_y2, _y3):
                # Left pin below both right pins (same col)
                for _c in range(_x1, _x2):
                    H_raw[_y1, _c] += np.float32(1.0)
                for _r in range(_y1, max(_y2, _y3)):
                    V_raw[_r, _x2] += np.float32(1.0)
            elif _y2 == _y3:
                # Middle and right pins on same row
                for _c in range(_x1, _x2):
                    H_raw[_y1, _c] += np.float32(1.0)
                for _c in range(_x2, _x3):
                    H_raw[_y2, _c] += np.float32(1.0)
                for _r in range(min(_y1, _y2), max(_y1, _y2)):
                    V_raw[_r, _x2] += np.float32(1.0)
            else:
                # T-routing: re-sort by (row, col)
                _tr0 = _y1; _tc0 = _x1
                _tr1 = _y2; _tc1 = _x2
                _tr2 = _y3; _tc2 = _x3
                if (_tr0 > _tr1) or (_tr0 == _tr1 and _tc0 > _tc1):
                    _t = _tr0; _tr0 = _tr1; _tr1 = _t
                    _t = _tc0; _tc0 = _tc1; _tc1 = _t
                if (_tr1 > _tr2) or (_tr1 == _tr2 and _tc1 > _tc2):
                    _t = _tr1; _tr1 = _tr2; _tr2 = _t
                    _t = _tc1; _tc1 = _tc2; _tc2 = _t
                if (_tr0 > _tr1) or (_tr0 == _tr1 and _tc0 > _tc1):
                    _t = _tr0; _tr0 = _tr1; _tr1 = _t
                    _t = _tc0; _tc0 = _tc1; _tc1 = _t
                _ty1 = _tr0; _tx1 = _tc0
                _ty2 = _tr1; _tx2 = _tc1
                _ty3 = _tr2; _tx3 = _tc2
                _txmin = min(_tx1, min(_tx2, _tx3))
                _txmax = max(_tx1, max(_tx2, _tx3))
                for _c in range(_txmin, _txmax):
                    H_raw[_ty2, _c] += np.float32(1.0)
                for _r in range(min(_ty1, _ty2), max(_ty1, _ty2)):
                    V_raw[_r, _tx1] += np.float32(1.0)
                for _r in range(min(_ty2, _ty3), max(_ty2, _ty3)):
                    V_raw[_r, _tx3] += np.float32(1.0)

        else:
            # count >= 4: star decomposition from pin 0
            _r_src = gcells_r[0]
            _c_src = gcells_c[0]
            for k in range(1, count):
                _r_snk = gcells_r[k]
                _c_snk = gcells_c[k]
                for _c in range(min(_c_src, _c_snk), max(_c_src, _c_snk)):
                    H_raw[_r_src, _c] += np.float32(1.0)
                for _r in range(min(_r_src, _r_snk), max(_r_src, _r_snk)):
                    V_raw[_r, _c_snk] += np.float32(1.0)

    # Macro blockage
    for m in range(num_hard):
        cx = pos[m, 0]
        cy = pos[m, 1]
        hw = macro_sizes[m, 0] * np.float32(0.5)
        hh = macro_sizes[m, 1] * np.float32(0.5)
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
                if ox > np.float32(0.0) and oy > np.float32(0.0):
                    V_mac[r, c] += ox * vrouting_alloc
                    H_mac[r, c] += oy * hrouting_alloc


@njit
def _compute_tilos_cong(
    H_raw, V_raw, H_mac, V_mac,
    n_rows, n_cols,
    grid_h_routes, grid_v_routes,
    smooth_range,
):
    """
    Normalize, smooth, add macro blockage, return top-5% congestion cost.
    Matches plc_client_os.get_congestion_cost() = abu(V+H combined, 0.05).

    Smoothing: V routing spread horizontally (across cols), H spread vertically (across rows).
    """
    inv_h = np.float32(1.0) / grid_h_routes if grid_h_routes > np.float32(0.0) else np.float32(0.0)
    inv_v = np.float32(1.0) / grid_v_routes if grid_v_routes > np.float32(0.0) else np.float32(0.0)

    H_norm = np.zeros((n_rows, n_cols), dtype=np.float32)
    V_norm = np.zeros((n_rows, n_cols), dtype=np.float32)
    H_mac_norm = np.zeros((n_rows, n_cols), dtype=np.float32)
    V_mac_norm = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            H_norm[r, c] = H_raw[r, c] * inv_h
            V_norm[r, c] = V_raw[r, c] * inv_v
            H_mac_norm[r, c] = H_mac[r, c] * inv_h
            V_mac_norm[r, c] = V_mac[r, c] * inv_v

    # Smooth V routing horizontally (spread to neighboring cols)
    V_smooth = np.zeros((n_rows, n_cols), dtype=np.float32)
    for row in range(n_rows):
        for col in range(n_cols):
            lp = col - smooth_range
            if lp < 0:
                lp = 0
            rp = col + smooth_range
            if rp >= n_cols:
                rp = n_cols - 1
            gcell_cnt = rp - lp + 1
            val = V_norm[row, col] / gcell_cnt
            for ptr in range(lp, rp + 1):
                V_smooth[row, ptr] += val

    # Smooth H routing vertically (spread to neighboring rows)
    H_smooth = np.zeros((n_rows, n_cols), dtype=np.float32)
    for row in range(n_rows):
        for col in range(n_cols):
            lp = row - smooth_range
            if lp < 0:
                lp = 0
            up = row + smooth_range
            if up >= n_rows:
                up = n_rows - 1
            gcell_cnt = up - lp + 1
            val = H_norm[row, col] / gcell_cnt
            for ptr in range(lp, up + 1):
                H_smooth[ptr, col] += val

    # Combined: all V cells + all H cells (length = 2 * n_rows * n_cols)
    n_cells = n_rows * n_cols
    combined = np.zeros(2 * n_cells, dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            combined[idx] = V_smooth[r, c] + V_mac_norm[r, c]
            combined[n_cells + idx] = H_smooth[r, c] + H_mac_norm[r, c]

    total = 2 * n_cells
    k = max(1, int(total * 0.05))
    part_idx = np.argpartition(combined, -k)[-k:]
    return combined[part_idx].mean()


# ── Wirelength helpers ────────────────────────────────────────────────────────

@njit
def _hpwl_batch(net_idxs, ni_np, nm_np, pos):
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


# Orientation helpers (0=N, 1=FN, 2=S, 3=FS — all preserve bounding box)
_ORIENT_NAMES = ["N", "FN", "S", "FS"]
# Cayley table: _ORIENT_CAYLEY[from_idx][to_idx] = delta to pass to update_macro_orientation
_ORIENT_CAYLEY = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]

def _centroid_for_orient(base, orient):
    """Transform N-orientation pin centroid offset to the given orientation."""
    ox, oy = float(base[0]), float(base[1])
    if orient == 1: ox = -ox          # FN: flip X
    elif orient == 2: ox, oy = -ox, -oy  # S:  flip X+Y
    elif orient == 3: oy = -oy        # FS: flip Y
    return np.array([ox, oy], dtype=np.float32)


class HybridAnalyticalPlacer:
    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 50000,
        lr: float = 1.0,
        momentum: float = 0.9,
        soft_macro_lr: float = 0.15,
        verbose: bool = True,
    ):
        self.seed = seed
        self.num_steps = num_steps
        self.lr = lr
        self.momentum = momentum
        self.soft_macro_lr = soft_macro_lr
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        placement = benchmark.macro_positions.clone().float()
        num_hard = benchmark.num_hard_macros
        num_all = benchmark.num_macros
        if num_all == 0:
            return placement

        plc = self._load_plc_for_logging(benchmark)
        nets = self._extract_nets(benchmark, plc)
        if not nets:
            return placement

        net_indices, net_mask, net_weights = self._precompute_net_tensors(nets)

        hard_sizes = benchmark.macro_sizes[:num_hard]
        hw_hard = hard_sizes[:, 0] / 2
        hh_hard = hard_sizes[:, 1] / 2

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

        # Routing params for TILOS-matched fast proxy
        h_rpm, v_rpm, smooth_range, h_alloc, v_alloc = self._get_routing_params(plc)
        grid_h_routes = np.float32(float(self._bin_h) * h_rpm)
        grid_v_routes = np.float32(float(self._bin_w) * v_rpm)

        _t_legal = time()
        placement = self._legalize_fast(placement, benchmark, gap=0.02, max_iters=400)
        overlaps = self._hard_overlap_count(placement, benchmark)
        print(f"Initial legalization: {time()-_t_legal:.1f}s  overlaps={overlaps}")
        # if overlaps > num_hard // 3:
        #     print(f"fast legal failed, applying strong legalization to fix {overlaps}")
        #     placement = strong_legalize(placement, benchmark, gap=0.02, max_iters=40)
        #     print(f"Strong legalization done: {time()-_t_legal:.1f}s total")

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

        start_time = time()
        total_budget = 1200  # seconds — competition allows ~10min; leave margin for overhead
        op_time_budget = 400
        step = 0
        track_proxies = []
        print("starting iters")
        target_grid = None
        rudy = None
        tilos_cong = 1.0  # updated every 200 steps via plc; used to scale RUDY weight
        for step in range(self.num_steps):
            progress = step / self.num_steps
            current_lr = base_lr * (0.5 * (1 + math.cos(math.pi * progress)))
            current_lr = max(current_lr, base_lr * 0.05)

            if time() - start_time > op_time_budget:
                print(f"Time budget reached at step {step}")
                break

            if step % 200 == 0 and step > 0:
                if self._hard_overlap_count(placement, benchmark) > 0:
                    old_pos = placement[:num_hard].clone()
                    placement = self._legalize_fast(placement, benchmark, gap=0.01, max_iters=40)
                    moved = (placement[:num_hard] - old_pos).abs().sum(dim=1) > 1e-4
                    velocity[:num_hard][moved] = 0.0

            lookahead = placement.clone()
            lookahead[:num_hard] = placement[:num_hard] + self.momentum * velocity[:num_hard]
            lookahead.requires_grad_(True)
            loss, wl = self._compute_wl_loss(
                lookahead, net_indices, net_mask, nets, canvas_norm, net_weights=net_weights
            )
            loss.backward()
            wl_grad = lookahead.grad.detach().clone()
            lookahead.requires_grad_(False)

            grid = self._compute_density_grid_fast(placement, benchmark)
            if step % 2000 == 0:
                rudy = self._compute_rudy_map(placement, benchmark, net_indices, net_mask)
                rudy_weight = min(0.5, 0.1 * max(1.0, tilos_cong))
                target_grid = grid.mean() - rudy_weight * (rudy / (rudy.max() + 1e-8))

            density_forces = self._compute_density_force_fast(
                self._solve_poisson(grid, target_grid), placement, benchmark
            )

            if step % 200 == 0 and plc is not None:
                sync_plc(step)
                tilos_den = plc.get_density_cost()
                tilos_cong = plc.get_congestion_cost()
                if tilos_den > 0.9:
                    density_weight = min(0.005, density_weight * 1.05)
                elif tilos_den < 0.7:
                    density_weight = max(0.0001, density_weight * 0.95)

            hard_grad = wl_grad[:num_hard].clone()
            hard_grad -= density_weight * density_forces[:num_hard]
            if fixed.any():
                hard_grad[fixed[:num_hard]] = 0.0

            velocity[:num_hard] = self.momentum * velocity[:num_hard] - current_lr * hard_grad
            placement[:num_hard] = (placement[:num_hard] + velocity[:num_hard]).clamp(
                min=torch.stack([hw_hard, hh_hard], dim=1),
                max=torch.stack([benchmark.canvas_width - hw_hard, benchmark.canvas_height - hh_hard], dim=1),
            )

            soft_density_weight = 0.01
            soft_grad = wl_grad[num_hard:num_all].clone()
            soft_grad -= soft_density_weight * density_forces[num_hard:num_all]
            placement[num_hard:num_all] -= self.soft_macro_lr * soft_grad
            placement[num_hard:num_all, 0].clamp_(min=hw_all[num_hard:], max=benchmark.canvas_width - hw_all[num_hard:])
            placement[num_hard:num_all, 1].clamp_(min=hh_all[num_hard:], max=benchmark.canvas_height - hh_all[num_hard:])

            if fixed.any():
                placement[fixed] = benchmark.macro_positions[fixed]

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
                if len(track_proxies) >= 4 and all(p > track_proxies[-4] for p in track_proxies[-3:]):
                    print(f"Proxy diverging, stopping early at step {step}")
                    break

            if (step + 1) % log_every == 0 or step >= self.num_steps - 1:
                if plc is not None:
                    sync_plc(step)
                    metrics = compute_proxy_cost(placement.detach(), benchmark, plc)
                    proxy_est = metrics["proxy_cost"]
                else:
                    metrics = None
                    proxy_est = wl.item()
                self._log_stats(
                    f"step_{step+1}", benchmark, placement, plc,
                    wl=wl.item(), density_weight=density_weight, metrics=metrics,
                )
                recent_proxies.append(float(proxy_est))

            placement = placement.detach()

        print(f"Nesterov done: {time()-start_time:.1f}s total elapsed")
        # ── Post-optimization: legalize top-k and pick best ──
        best_valid_proxy = float("inf")
        best_valid_placement = None

        for proxy_est, ckpt_step, candidate in top_k_candidates:
            c = candidate.clone()
            for i in range(8):
                if self._hard_overlap_count(c, benchmark) == 0:
                    break
                c = self._legalize_fast(c, benchmark, gap=0.01 * (i + 1), max_iters=200)
            if self._hard_overlap_count(c, benchmark) > 0:
                if time() - start_time < 600:
                    print("attempting strong legalize")
                    c = strong_legalize(c, benchmark, gap=0.01, max_iters=40)
                    print(f"strong legalize finished: {self._hard_overlap_count(c, benchmark)}")
                else:
                    print("strong legalize took too long, continuing")
                    continue
            if self._hard_overlap_count(c, benchmark) == 0:
                _set_placement(plc, c.detach(), benchmark)
                proxy = compute_proxy_cost(c, benchmark, plc)["proxy_cost"]
                print(f"legalized checkpoint from step {ckpt_step} has proxy cost {proxy}")
                if proxy < best_valid_proxy:
                    best_valid_proxy = proxy
                    best_valid_placement = c.clone()

        if best_valid_placement is None:
            best_valid_placement = placement.clone()
            best_valid_placement = self._legalize_fast(best_valid_placement, benchmark, gap=0.05, max_iters=500)

        if best_valid_placement is not None and plc is not None:
            _remaining = total_budget - (time() - start_time)
            _swap_budget = max(10, min(150, _remaining - 200))
            print(f"SA swap budget: {_swap_budget:.0f}s  (elapsed={time()-start_time:.1f}s remaining={_remaining:.1f}s)")
            best_valid_placement = self._sa_refine(
                best_valid_placement, benchmark, plc,
                net_indices, net_mask, net_weights, canvas_norm,
                num_hard, num_all, fixed,
                self._leg_sep_x_base.numpy().copy(),
                self._leg_sep_y_base.numpy().copy(),
                budget=_swap_budget, checkpoint_every=200,
            )

            _remaining = total_budget - (time() - start_time)
            _disp_budget = max(10, _remaining - 30)
            print(f"SA displace budget: {_disp_budget:.0f}s  (elapsed={time()-start_time:.1f}s remaining={_remaining:.1f}s)")
            best_valid_placement = self._sa_displace(
                best_valid_placement, benchmark, plc,
                net_indices, net_mask, net_weights, canvas_norm,
                num_hard, num_all, fixed,
                self._leg_sep_x_base.numpy().copy(),
                self._leg_sep_y_base.numpy().copy(),
                grid_h_routes, grid_v_routes, smooth_range, h_alloc, v_alloc,
                budget=_disp_budget, checkpoint_every=200,
            )

        if best_valid_placement is not None:
            best_valid_placement = self._settle_soft_macros(
                best_valid_placement, net_indices, net_mask, nets, benchmark, num_hard, steps=80, lr=0.1,
            )

        final = best_valid_placement

        if self._hard_overlap_count(final, benchmark) > 0:
            print("WARNING: final has overlaps, emergency legalize")
            final = strong_legalize(final, benchmark, gap=0.05, max_iters=200)

        self._log_stats("final", benchmark, final, plc, wl=None, density_weight=density_weight)
        return final

    # ── SA: swap (WL-only accept/reject, real proxy at checkpoints) ───────────

    def _sa_refine(
        self, placement, benchmark, plc,
        net_indices, net_mask, net_weights, canvas_norm,
        num_hard, num_all, fixed, sep_x_np, sep_y_np,
        budget=400, checkpoint_every=200,
    ):
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
        print(f"SA swap start: proxy={best_proxy:.4f} wl={total_wl:.4f}")

        accepts = total = stalls = 0
        last_accept_step = 0
        t0 = time()
        _sw_check_interval = 500_000
        _sw_last_accepts = 0
        _sw_min_rate = 1e-4

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break
            if total % 100000 == 0:
                print(f"  step={total} accepts={accepts} wl={total_wl:.4f} [{time()-t0:.0f}s]", end="\r")

            if total % _sw_check_interval == 0 and total > 1_000_000:
                _sw_rate = (accepts - _sw_last_accepts) / _sw_check_interval
                if _sw_rate < _sw_min_rate:
                    print(f"\nSA swap: accept rate {_sw_rate:.2e} < {_sw_min_rate:.2e}, stopping early")
                    break
                _sw_last_accepts = accepts

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
                    print(f"  step={total} accepts={accepts} wl={total_wl:.4f} proxy={proxy:.4f} [{time()-t0:.0f}s]")
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_placement = sa_placement.clone()
                        stalls = 0
                    else:
                        sa_placement = best_placement.clone()
                        sa_pos[:] = best_placement.detach().numpy()
                        net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
                        total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                        stalls += 1
                        if stalls >= 3:
                            print("SA swap stalled")
                            break
            else:
                sa_pos[[i, j]] = sa_pos[[j, i]]

            if total - last_accept_step > 5_000_000:
                print("SA swap: no accepts in 5M attempts, stopping")
                break

        print(f"SA swap done: {total} attempts, {accepts} accepts, best proxy={best_proxy:.4f}")
        return best_placement

    # ── SA: displace (TILOS-matched fast proxy, no calibration) ──────────────

    def _sa_displace(
        self, placement, benchmark, plc,
        net_indices, net_mask, net_weights, canvas_norm,
        num_hard, num_all, fixed, sep_x_np, sep_y_np,
        grid_h_routes, grid_v_routes, smooth_range, hrouting_alloc, vrouting_alloc,
        budget=400, checkpoint_every=200,
    ):
        """
        SA displacement with TILOS-matched fast proxy.
        Density: top-10% × 0.5 (incremental).
        Congestion: TILOS net routing + macro blockage + smoothing (rebuilt every 100 accepts).
        No den_scale / cong_scale calibration needed.
        """
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
        macro_sizes_np = benchmark.macro_sizes[:num_hard].numpy().astype(np.float32)

        # Hard macro pin centroid offsets for routing (center + centroid ≈ pin cluster center)
        pin_centroid_offsets = np.zeros((num_hard, 2), dtype=np.float32)
        if len(benchmark.macro_pin_offsets) >= num_hard:
            for _i in range(num_hard):
                _off = benchmark.macro_pin_offsets[_i]
                if len(_off) > 0:
                    pin_centroid_offsets[_i] = _off.mean(0).numpy().astype(np.float32)
        pin_centroid_base = pin_centroid_offsets.copy()  # canonical N-orientation centroids

        # Orientation tracking
        macro_orientations = np.zeros(num_hard, dtype=np.int32)   # current (0=N,1=FN,2=S,3=FS)
        best_orientations  = np.zeros(num_hard, dtype=np.int32)
        plc_orientations   = np.zeros(num_hard, dtype=np.int32)   # what plc currently reflects
        if plc is not None:
            for _i in range(num_hard):
                _o = plc.get_macro_orientation(benchmark.hard_macro_indices[_i]) or "N"
                _oi = _ORIENT_NAMES.index(_o) if _o in _ORIENT_NAMES else 0
                macro_orientations[_i] = plc_orientations[_i] = _oi
                pin_centroid_offsets[_i] = _centroid_for_orient(pin_centroid_base[_i], _oi)
        best_orientations[:] = macro_orientations

        bl = self._bin_left.numpy().copy()
        br = self._bin_right.numpy().copy()
        bb = self._bin_bottom.numpy().copy()
        bt = self._bin_top.numpy().copy()
        bin_w = np.float32(float(self._bin_w))
        bin_h = np.float32(float(self._bin_h))
        bin_area = float(bin_w * bin_h)
        n_rows = benchmark.grid_rows
        n_cols = benchmark.grid_cols

        # Build density grid (incremental)
        density_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        _build_density_grid(density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt, bin_area, n_rows, n_cols, float(bin_w), float(bin_h))

        # Build TILOS routing grids
        H_raw = np.zeros((n_rows, n_cols), dtype=np.float32)
        V_raw = np.zeros((n_rows, n_cols), dtype=np.float32)
        H_mac = np.zeros((n_rows, n_cols), dtype=np.float32)
        V_mac = np.zeros((n_rows, n_cols), dtype=np.float32)

        # Warm up numba JIT on first call
        _routing_pos = sa_pos.copy(); _routing_pos[:num_hard] += pin_centroid_offsets
        _build_tilos_routing_raw(
            H_raw, V_raw, H_mac, V_mac,
            ni_np, nm_np, _routing_pos,
            num_nets, num_hard, macro_sizes_np,
            bin_w, bin_h, n_rows, n_cols,
            np.float32(hrouting_alloc), np.float32(vrouting_alloc),
            bl, br, bb, bt,
        )
        _compute_tilos_cong(H_raw, V_raw, H_mac, V_mac, n_rows, n_cols, grid_h_routes, grid_v_routes, smooth_range)

        current_den = _density_cost_tilos(density_grid)
        current_cong = _compute_tilos_cong(
            H_raw, V_raw, H_mac, V_mac,
            n_rows, n_cols, grid_h_routes, grid_v_routes, smooth_range,
        )
        current_proxy = total_wl + np.float32(0.5) * current_den + np.float32(0.5) * current_cong

        _set_placement(plc, sa_placement.detach(), benchmark)
        real_metrics = compute_proxy_cost(sa_placement.detach(), benchmark, plc)
        best_proxy = real_metrics["proxy_cost"]
        best_placement = sa_placement.clone()

        print(
            f"SA displace start: real={best_proxy:.4f} fast={float(current_proxy):.4f} "
            f"wl={total_wl:.4f} den={float(current_den):.4f} cong={float(current_cong):.4f}"
        )

        accepts = total = 0
        last_accept_step = 0
        t0 = time()
        _disp_start = canvas_norm * 0.12  # start large to allow restructuring
        _disp_end   = canvas_norm * 0.02  # decay to fine-tuning
        max_displacement = _disp_start
        cong_rebuild_every = 100  # rebuild TILOS congestion every N accepts

        _check_interval = 500_000
        _last_check_accepts = 0
        # Improvement stagnation: stop if best hasn't improved by min_delta in last stagnation_window checkpoints
        _stagnation_window = 5
        _min_improvement = 2e-4
        _proxy_at_window_start = best_proxy
        _checkpoints_since_improvement = 0
        _min_rate = 10e-5  # stop if window accept rate drops below this (after warmup)

        for _ in range(100_000_000):
            if time() - t0 > budget:
                break

            if total % 100000 == 0 and total > 0:
                _elapsed_frac = min(1.0, (time() - t0) / budget)
                max_displacement = _disp_start * (1 - _elapsed_frac) + _disp_end * _elapsed_frac
                print(
                    f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                    f"fast={float(current_proxy):.4f} disp={max_displacement/canvas_norm:.3f} [{time()-t0:.0f}s]",
                    end="\r",
                )

            if total % _check_interval == 0 and total > 2_000_000:
                _win_rate = (accepts - _last_check_accepts) / _check_interval
                if _win_rate < _min_rate:
                    print(f"\nSA displace: accept rate {_win_rate:.2e} < {_min_rate:.2e}, stopping early")
                    break
                _last_check_accepts = accepts

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

            # Update density incrementally
            _update_density_incr(
                density_grid, old_x, old_y, new_x, new_y,
                hw_np[i], hh_np[i], bl, br, bb, bt, bin_area,
                n_rows, n_cols, float(bin_w), float(bin_h),
            )
            new_den = _density_cost_tilos(density_grid)

            # Use cached TILOS congestion (rebuilt every cong_rebuild_every accepts)
            new_proxy = np.float32(total_wl + wl_delta) + np.float32(0.5) * new_den + np.float32(0.5) * current_cong

            if new_proxy < current_proxy:
                sa_placement[i, 0] = new_x
                sa_placement[i, 1] = new_y
                net_hpwl[aff] = new_hpwl
                total_wl += wl_delta
                current_proxy = new_proxy
                current_den = new_den
                accepts += 1
                last_accept_step = total

                # Rebuild TILOS congestion periodically
                if accepts % cong_rebuild_every == 0:
                    _routing_pos = sa_pos.copy(); _routing_pos[:num_hard] += pin_centroid_offsets
                    _build_tilos_routing_raw(
                        H_raw, V_raw, H_mac, V_mac,
                        ni_np, nm_np, _routing_pos,
                        num_nets, num_hard, macro_sizes_np,
                        bin_w, bin_h, n_rows, n_cols,
                        np.float32(hrouting_alloc), np.float32(vrouting_alloc),
                        bl, br, bb, bt,
                    )
                    current_cong = _compute_tilos_cong(
                        H_raw, V_raw, H_mac, V_mac,
                        n_rows, n_cols, grid_h_routes, grid_v_routes, smooth_range,
                    )
                    current_proxy = np.float32(total_wl) + np.float32(0.5) * current_den + np.float32(0.5) * current_cong

                # Checkpoint against real proxy
                if accepts % checkpoint_every == 0:
                    _set_placement(plc, sa_placement.detach(), benchmark)
                    real_proxy = compute_proxy_cost(sa_placement.detach(), benchmark, plc)["proxy_cost"]
                    print(
                        f"  step={total} accepts={accepts} wl={total_wl:.4f} "
                        f"fast={float(current_proxy):.4f} real={real_proxy:.4f} [{time()-t0:.0f}s]"
                    )
                    _checkpoints_since_improvement += 1
                    if real_proxy < best_proxy:
                        best_proxy = real_proxy
                        best_placement = sa_placement.clone()
                    if _checkpoints_since_improvement >= _stagnation_window:
                        if _proxy_at_window_start - best_proxy < _min_improvement:
                            print(f"\nSA displace: no meaningful improvement in {_stagnation_window} checkpoints, stopping")
                            break
                        _proxy_at_window_start = best_proxy
                        _checkpoints_since_improvement = 0
                    if real_proxy >= best_proxy:
                        # Revert to best to avoid drifting from good placement
                        sa_placement = best_placement.clone()
                        sa_pos[:] = best_placement.detach().numpy()
                        net_hpwl[:] = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)
                        total_wl = float(net_hpwl.sum()) / (num_nets * canvas_norm)
                        density_grid[:] = 0.0
                        _build_density_grid(density_grid, sa_pos, sizes_np, num_all, bl, br, bb, bt, bin_area, n_rows, n_cols, float(bin_w), float(bin_h))
                        _routing_pos = sa_pos.copy(); _routing_pos[:num_hard] += pin_centroid_offsets
                        _build_tilos_routing_raw(
                            H_raw, V_raw, H_mac, V_mac,
                            ni_np, nm_np, _routing_pos,
                            num_nets, num_hard, macro_sizes_np,
                            bin_w, bin_h, n_rows, n_cols,
                            np.float32(hrouting_alloc), np.float32(vrouting_alloc),
                            bl, br, bb, bt,
                        )
                        current_den = _density_cost_tilos(density_grid)
                        current_cong = _compute_tilos_cong(
                            H_raw, V_raw, H_mac, V_mac,
                            n_rows, n_cols, grid_h_routes, grid_v_routes, smooth_range,
                        )
                        current_proxy = np.float32(total_wl) + np.float32(0.5) * current_den + np.float32(0.5) * current_cong
            else:
                # Revert
                sa_pos[i, 0] = old_x
                sa_pos[i, 1] = old_y
                _update_density_incr(
                    density_grid, new_x, new_y, old_x, old_y,
                    hw_np[i], hh_np[i], bl, br, bb, bt, bin_area,
                    n_rows, n_cols, float(bin_w), float(bin_h),
                )

            if total - last_accept_step > 5_000_000:
                print("SA displace: no accepts in 5M attempts, stopping")
                break

        print(f"SA displace done: {total} attempts, {accepts} accepts, best real_proxy={best_proxy:.4f}")
        return best_placement

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_routing_params(self, plc):
        """Extract TILOS routing params from plc, or return typical defaults."""
        defaults = (70.0, 70.0, 2, 0.4, 0.4)
        if plc is None:
            return defaults
        try:
            h_rpm = float(plc.hroutes_per_micron) if plc.hroutes_per_micron > 0 else 70.0
            v_rpm = float(plc.vroutes_per_micron) if plc.vroutes_per_micron > 0 else 70.0
            sr = int(plc.smooth_range) if plc.smooth_range >= 0 else 2
            h_alloc = float(plc.hrouting_alloc) if plc.hrouting_alloc > 0 else 0.4
            v_alloc = float(plc.vrouting_alloc) if plc.vrouting_alloc > 0 else 0.4
            return h_rpm, v_rpm, sr, h_alloc, v_alloc
        except Exception:
            return defaults

    def _build_incremental_wl(self, net_indices, net_mask, net_weights, placement, num_all, canvas_norm):
        ni_np = net_indices.numpy().copy()
        nm_np = net_mask.numpy().copy()
        nw_np = net_weights.numpy().copy()
        num_nets = ni_np.shape[0]

        sa_pos = placement.detach().numpy().copy()

        macro_to_nets = [[] for _ in range(num_all)]
        for net_idx in range(num_nets):
            for d in range(ni_np.shape[1]):
                if not nm_np[net_idx, d]:
                    break
                macro_to_nets[ni_np[net_idx, d]].append(net_idx)
        macro_to_nets = [np.array(v, dtype=np.int32) for v in macro_to_nets]

        _ = _hpwl_batch(np.array([0], dtype=np.int32), ni_np, nm_np, sa_pos)
        net_hpwl = _hpwl_batch(np.arange(num_nets, dtype=np.int32), ni_np, nm_np, sa_pos)

        def eval_delta(i, j):
            aff = np.union1d(macro_to_nets[i], macro_to_nets[j])
            old_vals = net_hpwl[aff]
            new_vals = _hpwl_batch(aff, ni_np, nm_np, sa_pos)
            delta = float((new_vals - old_vals).sum()) / (num_nets * canvas_norm)
            return delta, aff, new_vals

        total_wl = float((net_hpwl * nw_np).sum()) / (num_nets * canvas_norm)
        return sa_pos, macro_to_nets, net_hpwl, eval_delta, total_wl

    def _compute_rudy_map(self, placement, benchmark, net_indices, net_mask):
        # Vectorized 2D prefix-sum RUDY. O(num_nets) scatter + cumsum, no Python loop.
        n_rows, n_cols = benchmark.grid_rows, benchmark.grid_cols
        pos = placement.detach()
        pos_net = pos[net_indices]  # [num_nets, max_deg, 2]
        x = pos_net[:, :, 0]
        y = pos_net[:, :, 1]
        x_min = x.masked_fill(~net_mask, float('inf')).min(dim=1).values.clamp(0, benchmark.canvas_width)
        x_max = x.masked_fill(~net_mask, float('-inf')).max(dim=1).values.clamp(0, benchmark.canvas_width)
        y_min = y.masked_fill(~net_mask, float('inf')).min(dim=1).values.clamp(0, benchmark.canvas_height)
        y_max = y.masked_fill(~net_mask, float('-inf')).max(dim=1).values.clamp(0, benchmark.canvas_height)
        demand = 1.0 / ((x_max - x_min + 1e-6) * (y_max - y_min + 1e-6))
        c_lo = (x_min / self._bin_w).long().clamp(0, n_cols - 1)
        c_hi = (x_max / self._bin_w).long().clamp(0, n_cols - 1)
        r_lo = (y_min / self._bin_h).long().clamp(0, n_rows - 1)
        r_hi = (y_max / self._bin_h).long().clamp(0, n_rows - 1)
        # 2D prefix-sum trick: scatter ±demand at box corners, then cumsum twice
        diff = torch.zeros(n_rows + 1, n_cols + 1)
        idx_r_lo = r_lo * (n_cols + 1) + c_lo
        idx_r_lo_c_hi = r_lo * (n_cols + 1) + (c_hi + 1).clamp(max=n_cols)
        idx_r_hi_c_lo = (r_hi + 1).clamp(max=n_rows) * (n_cols + 1) + c_lo
        idx_r_hi_c_hi = (r_hi + 1).clamp(max=n_rows) * (n_cols + 1) + (c_hi + 1).clamp(max=n_cols)
        diff.view(-1).scatter_add_(0, idx_r_lo, demand)
        diff.view(-1).scatter_add_(0, idx_r_lo_c_hi, -demand)
        diff.view(-1).scatter_add_(0, idx_r_hi_c_lo, -demand)
        diff.view(-1).scatter_add_(0, idx_r_hi_c_hi, demand)
        return diff.cumsum(0).cumsum(1)[:n_rows, :n_cols].clamp(min=0)

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
                    / design / "netlist" / "output_CT_Grouping"
                )
                if (base / "netlist.pb.txt").exists():
                    _, plc = load_benchmark(
                        str(base / "netlist.pb.txt"),
                        str(base / "initial.plc"),
                        name=benchmark.name,
                    )
                    return plc
        except Exception:
            pass
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

    def _precompute_net_tensors(self, nets: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_degree = max(len(n) for n in nets)
        net_indices = torch.zeros(len(nets), max_degree, dtype=torch.long)
        net_mask = torch.zeros(len(nets), max_degree, dtype=torch.bool)
        for i, net in enumerate(nets):
            net_indices[i, :len(net)] = torch.tensor(net, dtype=torch.long)
            net_mask[i, :len(net)] = True
        degrees = torch.tensor([len(n) for n in nets], dtype=torch.float32)
        net_weights = torch.log2(degrees.clamp(min=2))
        net_weights /= net_weights.mean()
        return net_indices, net_mask, net_weights

    def _compute_wl_loss(
        self, placement, net_indices, net_mask, nets, canvas_norm, net_weights=None, alpha=6.0,
    ):
        pos_net = placement[net_indices]
        x = pos_net[:, :, 0]
        y = pos_net[:, :, 1]
        x_max = (1 / alpha) * torch.logsumexp(alpha * x.masked_fill(~net_mask, float("-inf")), dim=1)
        x_min = (-1 / alpha) * torch.logsumexp(-alpha * x.masked_fill(~net_mask, float("inf")), dim=1)
        y_max = (1 / alpha) * torch.logsumexp(alpha * y.masked_fill(~net_mask, float("-inf")), dim=1)
        y_min = (-1 / alpha) * torch.logsumexp(-alpha * y.masked_fill(~net_mask, float("inf")), dim=1)
        span = (x_max - x_min) + (y_max - y_min)
        if net_weights is not None:
            span = span * net_weights
        wl = span.sum() / (len(nets) * canvas_norm)
        return wl, wl

    def _settle_soft_macros(self, placement, net_indices, net_mask, nets, benchmark, num_hard, steps=80, lr=0.1):
        num_all = num_hard + benchmark.num_soft_macros
        all_sizes = benchmark.macro_sizes[:num_all]
        hw = all_sizes[:, 0] / 2
        hh = all_sizes[:, 1] / 2
        canvas_norm = benchmark.canvas_width + benchmark.canvas_height
        placement = placement.clone()
        for s in range(steps):
            placement.requires_grad_(True)
            alpha = 5.0 + 8.0 * (s / steps)
            loss, _ = self._compute_wl_loss(placement, net_indices, net_mask, nets, canvas_norm, net_weights=None, alpha=alpha)
            loss.backward()
            soft_grad = placement.grad.detach()[num_hard:num_all]
            placement.requires_grad_(False)
            placement.data[num_hard:num_all] -= lr * soft_grad
            placement.data[num_hard:num_all, 0].clamp_(min=hw[num_hard:], max=benchmark.canvas_width - hw[num_hard:])
            placement.data[num_hard:num_all, 1].clamp_(min=hh[num_hard:], max=benchmark.canvas_height - hh[num_hard:])
            placement = placement.detach()
        return placement

    def _compute_density_grid_fast(self, placement, benchmark, inflation=1.0):
        num_macros = benchmark.num_macros
        sizes = benchmark.macro_sizes[:num_macros]
        if inflation != 1.0:
            inflated = sizes.clone()
            inflated[:benchmark.num_hard_macros] *= inflation
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

    def _solve_poisson(self, density_grid, target_grid=None):
        if target_grid is None:
            rho = density_grid - density_grid.mean()
        else:
            rho = density_grid - target_grid
        rho_freq = torch.tensor(dctn(rho.numpy()), dtype=torch.float32)
        potential = torch.tensor(idctn((rho_freq / self._poisson_eigenvalues).numpy()), dtype=torch.float32)
        return -potential

    def _compute_density_force_fast(self, potential, placement, benchmark):
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

    def _legalize_fast(self, placement, benchmark, gap=0.01, max_iters=500):
        placement = placement.clone()
        num_hard = benchmark.num_hard_macros
        sep_x = self._leg_sep_x_base + gap
        sep_y = self._leg_sep_y_base + gap
        half_w = self._leg_half_w
        half_h = self._leg_half_h

        for _ in range(max_iters):
            pos = placement[:num_hard]
            dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
            dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
            abs_dx = torch.abs(dx)
            abs_dy = torch.abs(dy)
            overlap_mask = (abs_dx < sep_x) & (abs_dy < sep_y) & self._leg_tri
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

    def _hard_overlap_count(self, placement, benchmark):
        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return 0
        p = placement[:num_hard]
        dx = torch.abs(p.unsqueeze(0)[:, :, 0] - p.unsqueeze(1)[:, :, 0])
        dy = torch.abs(p.unsqueeze(0)[:, :, 1] - p.unsqueeze(1)[:, :, 1])
        return int(((dx < self._leg_sep_x_base) & (dy < self._leg_sep_y_base) & self._leg_tri).sum().item())

    def _log_stats(self, label, benchmark, placement, plc, wl, density_weight, metrics=None):
        if not self.verbose:
            return
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
            overlaps = self._hard_overlap_count(placement, benchmark)
            print(
                f"[{benchmark.name}] {label:<12} "
                f"wl={float(wl) if wl is not None else float('nan'):.4f} "
                f"ovlp={overlaps} dw={density_weight:.4f}",
                flush=True,
            )
