"""Incremental density/congestion cost helpers for HAPpy Placer.

The goal of this module is exactness first: keep TILOS-shaped density and
smoothed HV congestion costs current while existing numba kernels mutate the
underlying grids. The production fast path maintains the smoothed H/V route
grids incrementally in numba, then reads the exact top-5 congestion cost from
those already-smoothed arrays.
"""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np
from numba import njit


class TopKMeanTracker:
    """Maintain the mean of the largest k values under point updates."""

    def __init__(self, values: np.ndarray, frac: float, scale: float = 1.0):
        flat = np.asarray(values, dtype=np.float32).reshape(-1).copy()
        self.values = flat
        self.n = int(flat.size)
        self.k = max(1, int(self.n * frac))
        self.scale = float(scale)
        self.in_top = np.zeros(self.n, dtype=np.bool_)
        if self.k >= self.n:
            top_idx = np.arange(self.n, dtype=np.int64)
        else:
            top_idx = np.argpartition(flat, -self.k)[-self.k:]
        self.in_top[top_idx] = True
        self.top_sum = float(flat[top_idx].sum(dtype=np.float64))
        self.top_size = int(len(top_idx))
        self.top_heap: list[tuple[float, int]] = []
        self.rest_heap: list[tuple[float, int]] = []
        self._rebuild_heaps()

    def _rebuild_heaps(self) -> None:
        self.top_heap.clear()
        self.rest_heap.clear()
        for idx, val in enumerate(self.values):
            if self.in_top[idx]:
                heapq.heappush(self.top_heap, (float(val), idx))
            else:
                heapq.heappush(self.rest_heap, (-float(val), idx))

    def _clean_top(self) -> None:
        while self.top_heap:
            val, idx = self.top_heap[0]
            if self.in_top[idx] and float(self.values[idx]) == val:
                return
            heapq.heappop(self.top_heap)

    def _clean_rest(self) -> None:
        while self.rest_heap:
            neg_val, idx = self.rest_heap[0]
            val = -neg_val
            if (not self.in_top[idx]) and float(self.values[idx]) == val:
                return
            heapq.heappop(self.rest_heap)

    def _pop_valid_top(self) -> tuple[float, int]:
        self._clean_top()
        return heapq.heappop(self.top_heap)

    def _pop_valid_rest(self) -> tuple[float, int]:
        self._clean_rest()
        neg_val, idx = heapq.heappop(self.rest_heap)
        return -neg_val, idx

    def _rebalance(self) -> None:
        while self.top_size < self.k and self.rest_heap:
            val, idx = self._pop_valid_rest()
            self.in_top[idx] = True
            self.top_size += 1
            self.top_sum += val
            heapq.heappush(self.top_heap, (val, idx))

        while self.top_size > self.k:
            val, idx = self._pop_valid_top()
            self.in_top[idx] = False
            self.top_size -= 1
            self.top_sum -= val
            heapq.heappush(self.rest_heap, (-val, idx))

        while True:
            self._clean_top()
            self._clean_rest()
            if not self.top_heap or not self.rest_heap:
                break
            top_min, top_idx = self.top_heap[0]
            rest_max = -self.rest_heap[0][0]
            rest_idx = self.rest_heap[0][1]
            if rest_max <= top_min:
                break
            heapq.heappop(self.top_heap)
            heapq.heappop(self.rest_heap)
            self.in_top[top_idx] = False
            self.in_top[rest_idx] = True
            self.top_sum += rest_max - top_min
            heapq.heappush(self.rest_heap, (-top_min, top_idx))
            heapq.heappush(self.top_heap, (rest_max, rest_idx))

        if len(self.top_heap) + len(self.rest_heap) > 8 * self.n:
            self._rebuild_heaps()

    def update(self, idx: int, new_value: float) -> None:
        idx = int(idx)
        old_value = float(self.values[idx])
        new_value = float(np.float32(new_value))
        if old_value == new_value:
            return
        self.values[idx] = new_value
        if self.in_top[idx]:
            self.top_sum += new_value - old_value
            heapq.heappush(self.top_heap, (new_value, idx))
        else:
            heapq.heappush(self.rest_heap, (-new_value, idx))
        self._rebalance()

    def update_many(self, indices: Iterable[int], new_values: Iterable[float]) -> None:
        for idx, val in zip(indices, new_values):
            self.update(int(idx), float(val))

    def mean(self) -> float:
        return self.top_sum / max(1, self.k)

    def cost(self) -> float:
        return self.scale * self.mean()

    def reference_cost(self) -> float:
        idx = np.argpartition(self.values, -self.k)[-self.k:]
        return self.scale * float(self.values[idx].mean())


def update_density_tracker_from_diff(
    tracker: TopKMeanTracker,
    old_grid: np.ndarray,
    new_grid: np.ndarray,
    eps: float = 1e-8,
) -> None:
    diff = np.asarray(new_grid) - np.asarray(old_grid)
    rows, cols = np.nonzero(np.abs(diff) > eps)
    n_cols = new_grid.shape[1]
    for r, c in zip(rows, cols):
        tracker.update(int(r) * n_cols + int(c), float(new_grid[r, c]))


def update_density_incr_tracked(
    grid: np.ndarray,
    tracker: TopKMeanTracker,
    old_cx: float,
    old_cy: float,
    new_cx: float,
    new_cy: float,
    hw: float,
    hh: float,
    bl: np.ndarray,
    br: np.ndarray,
    bb: np.ndarray,
    bt: np.ndarray,
    bin_area: float,
    n_rows: int,
    n_cols: int,
    bin_w: float,
    bin_h: float,
) -> None:
    _add_density_rect(grid, tracker, old_cx, old_cy, hw, hh, bl, br, bb, bt,
                      bin_area, n_rows, n_cols, bin_w, bin_h, -1.0)
    _add_density_rect(grid, tracker, new_cx, new_cy, hw, hh, bl, br, bb, bt,
                      bin_area, n_rows, n_cols, bin_w, bin_h, 1.0)


def _add_density_rect(
    grid: np.ndarray,
    tracker: TopKMeanTracker,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    bl: np.ndarray,
    br: np.ndarray,
    bb: np.ndarray,
    bt: np.ndarray,
    bin_area: float,
    n_rows: int,
    n_cols: int,
    bin_w: float,
    bin_h: float,
    sign: float,
) -> None:
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
            if ox > 0.0 and oy > 0.0:
                grid[r, c] += np.float32(sign * ox * oy / bin_area)
                tracker.update(r * n_cols + c, float(grid[r, c]))


class SmoothHVTopKTracker:
    """Track TILOS-style smoothed HV congestion top-5 mean."""

    def __init__(
        self,
        h_route: np.ndarray,
        v_route: np.ndarray,
        h_macro: np.ndarray,
        v_macro: np.ndarray,
        smooth_range: int,
    ) -> None:
        self.h_route = h_route
        self.v_route = v_route
        self.h_macro = h_macro
        self.v_macro = v_macro
        self.smooth_range = smooth_range
        self.n_rows = int(self.h_route.shape[0])
        self.n_cols = int(self.h_route.shape[1])
        self.n = self.n_rows * self.n_cols
        self.smooth_range = max(0, int(self.smooth_range))
        self.h_smooth = np.zeros_like(self.h_route, dtype=np.float32)
        self.v_smooth = np.zeros_like(self.v_route, dtype=np.float32)
        self._rebuild_smooth()
        flat = np.empty(self.n * 2, dtype=np.float32)
        flat[: self.n] = (self.v_smooth + self.v_macro).reshape(-1)
        flat[self.n :] = (self.h_smooth + self.h_macro).reshape(-1)
        self.tracker = TopKMeanTracker(flat, frac=0.05, scale=1.0)

    def _rebuild_smooth(self) -> None:
        sr = self.smooth_range
        self.v_smooth.fill(0.0)
        self.h_smooth.fill(0.0)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                lp = max(0, c - sr)
                rp = min(self.n_cols - 1, c + sr)
                self.v_smooth[r, lp : rp + 1] += self.v_route[r, c] / (rp - lp + 1)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                lp = max(0, r - sr)
                up = min(self.n_rows - 1, r + sr)
                self.h_smooth[lp : up + 1, c] += self.h_route[r, c] / (up - lp + 1)

    def cost(self) -> float:
        return self.tracker.cost()

    def reference_cost(self) -> float:
        flat = np.empty(self.n * 2, dtype=np.float32)
        flat[: self.n] = (self.v_smooth + self.v_macro).reshape(-1)
        flat[self.n :] = (self.h_smooth + self.h_macro).reshape(-1)
        k = max(1, int(flat.size * 0.05))
        idx = np.argpartition(flat, -k)[-k:]
        return float(flat[idx].mean())

    def _update_v_total(self, r: int, c: int) -> None:
        idx = int(r) * self.n_cols + int(c)
        self.tracker.update(idx, float(self.v_smooth[r, c] + self.v_macro[r, c]))

    def _update_h_total(self, r: int, c: int) -> None:
        idx = self.n + int(r) * self.n_cols + int(c)
        self.tracker.update(idx, float(self.h_smooth[r, c] + self.h_macro[r, c]))

    def refresh_v_macro_cell(self, r: int, c: int) -> None:
        self._update_v_total(r, c)

    def refresh_h_macro_cell(self, r: int, c: int) -> None:
        self._update_h_total(r, c)

    def apply_v_route_delta(self, r: int, c: int, delta: float) -> None:
        if delta == 0.0:
            return
        sr = self.smooth_range
        lp = max(0, int(c) - sr)
        rp = min(self.n_cols - 1, int(c) + sr)
        inc = float(delta) / (rp - lp + 1)
        for cc in range(lp, rp + 1):
            self.v_smooth[r, cc] += inc
            self._update_v_total(r, cc)

    def apply_h_route_delta(self, r: int, c: int, delta: float) -> None:
        if delta == 0.0:
            return
        sr = self.smooth_range
        lp = max(0, int(r) - sr)
        up = min(self.n_rows - 1, int(r) + sr)
        inc = float(delta) / (up - lp + 1)
        for rr in range(lp, up + 1):
            self.h_smooth[rr, c] += inc
            self._update_h_total(rr, c)

    def apply_route_grid_diff(
        self,
        old_h_route: np.ndarray,
        old_v_route: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        dv = self.v_route - old_v_route
        rows, cols = np.nonzero(np.abs(dv) > eps)
        for r, c in zip(rows, cols):
            self.apply_v_route_delta(int(r), int(c), float(dv[r, c]))
        dh = self.h_route - old_h_route
        rows, cols = np.nonzero(np.abs(dh) > eps)
        for r, c in zip(rows, cols):
            self.apply_h_route_delta(int(r), int(c), float(dh[r, c]))

    def apply_macro_grid_diff(
        self,
        old_h_macro: np.ndarray,
        old_v_macro: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        dv = self.v_macro - old_v_macro
        rows, cols = np.nonzero(np.abs(dv) > eps)
        for r, c in zip(rows, cols):
            self._update_v_total(int(r), int(c))
        dh = self.h_macro - old_h_macro
        rows, cols = np.nonzero(np.abs(dh) > eps)
        for r, c in zip(rows, cols):
            self._update_h_total(int(r), int(c))


class SmoothHVCostTracker:
    """Maintain smoothed HV route grids; compute exact top-5 cost on read.

    A Python top-K heap sounds asymptotically nicer, but on the challenge grid
    sizes the Python update overhead dominates. This tracker keeps the
    expensive smoothing incremental in numba and leaves only the already-flat
    top-K read to a numba argpartition.
    """

    def __init__(
        self,
        h_route: np.ndarray,
        v_route: np.ndarray,
        h_macro: np.ndarray,
        v_macro: np.ndarray,
        smooth_range: int,
    ) -> None:
        self.h_route = h_route
        self.v_route = v_route
        self.h_macro = h_macro
        self.v_macro = v_macro
        self.smooth_range = max(0, int(smooth_range))
        self.h_smooth = np.zeros_like(self.h_route, dtype=np.float32)
        self.v_smooth = np.zeros_like(self.v_route, dtype=np.float32)
        _rebuild_smooth_numba(
            self.h_route, self.v_route, self.h_smooth, self.v_smooth, self.smooth_range
        )

    def cost(self) -> float:
        return float(_hv_congestion_cost_from_smooth(
            self.h_smooth, self.v_smooth, self.h_macro, self.v_macro
        ))

    def reference_cost(self) -> float:
        return self.cost()

    def refresh_v_macro_cell(self, r: int, c: int) -> None:
        # Macro blockage is unsmoothed; cost() reads v_macro directly.
        return None

    def refresh_h_macro_cell(self, r: int, c: int) -> None:
        # Macro blockage is unsmoothed; cost() reads h_macro directly.
        return None

    def apply_v_route_delta(self, r: int, c: int, delta: float) -> None:
        _apply_v_smooth_delta(
            self.v_smooth, int(r), int(c), float(delta), self.smooth_range,
            self.v_smooth.shape[1],
        )

    def apply_h_route_delta(self, r: int, c: int, delta: float) -> None:
        _apply_h_smooth_delta(
            self.h_smooth, int(r), int(c), float(delta), self.smooth_range,
            self.h_smooth.shape[0],
        )

    def apply_route_grid_diff(
        self,
        old_h_route: np.ndarray,
        old_v_route: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        dv = self.v_route - old_v_route
        rows, cols = np.nonzero(np.abs(dv) > eps)
        for r, c in zip(rows, cols):
            self.apply_v_route_delta(int(r), int(c), float(dv[r, c]))
        dh = self.h_route - old_h_route
        rows, cols = np.nonzero(np.abs(dh) > eps)
        for r, c in zip(rows, cols):
            self.apply_h_route_delta(int(r), int(c), float(dh[r, c]))

    def apply_macro_grid_diff(
        self,
        old_h_macro: np.ndarray,
        old_v_macro: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        # Macro grids are read directly in cost(); no cached total exists.
        return None


@njit
def _rebuild_smooth_numba(h_route, v_route, h_smooth, v_smooth, smooth_range):
    n_rows = h_route.shape[0]
    n_cols = h_route.shape[1]
    h_smooth[:, :] = 0.0
    v_smooth[:, :] = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            lp = c - smooth_range
            if lp < 0:
                lp = 0
            rp = c + smooth_range
            if rp >= n_cols:
                rp = n_cols - 1
            count = rp - lp + 1
            val = v_route[r, c] / count
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
            val = h_route[r, c] / count
            for rr in range(lp, up + 1):
                h_smooth[rr, c] += val


@njit
def _hv_congestion_cost_from_smooth(h_smooth, v_smooth, h_macro, v_macro):
    n_rows = h_smooth.shape[0]
    n_cols = h_smooth.shape[1]
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
def _apply_v_smooth_delta(v_smooth, r, c, delta, smooth_range, n_cols):
    if delta == 0.0:
        return
    lp = c - smooth_range
    if lp < 0:
        lp = 0
    rp = c + smooth_range
    if rp >= n_cols:
        rp = n_cols - 1
    inc = delta / (rp - lp + 1)
    for cc in range(lp, rp + 1):
        v_smooth[r, cc] += inc


@njit
def _apply_h_smooth_delta(h_smooth, r, c, delta, smooth_range, n_rows):
    if delta == 0.0:
        return
    lp = r - smooth_range
    if lp < 0:
        lp = 0
    up = r + smooth_range
    if up >= n_rows:
        up = n_rows - 1
    inc = delta / (up - lp + 1)
    for rr in range(lp, up + 1):
        h_smooth[rr, c] += inc


def update_macro_route_incr_tracked(
    h_macro: np.ndarray,
    v_macro: np.ndarray,
    tracker: SmoothHVTopKTracker,
    old_x: float,
    old_y: float,
    new_x: float,
    new_y: float,
    hw: float,
    hh: float,
    bl: np.ndarray,
    br: np.ndarray,
    bb: np.ndarray,
    bt: np.ndarray,
    n_rows: int,
    n_cols: int,
    bin_w: float,
    bin_h: float,
    hcap: float,
    vcap: float,
    h_alloc: float,
    v_alloc: float,
) -> None:
    _add_macro_route_blockage_tracked(
        h_macro, v_macro, tracker, old_x, old_y, hw, hh, bl, br, bb, bt,
        n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc, -1.0,
    )
    _add_macro_route_blockage_tracked(
        h_macro, v_macro, tracker, new_x, new_y, hw, hh, bl, br, bb, bt,
        n_rows, n_cols, bin_w, bin_h, hcap, vcap, h_alloc, v_alloc, 1.0,
    )


def _add_macro_route_blockage_tracked(
    h_macro: np.ndarray,
    v_macro: np.ndarray,
    tracker: SmoothHVTopKTracker,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    bl: np.ndarray,
    br: np.ndarray,
    bb: np.ndarray,
    bt: np.ndarray,
    n_rows: int,
    n_cols: int,
    bin_w: float,
    bin_h: float,
    hcap: float,
    vcap: float,
    h_alloc: float,
    v_alloc: float,
    sign: float,
) -> None:
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
                v_macro[r, c] += np.float32(sign * ox * v_alloc / vcap)
                h_macro[r, c] += np.float32(sign * oy * h_alloc / hcap)
                tracker.refresh_v_macro_cell(r, c)
                tracker.refresh_h_macro_cell(r, c)
    if partial_vertical:
        r = r1
        for c in range(c0, c1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0.0 and oy > 0.0:
                v_macro[r, c] -= np.float32(sign * ox * v_alloc / vcap)
                tracker.refresh_v_macro_cell(r, c)
    if partial_horizontal:
        c = c1
        for r in range(r0, r1 + 1):
            ox = min(right, br[c]) - max(left, bl[c])
            oy = min(top, bt[r]) - max(bottom, bb[r])
            if ox > 0.0 and oy > 0.0:
                h_macro[r, c] -= np.float32(sign * oy * h_alloc / hcap)
                tracker.refresh_h_macro_cell(r, c)


def update_hv_route_incr_single_tracked(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    ni_np: np.ndarray,
    nm_np: np.ndarray,
    nw_np: np.ndarray,
    pos: np.ndarray,
    affected_nets: np.ndarray,
    macro_i: int,
    old_x: float,
    old_y: float,
    bin_w: float,
    bin_h: float,
    n_rows: int,
    n_cols: int,
    hcap: float,
    vcap: float,
) -> None:
    h_delta = np.zeros_like(h_grid, dtype=np.float32)
    v_delta = np.zeros_like(v_grid, dtype=np.float32)
    _update_hv_route_incr_single_delta(
        h_grid, v_grid, h_delta, v_delta, ni_np, nm_np, nw_np, pos, affected_nets,
        macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
    )
    _apply_route_delta_grids(tracker, h_delta, v_delta)


def update_pin_hv_route_incr_single_tracked(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    pin_owner_np: np.ndarray,
    pin_mask_np: np.ndarray,
    pin_xoff_np: np.ndarray,
    pin_yoff_np: np.ndarray,
    pin_fixed_x_np: np.ndarray,
    pin_fixed_y_np: np.ndarray,
    nw_np: np.ndarray,
    pos: np.ndarray,
    affected_nets: np.ndarray,
    macro_i: int,
    old_x: float,
    old_y: float,
    bin_w: float,
    bin_h: float,
    n_rows: int,
    n_cols: int,
    hcap: float,
    vcap: float,
) -> None:
    h_delta = np.zeros_like(h_grid, dtype=np.float32)
    v_delta = np.zeros_like(v_grid, dtype=np.float32)
    _update_pin_hv_route_incr_single_delta(
        h_grid, v_grid, h_delta, v_delta,
        pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
        pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, affected_nets,
        macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
    )
    _apply_route_delta_grids(tracker, h_delta, v_delta)


def update_hv_route_incr_single_smooth(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVCostTracker,
    ni_np: np.ndarray,
    nm_np: np.ndarray,
    nw_np: np.ndarray,
    pos: np.ndarray,
    affected_nets: np.ndarray,
    macro_i: int,
    old_x: float,
    old_y: float,
    bin_w: float,
    bin_h: float,
    n_rows: int,
    n_cols: int,
    hcap: float,
    vcap: float,
) -> None:
    _update_hv_route_incr_single_smooth(
        h_grid, v_grid, tracker.h_smooth, tracker.v_smooth,
        ni_np, nm_np, nw_np, pos, affected_nets,
        macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
        tracker.smooth_range,
    )


def update_pin_hv_route_incr_single_smooth(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVCostTracker,
    pin_owner_np: np.ndarray,
    pin_mask_np: np.ndarray,
    pin_xoff_np: np.ndarray,
    pin_yoff_np: np.ndarray,
    pin_fixed_x_np: np.ndarray,
    pin_fixed_y_np: np.ndarray,
    nw_np: np.ndarray,
    pos: np.ndarray,
    affected_nets: np.ndarray,
    macro_i: int,
    old_x: float,
    old_y: float,
    bin_w: float,
    bin_h: float,
    n_rows: int,
    n_cols: int,
    hcap: float,
    vcap: float,
) -> None:
    _update_pin_hv_route_incr_single_smooth(
        h_grid, v_grid, tracker.h_smooth, tracker.v_smooth,
        pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
        pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, affected_nets,
        macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
        tracker.smooth_range,
    )


def _apply_route_delta_grids(
    tracker: SmoothHVTopKTracker,
    h_delta: np.ndarray,
    v_delta: np.ndarray,
    eps: float = 1e-12,
) -> None:
    rows, cols = np.nonzero(np.abs(v_delta) > eps)
    for r, c in zip(rows, cols):
        tracker.apply_v_route_delta(int(r), int(c), float(v_delta[r, c]))
    rows, cols = np.nonzero(np.abs(h_delta) > eps)
    for r, c in zip(rows, cols):
        tracker.apply_h_route_delta(int(r), int(c), float(h_delta[r, c]))


@njit
def _grid_cell_from_xy_delta(x, y, bin_w, bin_h, n_rows, n_cols):
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
def _add_h_segment_delta(h_grid, h_delta, row, c0, c1, weight, hcap, sign, n_rows, n_cols):
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
        old = h_grid[row, c]
        h_grid[row, c] += delta
        h_delta[row, c] += h_grid[row, c] - old


@njit
def _add_v_segment_delta(v_grid, v_delta, col, r0, r1, weight, vcap, sign, n_rows, n_cols):
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
        old = v_grid[r, col]
        v_grid[r, col] += delta
        v_delta[r, col] += v_grid[r, col] - old


@njit
def _add_two_pin_route_hv_delta(
    h_grid, v_grid, h_delta, v_delta,
    sr, sc, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols,
):
    _add_h_segment_delta(h_grid, h_delta, sr, min(sc, tc), max(sc, tc), weight, hcap, sign, n_rows, n_cols)
    _add_v_segment_delta(v_grid, v_delta, tc, min(sr, tr), max(sr, tr), weight, vcap, sign, n_rows, n_cols)


@njit
def _sort3_by_col_row_delta(r0, c0, r1, c1, r2, c2):
    if c0 > c1 or (c0 == c1 and r0 > r1):
        tr = r0
        tc = c0
        r0 = r1
        c0 = c1
        r1 = tr
        c1 = tc
    if c1 > c2 or (c1 == c2 and r1 > r2):
        tr = r1
        tc = c1
        r1 = r2
        c1 = c2
        r2 = tr
        c2 = tc
    if c0 > c1 or (c0 == c1 and r0 > r1):
        tr = r0
        tc = c0
        r0 = r1
        c0 = c1
        r1 = tr
        c1 = tc
    return r0, c0, r1, c1, r2, c2


@njit
def _sort3_by_row_col_delta(r0, c0, r1, c1, r2, c2):
    if r0 > r1 or (r0 == r1 and c0 > c1):
        tr = r0
        tc = c0
        r0 = r1
        c0 = c1
        r1 = tr
        c1 = tc
    if r1 > r2 or (r1 == r2 and c1 > c2):
        tr = r1
        tc = c1
        r1 = r2
        c1 = c2
        r2 = tr
        c2 = tc
    if r0 > r1 or (r0 == r1 and c0 > c1):
        tr = r0
        tc = c0
        r0 = r1
        c0 = c1
        r1 = tr
        c1 = tc
    return r0, c0, r1, c1, r2, c2


@njit
def _add_three_pin_route_hv_delta(
    h_grid, v_grid, h_delta, v_delta, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols,
):
    r1, c1, r2, c2, r3, c3 = _sort3_by_col_row_delta(
        rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]
    )
    if c1 < c2 and c2 < c3 and min(r1, r3) < r2 and max(r1, r3) > r2:
        _add_h_segment_delta(h_grid, h_delta, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_h_segment_delta(h_grid, h_delta, r2, c2, c3, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_delta(v_grid, v_delta, c2, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols)
        _add_v_segment_delta(v_grid, v_delta, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols)
    elif c2 == c3 and c1 < c2 and r1 < min(r2, r3):
        _add_h_segment_delta(h_grid, h_delta, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_delta(v_grid, v_delta, c2, r1, max(r2, r3), weight, vcap, sign, n_rows, n_cols)
    elif r2 == r3:
        _add_h_segment_delta(h_grid, h_delta, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_h_segment_delta(h_grid, h_delta, r2, c2, c3, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_delta(v_grid, v_delta, c2, min(r2, r1), max(r2, r1), weight, vcap, sign, n_rows, n_cols)
    else:
        r1, c1, r2, c2, r3, c3 = _sort3_by_row_col_delta(
            rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]
        )
        xmin = min(c1, min(c2, c3))
        xmax = max(c1, max(c2, c3))
        _add_h_segment_delta(h_grid, h_delta, r2, xmin, xmax, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_delta(v_grid, v_delta, c1, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols)
        _add_v_segment_delta(v_grid, v_delta, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols)


@njit
def _add_net_route_hv_delta(
    h_grid, v_grid, h_delta, v_delta, net_idx, ni_np, nm_np, nw_np, pos,
    macro_i, override_x, override_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, sign,
):
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
        r, c = _grid_cell_from_xy_delta(x, y, bin_w, bin_h, n_rows, n_cols)
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
        _add_two_pin_route_hv_delta(h_grid, v_grid, h_delta, v_delta, src_r, src_c, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count == 3:
        _add_three_pin_route_hv_delta(h_grid, v_grid, h_delta, v_delta, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count > 3:
        for k in range(count):
            if rows[k] == src_r and cols[k] == src_c:
                continue
            _add_two_pin_route_hv_delta(h_grid, v_grid, h_delta, v_delta, src_r, src_c, rows[k], cols[k], weight, hcap, vcap, sign, n_rows, n_cols)


@njit
def _add_net_route_pin_hv_delta(
    h_grid, v_grid, h_delta, v_delta, net_idx,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
    macro_i, override_x, override_y,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap, sign,
):
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
            if owner == macro_i and override_x >= 0.0:
                x = override_x + pin_xoff_np[net_idx, d]
                y = override_y + pin_yoff_np[net_idx, d]
            else:
                x = pos[owner, 0] + pin_xoff_np[net_idx, d]
                y = pos[owner, 1] + pin_yoff_np[net_idx, d]
        else:
            x = pin_fixed_x_np[net_idx, d]
            y = pin_fixed_y_np[net_idx, d]
        r, c = _grid_cell_from_xy_delta(x, y, bin_w, bin_h, n_rows, n_cols)
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
        _add_two_pin_route_hv_delta(h_grid, v_grid, h_delta, v_delta, src_r, src_c, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count == 3:
        _add_three_pin_route_hv_delta(h_grid, v_grid, h_delta, v_delta, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count > 3:
        for k in range(count):
            if rows[k] == src_r and cols[k] == src_c:
                continue
            _add_two_pin_route_hv_delta(h_grid, v_grid, h_delta, v_delta, src_r, src_c, rows[k], cols[k], weight, hcap, vcap, sign, n_rows, n_cols)


@njit
def _update_hv_route_incr_single_delta(
    h_grid, v_grid, h_delta, v_delta, ni_np, nm_np, nw_np, pos, affected_nets,
    macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
):
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        _add_net_route_hv_delta(h_grid, v_grid, h_delta, v_delta, n, ni_np, nm_np, nw_np, pos, macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, -1.0)
        _add_net_route_hv_delta(h_grid, v_grid, h_delta, v_delta, n, ni_np, nm_np, nw_np, pos, -1, -1.0, -1.0, bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0)


@njit
def _update_pin_hv_route_incr_single_delta(
    h_grid, v_grid, h_delta, v_delta,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, affected_nets,
    macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap,
):
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        _add_net_route_pin_hv_delta(
            h_grid, v_grid, h_delta, v_delta, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
            macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, -1.0,
        )
        _add_net_route_pin_hv_delta(
            h_grid, v_grid, h_delta, v_delta, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
            -1, -1.0, -1.0, bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0,
        )


@njit
def _add_h_segment_smooth(h_grid, h_smooth, row, c0, c1, weight, hcap, sign, n_rows, n_cols, smooth_range):
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
        old = h_grid[row, c]
        h_grid[row, c] += delta
        actual = h_grid[row, c] - old
        _apply_h_smooth_delta(h_smooth, row, c, actual, smooth_range, n_rows)


@njit
def _add_v_segment_smooth(v_grid, v_smooth, col, r0, r1, weight, vcap, sign, n_rows, n_cols, smooth_range):
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
        old = v_grid[r, col]
        v_grid[r, col] += delta
        actual = v_grid[r, col] - old
        _apply_v_smooth_delta(v_smooth, r, col, actual, smooth_range, n_cols)


@njit
def _add_two_pin_route_hv_smooth(
    h_grid, v_grid, h_smooth, v_smooth,
    sr, sc, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols, smooth_range,
):
    _add_h_segment_smooth(h_grid, h_smooth, sr, min(sc, tc), max(sc, tc), weight, hcap, sign, n_rows, n_cols, smooth_range)
    _add_v_segment_smooth(v_grid, v_smooth, tc, min(sr, tr), max(sr, tr), weight, vcap, sign, n_rows, n_cols, smooth_range)


@njit
def _add_three_pin_route_hv_smooth(
    h_grid, v_grid, h_smooth, v_smooth, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols, smooth_range,
):
    r1, c1, r2, c2, r3, c3 = _sort3_by_col_row_delta(
        rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]
    )
    if c1 < c2 and c2 < c3 and min(r1, r3) < r2 and max(r1, r3) > r2:
        _add_h_segment_smooth(h_grid, h_smooth, r1, c1, c2, weight, hcap, sign, n_rows, n_cols, smooth_range)
        _add_h_segment_smooth(h_grid, h_smooth, r2, c2, c3, weight, hcap, sign, n_rows, n_cols, smooth_range)
        _add_v_segment_smooth(v_grid, v_smooth, c2, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols, smooth_range)
        _add_v_segment_smooth(v_grid, v_smooth, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols, smooth_range)
    elif c2 == c3 and c1 < c2 and r1 < min(r2, r3):
        _add_h_segment_smooth(h_grid, h_smooth, r1, c1, c2, weight, hcap, sign, n_rows, n_cols, smooth_range)
        _add_v_segment_smooth(v_grid, v_smooth, c2, r1, max(r2, r3), weight, vcap, sign, n_rows, n_cols, smooth_range)
    elif r2 == r3:
        _add_h_segment_smooth(h_grid, h_smooth, r1, c1, c2, weight, hcap, sign, n_rows, n_cols, smooth_range)
        _add_h_segment_smooth(h_grid, h_smooth, r2, c2, c3, weight, hcap, sign, n_rows, n_cols, smooth_range)
        _add_v_segment_smooth(v_grid, v_smooth, c2, min(r2, r1), max(r2, r1), weight, vcap, sign, n_rows, n_cols, smooth_range)
    else:
        r1, c1, r2, c2, r3, c3 = _sort3_by_row_col_delta(
            rows[0], cols[0], rows[1], cols[1], rows[2], cols[2]
        )
        xmin = min(c1, min(c2, c3))
        xmax = max(c1, max(c2, c3))
        _add_h_segment_smooth(h_grid, h_smooth, r2, xmin, xmax, weight, hcap, sign, n_rows, n_cols, smooth_range)
        _add_v_segment_smooth(v_grid, v_smooth, c1, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols, smooth_range)
        _add_v_segment_smooth(v_grid, v_smooth, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols, smooth_range)


@njit
def _add_net_route_hv_smooth(
    h_grid, v_grid, h_smooth, v_smooth, net_idx, ni_np, nm_np, nw_np, pos,
    macro_i, override_x, override_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, sign, smooth_range,
):
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
        r, c = _grid_cell_from_xy_delta(x, y, bin_w, bin_h, n_rows, n_cols)
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
        _add_two_pin_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, src_r, src_c, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols, smooth_range)
    elif count == 3:
        _add_three_pin_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols, smooth_range)
    elif count > 3:
        for k in range(count):
            if rows[k] == src_r and cols[k] == src_c:
                continue
            _add_two_pin_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, src_r, src_c, rows[k], cols[k], weight, hcap, vcap, sign, n_rows, n_cols, smooth_range)


@njit
def _add_net_route_pin_hv_smooth(
    h_grid, v_grid, h_smooth, v_smooth, net_idx,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
    macro_i, override_x, override_y,
    bin_w, bin_h, n_rows, n_cols, hcap, vcap, sign, smooth_range,
):
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
            if owner == macro_i and override_x >= 0.0:
                x = override_x + pin_xoff_np[net_idx, d]
                y = override_y + pin_yoff_np[net_idx, d]
            else:
                x = pos[owner, 0] + pin_xoff_np[net_idx, d]
                y = pos[owner, 1] + pin_yoff_np[net_idx, d]
        else:
            x = pin_fixed_x_np[net_idx, d]
            y = pin_fixed_y_np[net_idx, d]
        r, c = _grid_cell_from_xy_delta(x, y, bin_w, bin_h, n_rows, n_cols)
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
        _add_two_pin_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, src_r, src_c, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols, smooth_range)
    elif count == 3:
        _add_three_pin_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols, smooth_range)
    elif count > 3:
        for k in range(count):
            if rows[k] == src_r and cols[k] == src_c:
                continue
            _add_two_pin_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, src_r, src_c, rows[k], cols[k], weight, hcap, vcap, sign, n_rows, n_cols, smooth_range)


@njit
def _update_hv_route_incr_single_smooth(
    h_grid, v_grid, h_smooth, v_smooth, ni_np, nm_np, nw_np, pos, affected_nets,
    macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, smooth_range,
):
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        _add_net_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, n, ni_np, nm_np, nw_np, pos, macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, -1.0, smooth_range)
        _add_net_route_hv_smooth(h_grid, v_grid, h_smooth, v_smooth, n, ni_np, nm_np, nw_np, pos, -1, -1.0, -1.0, bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0, smooth_range)


@njit
def _update_pin_hv_route_incr_single_smooth(
    h_grid, v_grid, h_smooth, v_smooth,
    pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
    pin_fixed_x_np, pin_fixed_y_np, nw_np, pos, affected_nets,
    macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, smooth_range,
):
    for k in range(len(affected_nets)):
        n = affected_nets[k]
        _add_net_route_pin_hv_smooth(
            h_grid, v_grid, h_smooth, v_smooth, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
            macro_i, old_x, old_y, bin_w, bin_h, n_rows, n_cols, hcap, vcap, -1.0, smooth_range,
        )
        _add_net_route_pin_hv_smooth(
            h_grid, v_grid, h_smooth, v_smooth, n,
            pin_owner_np, pin_mask_np, pin_xoff_np, pin_yoff_np,
            pin_fixed_x_np, pin_fixed_y_np, nw_np, pos,
            -1, -1.0, -1.0, bin_w, bin_h, n_rows, n_cols, hcap, vcap, 1.0, smooth_range,
        )


def _grid_cell_from_xy(x: float, y: float, bin_w: float, bin_h: float, n_rows: int, n_cols: int):
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


def _add_h_segment_tracked(
    h_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    row: int,
    c0: int,
    c1: int,
    weight: float,
    hcap: float,
    sign: float,
    n_rows: int,
    n_cols: int,
) -> None:
    if row < 0:
        row = 0
    elif row >= n_rows:
        row = n_rows - 1
    lo = c0
    hi = c1
    if lo > hi:
        lo, hi = hi, lo
    if lo < 0:
        lo = 0
    if hi > n_cols:
        hi = n_cols
    delta = sign * weight / hcap
    for c in range(lo, hi):
        old = float(h_grid[row, c])
        h_grid[row, c] += np.float32(delta)
        actual = float(h_grid[row, c]) - old
        tracker.apply_h_route_delta(row, c, actual)


def _add_v_segment_tracked(
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    col: int,
    r0: int,
    r1: int,
    weight: float,
    vcap: float,
    sign: float,
    n_rows: int,
    n_cols: int,
) -> None:
    if col < 0:
        col = 0
    elif col >= n_cols:
        col = n_cols - 1
    lo = r0
    hi = r1
    if lo > hi:
        lo, hi = hi, lo
    if lo < 0:
        lo = 0
    if hi > n_rows:
        hi = n_rows
    delta = sign * weight / vcap
    for r in range(lo, hi):
        old = float(v_grid[r, col])
        v_grid[r, col] += np.float32(delta)
        actual = float(v_grid[r, col]) - old
        tracker.apply_v_route_delta(r, col, actual)


def _add_two_pin_route_hv_tracked(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    sr: int,
    sc: int,
    tr: int,
    tc: int,
    weight: float,
    hcap: float,
    vcap: float,
    sign: float,
    n_rows: int,
    n_cols: int,
) -> None:
    _add_h_segment_tracked(h_grid, tracker, sr, min(sc, tc), max(sc, tc), weight, hcap, sign, n_rows, n_cols)
    _add_v_segment_tracked(v_grid, tracker, tc, min(sr, tr), max(sr, tr), weight, vcap, sign, n_rows, n_cols)


def _sort3_by_col_row_py(r0, c0, r1, c1, r2, c2):
    pts = sorted(((r0, c0), (r1, c1), (r2, c2)), key=lambda p: (p[1], p[0]))
    return pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]


def _sort3_by_row_col_py(r0, c0, r1, c1, r2, c2):
    pts = sorted(((r0, c0), (r1, c1), (r2, c2)), key=lambda p: (p[0], p[1]))
    return pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1]


def _add_three_pin_route_hv_tracked(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    rows: list[int],
    cols: list[int],
    weight: float,
    hcap: float,
    vcap: float,
    sign: float,
    n_rows: int,
    n_cols: int,
) -> None:
    r1, c1, r2, c2, r3, c3 = _sort3_by_col_row_py(rows[0], cols[0], rows[1], cols[1], rows[2], cols[2])
    if c1 < c2 and c2 < c3 and min(r1, r3) < r2 and max(r1, r3) > r2:
        _add_h_segment_tracked(h_grid, tracker, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_h_segment_tracked(h_grid, tracker, r2, c2, c3, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_tracked(v_grid, tracker, c2, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols)
        _add_v_segment_tracked(v_grid, tracker, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols)
    elif c2 == c3 and c1 < c2 and r1 < min(r2, r3):
        _add_h_segment_tracked(h_grid, tracker, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_tracked(v_grid, tracker, c2, r1, max(r2, r3), weight, vcap, sign, n_rows, n_cols)
    elif r2 == r3:
        _add_h_segment_tracked(h_grid, tracker, r1, c1, c2, weight, hcap, sign, n_rows, n_cols)
        _add_h_segment_tracked(h_grid, tracker, r2, c2, c3, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_tracked(v_grid, tracker, c2, min(r2, r1), max(r2, r1), weight, vcap, sign, n_rows, n_cols)
    else:
        r1, c1, r2, c2, r3, c3 = _sort3_by_row_col_py(rows[0], cols[0], rows[1], cols[1], rows[2], cols[2])
        xmin = min(c1, c2, c3)
        xmax = max(c1, c2, c3)
        _add_h_segment_tracked(h_grid, tracker, r2, xmin, xmax, weight, hcap, sign, n_rows, n_cols)
        _add_v_segment_tracked(v_grid, tracker, c1, min(r1, r2), max(r1, r2), weight, vcap, sign, n_rows, n_cols)
        _add_v_segment_tracked(v_grid, tracker, c3, min(r2, r3), max(r2, r3), weight, vcap, sign, n_rows, n_cols)


def _append_unique(rows: list[int], cols: list[int], r: int, c: int) -> None:
    for rr, cc in zip(rows, cols):
        if rr == r and cc == c:
            return
    rows.append(r)
    cols.append(c)


def _route_from_points(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    rows: list[int],
    cols: list[int],
    src_r: int,
    src_c: int,
    weight: float,
    hcap: float,
    vcap: float,
    sign: float,
    n_rows: int,
    n_cols: int,
) -> None:
    count = len(rows)
    if count == 2:
        if rows[0] == src_r and cols[0] == src_c:
            tr, tc = rows[1], cols[1]
        else:
            tr, tc = rows[0], cols[0]
        _add_two_pin_route_hv_tracked(h_grid, v_grid, tracker, src_r, src_c, tr, tc, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count == 3:
        _add_three_pin_route_hv_tracked(h_grid, v_grid, tracker, rows, cols, weight, hcap, vcap, sign, n_rows, n_cols)
    elif count > 3:
        for r, c in zip(rows, cols):
            if r == src_r and c == src_c:
                continue
            _add_two_pin_route_hv_tracked(h_grid, v_grid, tracker, src_r, src_c, r, c, weight, hcap, vcap, sign, n_rows, n_cols)


def _add_net_route_hv_tracked(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    net_idx: int,
    ni_np: np.ndarray,
    nm_np: np.ndarray,
    nw_np: np.ndarray,
    pos: np.ndarray,
    macro_i: int,
    override_x: float,
    override_y: float,
    bin_w: float,
    bin_h: float,
    n_rows: int,
    n_cols: int,
    hcap: float,
    vcap: float,
    sign: float,
) -> None:
    rows: list[int] = []
    cols: list[int] = []
    src_r = 0
    src_c = 0
    source_seen = False
    weight = float(nw_np[net_idx])
    for d in range(ni_np.shape[1]):
        if not nm_np[net_idx, d]:
            break
        idx = int(ni_np[net_idx, d])
        if idx == macro_i and override_x >= 0.0:
            x, y = override_x, override_y
        else:
            x, y = float(pos[idx, 0]), float(pos[idx, 1])
        r, c = _grid_cell_from_xy(x, y, bin_w, bin_h, n_rows, n_cols)
        if not source_seen:
            src_r, src_c = r, c
            source_seen = True
        _append_unique(rows, cols, r, c)
    _route_from_points(h_grid, v_grid, tracker, rows, cols, src_r, src_c, weight, hcap, vcap, sign, n_rows, n_cols)


def _add_net_route_pin_hv_tracked(
    h_grid: np.ndarray,
    v_grid: np.ndarray,
    tracker: SmoothHVTopKTracker,
    net_idx: int,
    pin_owner_np: np.ndarray,
    pin_mask_np: np.ndarray,
    pin_xoff_np: np.ndarray,
    pin_yoff_np: np.ndarray,
    pin_fixed_x_np: np.ndarray,
    pin_fixed_y_np: np.ndarray,
    nw_np: np.ndarray,
    pos: np.ndarray,
    macro_i: int,
    override_x: float,
    override_y: float,
    bin_w: float,
    bin_h: float,
    n_rows: int,
    n_cols: int,
    hcap: float,
    vcap: float,
    sign: float,
) -> None:
    rows: list[int] = []
    cols: list[int] = []
    src_r = 0
    src_c = 0
    source_seen = False
    weight = float(nw_np[net_idx])
    for d in range(pin_owner_np.shape[1]):
        if not pin_mask_np[net_idx, d]:
            break
        owner = int(pin_owner_np[net_idx, d])
        if owner >= 0:
            if owner == macro_i and override_x >= 0.0:
                x = override_x + float(pin_xoff_np[net_idx, d])
                y = override_y + float(pin_yoff_np[net_idx, d])
            else:
                x = float(pos[owner, 0]) + float(pin_xoff_np[net_idx, d])
                y = float(pos[owner, 1]) + float(pin_yoff_np[net_idx, d])
        else:
            x = float(pin_fixed_x_np[net_idx, d])
            y = float(pin_fixed_y_np[net_idx, d])
        r, c = _grid_cell_from_xy(x, y, bin_w, bin_h, n_rows, n_cols)
        if not source_seen:
            src_r, src_c = r, c
            source_seen = True
        _append_unique(rows, cols, r, c)
    _route_from_points(h_grid, v_grid, tracker, rows, cols, src_r, src_c, weight, hcap, vcap, sign, n_rows, n_cols)
