"""
Hybrid analytical placer built from the faster grad_place_benches flow.

Design goals:
1. Keep the staged optimization loop that already runs fast.
2. Stay CPU-first and submission-friendly.
3. Print exact progress stats every 1/10 of the run without spamming.
"""

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from scipy.fft import dctn, idctn

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import _set_placement, compute_proxy_cost
# from gradient_place import legalize as strong_legalize

def strong_legalize(placement, benchmark, gap=0.001, max_iters=200):
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

        push_amount_x = torch.where(overlap & push_x_axis, push_amount_x, torch.zeros_like(push_amount_x))
        push_amount_y = torch.where(overlap & ~push_x_axis, push_amount_y, torch.zeros_like(push_amount_y))
        sign_x = torch.where(overlap & push_x_axis, sign_x, torch.zeros_like(sign_x))
        sign_y = torch.where(overlap & ~push_x_axis, sign_y, torch.zeros_like(sign_y))

        force_x = (sign_x * push_amount_x).sum(dim=1) - (sign_x * push_amount_x).sum(dim=0)
        force_y = (sign_y * push_amount_y).sum(dim=1) - (sign_y * push_amount_y).sum(dim=0)

        num_overlaps = overlap.sum().float()
        damping = min(1.0, 2.0 / (num_overlaps.item() ** 0.5 + 1))
        placement[:num_hard, 0] += force_x * damping
        placement[:num_hard, 1] += force_y * damping

        placement[:num_hard, 0].clamp_(min=half_w, max=benchmark.canvas_width - half_w)
        placement[:num_hard, 1].clamp_(min=half_h, max=benchmark.canvas_height - half_h)

    # ── Phase 2: Sequential cleanup (guaranteed resolution) ──
    for iteration in range(max_iters):
        moved = False
        for i in range(num_hard):
            for j in range(i+1, num_hard):
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


class HybridAnalyticalPlacer:
    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 800,
        lr: float = 1.0,
        momentum: float = 0.9,
        warmup_frac: float = 0.10,
        density_phase_start: float = 0.45,
        soft_macro_lr: float = 0.15,
        rudy_weight: float = 0.005,
        verbose: bool = True,
    ):
        self.seed = seed
        self.num_steps = num_steps
        self.lr = lr
        self.momentum = momentum
        self.warmup_frac = warmup_frac
        self.density_phase_start = density_phase_start
        self.soft_macro_lr = soft_macro_lr
        self.rudy_weight = rudy_weight
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

        net_indices, net_mask = self._precompute_net_tensors(nets)
        placement = self._make_initial_placement(placement, benchmark, nets, net_indices, net_mask)

        hard_sizes = benchmark.macro_sizes[:num_hard]
        all_sizes = benchmark.macro_sizes[:num_all]
        hw_hard = hard_sizes[:, 0] / 2
        hh_hard = hard_sizes[:, 1] / 2
        hw_all = all_sizes[:, 0] / 2
        hh_all = all_sizes[:, 1] / 2
        fixed = benchmark.macro_fixed

        velocity = torch.zeros_like(placement)
        density_weight = 0.0001
        best_proxy = float("inf")
        best_placement = placement.clone()
        canvas_norm = benchmark.canvas_width + benchmark.canvas_height
        warmup_steps = int(self.num_steps * self.warmup_frac)
        density_phase_step = int(self.num_steps * self.density_phase_start)
        log_every = max(1, self.num_steps // 10)
        plc_synced_at = -1

        def sync_plc(step: int) -> None:
            nonlocal plc_synced_at
            if plc is None:
                return
            if plc_synced_at != step:
                _set_placement(plc, placement.detach(), benchmark)
                plc.FLAG_UPDATE_WIRELENGTH = False
                plc_synced_at = step

        self._log_stats("start", benchmark, placement, plc, wl=None, density_weight=density_weight)

        for step in range(self.num_steps):
            in_warmup = step < warmup_steps
            in_density_phase = step >= density_phase_step

            if step == density_phase_step:
                placement = self._legalize_fast(placement, benchmark, gap=0.01, max_iters=200)
                velocity.zero_()

            placement.requires_grad_(True)
            loss, wl = self._compute_wl_loss(
                placement, net_indices, net_mask, nets, canvas_norm, self.rudy_weight
            )
            loss.backward()
            wl_grad = placement.grad.detach().clone()
            placement.requires_grad_(False)

            if in_density_phase:
                grid = self._compute_density_grid_fast(placement, benchmark)
                density_forces = self._compute_density_force_fast(
                    self._solve_poisson(grid), placement, benchmark
                )

            if not in_warmup:
                hard_grad = wl_grad[:num_hard].clone()
                if in_density_phase:
                    hard_grad -= density_weight * density_forces
                if fixed.any():
                    hard_grad[fixed[:num_hard]] = 0.0

                velocity[:num_hard] = self.momentum * velocity[:num_hard] - self.lr * hard_grad
                placement[:num_hard] = (placement[:num_hard] + velocity[:num_hard]).clamp(
                    min=torch.stack([hw_hard, hh_hard], dim=1),
                    max=torch.stack(
                        [benchmark.canvas_width - hw_hard, benchmark.canvas_height - hh_hard], dim=1
                    ),
                )

            effective_soft_lr = 0.30 if in_warmup else self.soft_macro_lr
            placement[num_hard:num_all] -= effective_soft_lr * wl_grad[num_hard:num_all]
            placement[num_hard:num_all, 0].clamp_(
                min=hw_all[num_hard:], max=benchmark.canvas_width - hw_all[num_hard:]
            )
            placement[num_hard:num_all, 1].clamp_(
                min=hh_all[num_hard:], max=benchmark.canvas_height - hh_all[num_hard:]
            )

            if fixed.any():
                placement[fixed] = benchmark.macro_positions[fixed]

            if in_density_phase and step % 20 == 0 and plc is not None:
                sync_plc(step)
                tilos_den = plc.get_density_cost()
                if tilos_den > 0.85:
                    density_weight = min(0.005, density_weight * 1.05)
                elif tilos_den < 0.75:
                    density_weight = max(0.0001, density_weight * 0.95)

            if step == self.num_steps - 1 or (step + 1) % log_every == 0:
                if plc is not None:
                    sync_plc(step)
                proxy_est = (
                    wl.item() + 0.5 * plc.get_density_cost() + 0.5 * plc.get_congestion_cost()
                    if plc is not None
                    else float("nan")
                )
                if plc is not None and proxy_est < best_proxy:
                    best_proxy = proxy_est
                    best_placement = placement.detach().clone()
                self._log_stats(
                    f"iter {step + 1}/{self.num_steps}",
                    benchmark,
                    placement,
                    plc,
                    wl=wl.item(),
                    density_weight=density_weight,
                )

            placement = placement.detach()

        if best_proxy == float("inf"):
            best_placement = placement.clone()

        final = best_placement.clone()
        prev_overlaps = self._hard_overlap_count(final, benchmark)
        for _ in range(8):
            if prev_overlaps == 0:
                break
            final = self._legalize_fast(final, benchmark, gap=0.01, max_iters=200)
            new_overlaps = self._hard_overlap_count(final, benchmark)
            if new_overlaps == 0:
                break
            if new_overlaps >= prev_overlaps:
                break
            prev_overlaps = new_overlaps
        if self._hard_overlap_count(final, benchmark) > 0:
            final = strong_legalize(final, benchmark, gap=0.01, max_iters=40)
        final = self._optimize_soft_macros_gentle(final, nets, benchmark, alpha=0.05)
        self._log_stats("final", benchmark, final, plc, wl=None, density_weight=density_weight)
        return final

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

    def _precompute_net_tensors(self, nets: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_degree = max(len(n) for n in nets)
        net_indices = torch.zeros(len(nets), max_degree, dtype=torch.long)
        net_mask = torch.zeros(len(nets), max_degree, dtype=torch.bool)
        for i, net in enumerate(nets):
            net_indices[i, : len(net)] = torch.tensor(net, dtype=torch.long)
            net_mask[i, : len(net)] = True
        return net_indices, net_mask

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
            order = sorted(movable_indices, key=lambda i: -float((sizes[i, 0] * sizes[i, 1]).item()))
            for count, idx in enumerate(order):
                row = count // grid_cols
                col = count % grid_cols
                target_x = spacing_x * (col + 1)
                target_y = spacing_y * (row + 1)
                placement[idx, 0] = 0.7 * placement[idx, 0] + 0.3 * target_x
                placement[idx, 1] = 0.7 * placement[idx, 1] + 0.3 * target_y

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
        rudy_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_net = placement[net_indices]
        x = pos_net[:, :, 0]
        y = pos_net[:, :, 1]
        alpha = 10.0

        x_max = (1 / alpha) * torch.logsumexp(alpha * x.masked_fill(~net_mask, float("-inf")), dim=1)
        x_min = (-1 / alpha) * torch.logsumexp(-alpha * x.masked_fill(~net_mask, float("inf")), dim=1)
        y_max = (1 / alpha) * torch.logsumexp(alpha * y.masked_fill(~net_mask, float("-inf")), dim=1)
        y_min = (-1 / alpha) * torch.logsumexp(-alpha * y.masked_fill(~net_mask, float("inf")), dim=1)

        x_span = x_max - x_min
        y_span = y_max - y_min
        wl = (x_span + y_span).sum() / (len(nets) * canvas_norm)

        if rudy_weight > 0.0:
            bbox_area = x_span * y_span + 1e-6
            rudy_loss = ((x_span + y_span) / bbox_area).sum() / len(nets)
            return wl + rudy_weight * rudy_loss, wl
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
            loss, _ = self._compute_wl_loss(placement, net_indices, net_mask, nets, canvas_norm)
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

    def _compute_density_grid_fast(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        num_macros = benchmark.num_macros
        sizes = benchmark.macro_sizes[:num_macros]
        grid_rows = benchmark.grid_rows
        grid_cols = benchmark.grid_cols
        bin_w = benchmark.canvas_width / grid_cols
        bin_h = benchmark.canvas_height / grid_rows

        cx = placement[:num_macros, 0]
        cy = placement[:num_macros, 1]
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        left = cx - half_w
        right = cx + half_w
        bottom = cy - half_h
        top = cy + half_h

        bin_left = torch.arange(grid_cols, dtype=placement.dtype) * bin_w
        bin_right = bin_left + bin_w
        bin_bottom = torch.arange(grid_rows, dtype=placement.dtype) * bin_h
        bin_top = bin_bottom + bin_h

        overlap_x = torch.clamp(
            torch.min(right.unsqueeze(1), bin_right.unsqueeze(0))
            - torch.max(left.unsqueeze(1), bin_left.unsqueeze(0)),
            min=0,
        )
        overlap_y = torch.clamp(
            torch.min(top.unsqueeze(1), bin_top.unsqueeze(0))
            - torch.max(bottom.unsqueeze(1), bin_bottom.unsqueeze(0)),
            min=0,
        )
        return torch.mm(overlap_y.t(), overlap_x) / (bin_w * bin_h)

    def _solve_poisson(self, density_grid: torch.Tensor) -> torch.Tensor:
        rows, cols = density_grid.shape
        rho = density_grid - density_grid.mean()
        rho_freq = torch.tensor(dctn(rho.numpy()), dtype=torch.float32)
        j = torch.arange(rows, dtype=torch.float32)
        k = torch.arange(cols, dtype=torch.float32)
        eig_j = 2 * torch.cos(torch.pi * j / rows)
        eig_k = 2 * torch.cos(torch.pi * k / cols)
        eigenvalues = eig_j.unsqueeze(1) + eig_k.unsqueeze(0) - 4
        eigenvalues[0, 0] = 1
        potential = torch.tensor(idctn((rho_freq / eigenvalues).numpy()), dtype=torch.float32)
        return -potential

    def _compute_density_force_fast(
        self, potential: torch.Tensor, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        bin_w = benchmark.canvas_width / benchmark.grid_cols
        bin_h = benchmark.canvas_height / benchmark.grid_rows

        grad_x = torch.zeros_like(potential)
        grad_y = torch.zeros_like(potential)
        grad_x[:, 1:-1] = (potential[:, 2:] - potential[:, :-2]) / (2 * bin_w)
        grad_y[1:-1, :] = (potential[2:, :] - potential[:-2, :]) / (2 * bin_h)

        cx = placement[:num_hard, 0].detach()
        cy = placement[:num_hard, 1].detach()
        c_bins = (cx / bin_w).long().clamp(0, benchmark.grid_cols - 1)
        r_bins = (cy / bin_h).long().clamp(0, benchmark.grid_rows - 1)

        forces = torch.zeros(num_hard, 2)
        forces[:, 0] = -grad_x[r_bins, c_bins]
        forces[:, 1] = -grad_y[r_bins, c_bins]
        return forces

    def _legalize_fast(
        self, placement: torch.Tensor, benchmark: Benchmark, gap: float = 0.01, max_iters: int = 500
    ) -> torch.Tensor:
        placement = placement.clone()
        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:num_hard]
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0].unsqueeze(1) + sizes[:, 0].unsqueeze(0)) / 2 + gap
        sep_y = (sizes[:, 1].unsqueeze(1) + sizes[:, 1].unsqueeze(0)) / 2 + gap
        tri = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)

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

    def _optimize_soft_macros_gentle(
        self, placement: torch.Tensor, nets: List[List[int]], benchmark: Benchmark, alpha: float = 0.3
    ) -> torch.Tensor:
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        for i in range(num_soft):
            soft_idx = num_hard + i
            connected = set()
            for net in nets:
                if soft_idx in net:
                    connected.update(net)
            connected.discard(soft_idx)
            if not connected:
                continue
            target_x = sum(float(placement[j, 0].item()) for j in connected) / len(connected)
            target_y = sum(float(placement[j, 1].item()) for j in connected) / len(connected)
            old_x = float(placement[soft_idx, 0].item())
            old_y = float(placement[soft_idx, 1].item())
            new_x = old_x + alpha * (target_x - old_x)
            new_y = old_y + alpha * (target_y - old_y)
            w = benchmark.macro_sizes[soft_idx, 0].item()
            h = benchmark.macro_sizes[soft_idx, 1].item()
            placement[soft_idx, 0] = max(w / 2, min(benchmark.canvas_width - w / 2, new_x))
            placement[soft_idx, 1] = max(h / 2, min(benchmark.canvas_height - h / 2, new_y))
        return placement

    def _hard_overlap_count(self, placement: torch.Tensor, benchmark: Benchmark) -> int:
        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return 0
        sizes = benchmark.macro_sizes[:num_hard]
        p = placement[:num_hard]
        dx = torch.abs(p.unsqueeze(0)[:, :, 0] - p.unsqueeze(1)[:, :, 0])
        dy = torch.abs(p.unsqueeze(0)[:, :, 1] - p.unsqueeze(1)[:, :, 1])
        sx = (sizes[:, 0].unsqueeze(0) + sizes[:, 0].unsqueeze(1)) / 2
        sy = (sizes[:, 1].unsqueeze(0) + sizes[:, 1].unsqueeze(1)) / 2
        tri = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
        return int(((dx < sx) & (dy < sy) & tri).sum().item())

    def _log_stats(
        self,
        label: str,
        benchmark: Benchmark,
        placement: torch.Tensor,
        plc,
        wl: Optional[float],
        density_weight: float,
    ) -> None:
        if not self.verbose:
            return
        overlaps = self._hard_overlap_count(placement, benchmark)
        if plc is not None:
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
