"""
Macro placement via basin hopping + projected gradient descent
Uses TILOS as acceptance oracle — no approximation errors
Adds differentiable density spreading + adaptive loss weights

Usage:
    uv run python gradient_place.py                    # ibm01 only
    uv run python gradient_place.py --benchmark ibm04  # specific benchmark
    uv run python gradient_place.py --all               # all 17 IBM benchmarks
"""

import argparse
import time
import torch
import sys

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

from density import DifferentiableDensity
from basin_hopper import BasinHopper

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED FUNCTIONS (used by both standalone mode and basin hopper)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_nets(plc, benchmark):
    """Extract net connectivity as list of macro index lists."""
    name_to_idx = {}
    for tensor_idx, plc_idx in enumerate(benchmark.hard_macro_indices):
        mod = plc.modules_w_pins[plc_idx]
        name_to_idx[mod.get_name()] = tensor_idx
    for tensor_idx, plc_idx in enumerate(benchmark.soft_macro_indices):
        mod = plc.modules_w_pins[plc_idx]
        name_to_idx[mod.get_name()] = benchmark.num_hard_macros + tensor_idx

    seen = set()
    nets = []
    for driver, sinks in plc.nets.items():
        members = set()
        driver_macro = driver.split('/')[0]
        if driver_macro in name_to_idx:
            members.add(name_to_idx[driver_macro])
        for sink in sinks:
            sink_macro = sink.split('/')[0]
            if sink_macro in name_to_idx:
                members.add(name_to_idx[sink_macro])
        if len(members) >= 2:
            key = frozenset(members)
            if key not in seen:
                seen.add(key)
                nets.append(list(members))

    return nets


def precompute_net_tensors(nets):
    """
    Precompute padded index tensor and mask for vectorized wirelength.
    Call ONCE per benchmark after extract_nets(), pass results to wirelength fn.
    """
    max_degree = max(len(n) for n in nets)
    num_nets = len(nets)
    net_indices = torch.zeros(num_nets, max_degree, dtype=torch.long)
    net_mask = torch.zeros(num_nets, max_degree, dtype=torch.bool)
    for i, net in enumerate(nets):
        net_indices[i, :len(net)] = torch.tensor(net)
        net_mask[i, :len(net)] = True
    return net_indices, net_mask


def differentiable_wirelength(placement, nets, benchmark, alpha=10.0,
                               net_indices=None, net_mask=None):
    """
    Vectorized log-sum-exp HPWL — all nets in one shot, no Python loop.
    
    If net_indices/net_mask are provided (from precompute_net_tensors),
    uses fast vectorized path. Otherwise falls back to slow loop.
    """
    if net_indices is None or net_mask is None:
        # Slow fallback (shouldn't happen in normal use)
        total = torch.zeros(1, dtype=torch.float32)
        for net in nets:
            if len(net) < 2:
                continue
            idx = torch.tensor(net)
            pos = placement[idx]
            x, y = pos[:, 0], pos[:, 1]
            x_max = (1/alpha) * torch.logsumexp(alpha * x, dim=0)
            x_min = -(1/alpha) * torch.logsumexp(-alpha * x, dim=0)
            y_max = (1/alpha) * torch.logsumexp(alpha * y, dim=0)
            y_min = -(1/alpha) * torch.logsumexp(-alpha * y, dim=0)
            total = total + (x_max - x_min) + (y_max - y_min)
        return total / (len(nets) * (benchmark.canvas_width + benchmark.canvas_height))

    # ── Fast vectorized path ──
    # Gather positions for all nets at once: (num_nets, max_degree, 2)
    pos = placement[net_indices]
    x = pos[:, :, 0]
    y = pos[:, :, 1]

    # Mask padded entries so they don't affect logsumexp
    x_for_max = x.masked_fill(~net_mask, float('-inf'))
    x_for_min = x.masked_fill(~net_mask, float('inf'))
    y_for_max = y.masked_fill(~net_mask, float('-inf'))
    y_for_min = y.masked_fill(~net_mask, float('inf'))

    # Smooth max/min across all nets simultaneously
    x_max = (1/alpha) * torch.logsumexp(alpha * x_for_max, dim=1)
    x_min = -(1/alpha) * torch.logsumexp(-alpha * x_for_min, dim=1)
    y_max = (1/alpha) * torch.logsumexp(alpha * y_for_max, dim=1)
    y_min = -(1/alpha) * torch.logsumexp(-alpha * y_for_min, dim=1)

    hpwl = (x_max - x_min) + (y_max - y_min)
    num_nets = len(nets)
    return hpwl.sum() / (num_nets * (benchmark.canvas_width + benchmark.canvas_height))


def differentiable_overlap_penalty(placement, benchmark):
    """Pairwise overlap area, normalized by canvas area."""
    num_hard = benchmark.num_hard_macros
    pos = placement[:num_hard]
    sizes = benchmark.macro_sizes[:num_hard].detach()

    pos_i = pos.unsqueeze(1)
    pos_j = pos.unsqueeze(0)
    sizes_i = sizes.unsqueeze(1)
    sizes_j = sizes.unsqueeze(0)

    dx = torch.abs(pos_i[:,:,0] - pos_j[:,:,0])
    dy = torch.abs(pos_i[:,:,1] - pos_j[:,:,1])
    min_sep_x = (sizes_i[:,:,0] + sizes_j[:,:,0]) / 2.0
    min_sep_y = (sizes_i[:,:,1] + sizes_j[:,:,1]) / 2.0

    overlap_x = torch.clamp(min_sep_x - dx, min=0.0)
    overlap_y = torch.clamp(min_sep_y - dy, min=0.0)
    overlap = overlap_x * overlap_y
    mask = 1 - torch.eye(num_hard)
    overlap = overlap * mask

    canvas_area = benchmark.canvas_width * benchmark.canvas_height
    return overlap.sum() / canvas_area


def legalize(placement, benchmark, gap=0.001, max_iters=200):
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


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(benchmark_name, time_budget=240, steps_per_basin=200, lr=0.002):
    """
    Run the full placer on one benchmark.
    Returns (best_placement, best_costs, elapsed_time).
    """
    start = time.time()

    # ── Load ──
    benchmark, plc = load_benchmark_from_dir(
        f'external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}'
    )
    print(f"\n{'='*60}")
    print(f"Benchmark:   {benchmark.name}")
    print(f"Canvas:      {benchmark.canvas_width:.0f} x {benchmark.canvas_height:.0f}")
    print(f"Hard macros: {benchmark.num_hard_macros}")
    print(f"Soft macros: {benchmark.num_soft_macros}")
    print(f"Grid:        {benchmark.grid_rows} x {benchmark.grid_cols}")
    print(f"{'='*60}")

    # ── Extract nets + precompute tensors ──
    nets = extract_nets(plc, benchmark)
    net_indices, net_mask = precompute_net_tensors(nets)
    print(f"Nets: {len(nets)}")

    # ── Baseline (initial placement as-is) ──
    costs_init = compute_proxy_cost(benchmark.macro_positions.clone(), benchmark, plc)
    print(f"\nBaseline proxy cost: {costs_init['proxy_cost']:.4f} "
          f"(WL={costs_init['wirelength_cost']:.4f} "
          f"den={costs_init['density_cost']:.4f} "
          f"cong={costs_init['congestion_cost']:.4f} "
          f"overlaps={costs_init['overlap_count']})")

    # ── Wirelength wrapper with precomputed tensors baked in ──
    def wl_fn(placement, nets, benchmark, alpha=10.0):
        return differentiable_wirelength(
            placement, nets, benchmark, alpha=alpha,
            net_indices=net_indices, net_mask=net_mask,
        )

    # ── Set up density function ──
    # overflow_weight=1.0 here because BasinHopper's AdaptiveLossWeights
    # manages the actual weight — don't double-count
    density_fn = DifferentiableDensity(benchmark, overflow_weight=1.0)

    # ── Set up basin hopper ──
    hopper = BasinHopper(
        benchmark=benchmark,
        plc=plc,
        nets=nets,
        density_fn=density_fn,
        legalize_fn=legalize,
        compute_cost_fn=compute_proxy_cost,
        differentiable_wl_fn=wl_fn,
        overlap_fn=differentiable_overlap_penalty,
    )

    # ── Run ──
    best_placement, best_costs = hopper.run(
        time_budget=time_budget,
        steps_per_basin=steps_per_basin,
        lr=lr,
    )

    # ── Validate ──
    is_valid, violations = validate_placement(best_placement, benchmark)
    if not is_valid:
        print(f"\n⚠️  VALIDATION FAILED:")
        for v in violations:
            print(f"  - {v}")
    else:
        print(f"\n✓ Placement valid (zero overlaps, all in bounds)")

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")

    return best_placement, best_costs, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

# RePlAce baseline per benchmark (from competition data, if available)
# Fill these in as you get numbers — used for comparison only
REPLACE_BASELINE = {
    "ibm01": 0.9976,
    # add more as you collect them
}

SA_BASELINE = {
    "ibm01": 1.3166,
    # add more
}


def main():
    parser = argparse.ArgumentParser(description="TendrIL Macro Placer")
    parser.add_argument('--benchmark', '-b', type=str, default='ibm01',
                        help='Benchmark name (e.g., ibm01)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 17 IBM benchmarks')
    parser.add_argument('--time-budget', type=int, default=240,
                        help='Time budget per benchmark in seconds (default: 240)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Gradient steps per basin (default: 200)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate (default: 0.002)')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results')
    args = parser.parse_args()

    if args.all:
        benchmarks_to_run = BENCHMARKS
    else:
        benchmarks_to_run = [args.benchmark]

    # ── Run all benchmarks ──
    results = {}
    for name in benchmarks_to_run:
        try:
            placement, costs, elapsed = run_benchmark(
                name,
                time_budget=args.time_budget,
                steps_per_basin=args.steps,
                lr=args.lr,
            )
            results[name] = {
                'proxy_cost': costs['proxy_cost'],
                'wirelength': costs['wirelength_cost'],
                'density': costs['density_cost'],
                'congestion': costs['congestion_cost'],
                'overlaps': costs['overlap_count'],
                'time': elapsed,
            }

            # Save placement tensor
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(placement, f"{args.save_dir}/{name}_placement.pt")

        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'proxy_cost': float('inf'), 'error': str(e)}

    # ── Summary table ──
    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print(f"{'SUMMARY':^80}")
        print(f"{'='*80}")
        print(f"{'Benchmark':<10} {'Proxy':>8} {'WL':>8} {'Den':>8} "
              f"{'Cong':>8} {'Overlaps':>8} {'Time':>8} {'vs RePlAce':>12}")
        print(f"{'-'*80}")

        total_proxy = 0
        count = 0
        for name in benchmarks_to_run:
            r = results.get(name, {})
            proxy = r.get('proxy_cost', float('inf'))
            if proxy == float('inf'):
                print(f"{name:<10} {'FAILED':>8}")
                continue

            replace_ref = REPLACE_BASELINE.get(name)
            vs_replace = ''
            if replace_ref:
                ratio = proxy / replace_ref
                vs_replace = f"{ratio:.3f}x"

            print(f"{name:<10} {proxy:>8.4f} {r['wirelength']:>8.4f} "
                  f"{r['density']:>8.4f} {r['congestion']:>8.4f} "
                  f"{r['overlaps']:>8d} {r['time']:>7.1f}s {vs_replace:>12}")

            total_proxy += proxy
            count += 1

        if count > 0:
            avg = total_proxy / count
            print(f"{'-'*80}")
            print(f"{'AVERAGE':<10} {avg:>8.4f}")
            print(f"\nRePlAce baseline avg: 1.4578")
            print(f"SA baseline avg:      2.1251")
            print(f"Your avg:             {avg:.4f}")


if __name__ == '__main__':
    main()