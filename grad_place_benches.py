import random
import time
import sys
from multiprocessing import Pool
import torch
from scipy.fft import dctn, idctn
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from gradient_place import (
    extract_nets,
    legalize,
    precompute_net_tensors,
    differentiable_wirelength,
    differentiable_overlap_penalty,
)

benchmark, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")


def extract_nets(plc, benchmark):
    name_to_idx = {}
    for ti, pi in enumerate(benchmark.hard_macro_indices):
        name_to_idx[plc.modules_w_pins[pi].get_name()] = ti
    for ti, pi in enumerate(benchmark.soft_macro_indices):
        name_to_idx[plc.modules_w_pins[pi].get_name()] = benchmark.num_hard_macros + ti
    seen, nets = set(), []
    for driver, sinks in plc.nets.items():
        members = set()
        dm = driver.split("/")[0]
        if dm in name_to_idx:
            members.add(name_to_idx[dm])
        for s in sinks:
            sm = s.split("/")[0]
            if sm in name_to_idx:
                members.add(name_to_idx[sm])
        if len(members) >= 2:
            key = frozenset(members)
            if key not in seen:
                seen.add(key)
                nets.append(list(members))
    return nets


def compute_density_grid(placement, benchmark):
    num_macros = benchmark.num_hard_macros + benchmark.num_soft_macros
    sizes = benchmark.macro_sizes[:num_macros]
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    bin_w = benchmark.canvas_width / grid_cols
    bin_h = benchmark.canvas_height / grid_rows

    density = torch.zeros(grid_rows, grid_cols)

    for i in range(num_macros):
        cx, cy = placement[i, 0].item(), placement[i, 1].item()
        w, h = sizes[i, 0].item(), sizes[i, 1].item()

        left = cx - w / 2
        right = cx + w / 2
        bottom = cy - h / 2
        top = cy + h / 2

        left_bin = max(0, int(left / bin_w))
        right_bin = min(grid_cols - 1, int(right / bin_w))
        bottom_bin = max(0, int(bottom / bin_h))
        top_bin = min(grid_rows - 1, int(top / bin_h))

        for r in range(bottom_bin, top_bin + 1):
            for c in range(left_bin, right_bin + 1):
                bin_left = c * bin_w
                bin_right = (c + 1) * bin_w
                bin_bottom = r * bin_h
                bin_top = (r + 1) * bin_h

                overlap_w = min(right, bin_right) - max(left, bin_left)
                overlap_h = min(top, bin_top) - max(bottom, bin_bottom)
                overlap_area = max(0, overlap_w) * max(0, overlap_h)

                density[r, c] += overlap_area

    density = density / (bin_w * bin_h)
    return density


def solve_poisson(density_grid, benchmark):
    rows, cols = density_grid.shape

    # Step A: subtract target (uniform) density
    target = density_grid.mean()
    rho = density_grid - target

    # Step B: DCT
    # torch.fft.dctn does the discrete cosine transform
    rho_freq = torch.tensor(dctn(rho.numpy()), dtype=torch.float32)

    # Step C: build eigenvalues
    # These come from the discrete Laplacian
    eigenvalues = torch.zeros_like(rho_freq)
    j = torch.arange(rows, dtype=torch.float32)
    k = torch.arange(cols, dtype=torch.float32)
    eig_j = 2 * torch.cos(torch.pi * j / rows)
    eig_k = 2 * torch.cos(torch.pi * k / cols)
    eigenvalues = eig_j.unsqueeze(1) + eig_k.unsqueeze(0) - 4
    eigenvalues[0, 0] = 1  # avoid division by zero
    rho_freq = rho_freq / eigenvalues
    potential = torch.tensor(idctn(rho_freq.numpy()), dtype=torch.float32)
    return -potential


def compute_density_force(potential, placement, benchmark):
    num_hard = benchmark.num_hard_macros
    bin_w = benchmark.canvas_width / benchmark.grid_cols
    bin_h = benchmark.canvas_height / benchmark.grid_rows

    # Gradient of potential field (finite differences on the grid)
    # dpsi/dx and dpsi/dy
    grad_x = torch.zeros_like(potential)
    grad_y = torch.zeros_like(potential)

    # Central differences (you fill this in)
    grad_x[:, 1:-1] = (potential[:, 2:] - potential[:, :-2]) / (2 * bin_w)
    grad_y[1:-1, :] = (potential[2:, :] - potential[:-2, :]) / (2 * bin_h)

    # Then for each macro, look up the force at its grid location
    # force[i] = -grad at macro i's bin
    forces = torch.zeros(num_hard, 2)
    for i in range(num_hard):
        cx, cy = placement[i, 0].item(), placement[i, 1].item()
        c_bin = min(benchmark.grid_cols - 1, max(0, int(cx / bin_w)))
        r_bin = min(benchmark.grid_rows - 1, max(0, int(cy / bin_h)))
        forces[i, 0] = -grad_x[r_bin, c_bin].item()
        forces[i, 1] = -grad_y[r_bin, c_bin].item()
    return forces


def optimize_soft_macros(placement, nets, benchmark):
    """Move each soft macro to the centroid of its connected macros."""
    num_hard = benchmark.num_hard_macros
    num_soft = benchmark.num_soft_macros

    for i in range(num_soft):
        soft_idx = num_hard + i
        # Find all nets this soft macro belongs to
        connected = set()
        for net in nets:
            if soft_idx in net:
                connected.update(net)
        connected.discard(soft_idx)

        if len(connected) == 0:
            continue

        # Move to centroid of connected macros
        cx = sum(placement[j, 0].item() for j in connected) / len(connected)
        cy = sum(placement[j, 1].item() for j in connected) / len(connected)

        # Clamp to canvas
        w = benchmark.macro_sizes[soft_idx, 0].item()
        h = benchmark.macro_sizes[soft_idx, 1].item()
        cx = max(w / 2, min(benchmark.canvas_width - w / 2, cx))
        cy = max(h / 2, min(benchmark.canvas_height - h / 2, cy))

        placement[soft_idx, 0] = cx
        placement[soft_idx, 1] = cy

    return placement

def legalize_fast(placement, benchmark, gap=0.01, max_iters=500):
    placement = placement.clone()
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    
    sep_x = (sizes[:, 0].unsqueeze(1) + sizes[:, 0].unsqueeze(0)) / 2 + gap
    sep_y = (sizes[:, 1].unsqueeze(1) + sizes[:, 1].unsqueeze(0)) / 2 + gap
    tri = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
    
    for iteration in range(max_iters):
        pos = placement[:num_hard]
        dx = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)
        
        overlap_mask = (abs_dx < sep_x) & (abs_dy < sep_y) & tri
        if not overlap_mask.any():
            break
        
        # Process overlapping pairs only
        pairs = overlap_mask.nonzero()
        for p in range(pairs.shape[0]):
            i, j = pairs[p, 0].item(), pairs[p, 1].item()
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

def run_placer(benchmark_name, num_steps=1500, lr=1.0, momentum=0.9, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    benchmark, plc = load_benchmark_from_dir(
        f"external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}"
    )

    nets = extract_nets(plc, benchmark)
    net_indices, net_mask = precompute_net_tensors(nets)
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]

    # Random start
    placement = benchmark.macro_positions.clone()
    for i in range(num_hard):
        if not benchmark.macro_fixed[i]:
            placement[i, 0] = random.uniform(
                sizes[i, 0].item() / 2, benchmark.canvas_width - sizes[i, 0].item() / 2
            )
            placement[i, 1] = random.uniform(
                sizes[i, 1].item() / 2, benchmark.canvas_height - sizes[i, 1].item() / 2
            )

    velocity = torch.zeros_like(placement)
    den_weight = 0.001
    best_proxy = float("inf")
    best_placement = placement.clone()

    start = time.time()

    for step in range(num_steps):
        placement.requires_grad_(True)
        wl = differentiable_wirelength(
            placement, nets, benchmark, net_indices=net_indices, net_mask=net_mask
        )

        # # Add overlap penalty in final phase only
        # if step > num_steps - 300:
        #     overlap = differentiable_overlap_penalty(placement, benchmark)
        #     loss = wl + 1 * overlap
        # else:
        #     loss = wl
        wl.backward()
        wl_grad = placement.grad[:num_hard].detach().clone()
        placement.requires_grad_(False)

        grid = compute_density_grid(placement, benchmark)
        potential = solve_poisson(grid, benchmark)
        density_forces = compute_density_force(potential, placement, benchmark)

        total_grad = wl_grad - den_weight * density_forces
        if benchmark.macro_fixed.any():
            total_grad[benchmark.macro_fixed[:num_hard]] = 0.0

        velocity[:num_hard] = momentum * velocity[:num_hard] - lr * total_grad
        placement[:num_hard] += velocity[:num_hard]

        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        placement[:num_hard, 0].clamp_(min=hw, max=benchmark.canvas_width - hw)
        placement[:num_hard, 1].clamp_(min=hh, max=benchmark.canvas_height - hh)
        if benchmark.macro_fixed.any():
            placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

        if step % 10 == 0 and step > 0:
            grid = compute_density_grid(placement, benchmark)
            overflow = (grid - grid.mean()).clamp(min=0).sum().item()
            if overflow > 0.1:
                den_weight = min(0.5, den_weight * 1.1)

        if step % 200 == 0:
            costs = compute_proxy_cost(placement.detach(), benchmark, plc)
            if costs["proxy_cost"] < best_proxy:
                best_proxy = costs["proxy_cost"]
                best_placement = placement.detach().clone()
            print(
                f"  {benchmark_name} step {step}: pc={costs['proxy_cost']:.4f} "
                f"ovlp={costs['overlap_count']} dw={den_weight:.4f}",
                flush=True,
            )

    # Save final too
    costs = compute_proxy_cost(placement.detach(), benchmark, plc)
    if costs["proxy_cost"] < best_proxy:
        best_proxy = costs["proxy_cost"]
        best_placement = placement.detach().clone()

    # Legalize
    max_leg_iters = 500
    first_legalize_start = time.time()
    print(f"  {benchmark_name} starting legalization with best proxy cost {best_proxy:.4f}")
    legal = legalize_fast(best_placement, benchmark, gap=0.01, max_iters=max_leg_iters)
    first_legalize_end = time.time()
    print(f"  First legalization pass took {first_legalize_end - first_legalize_start:.0f}s")

    # costs_pre_soft = compute_proxy_cost(legal, benchmark, plc)
    # print(f"  After legal, before soft: den={costs_pre_soft['density_cost']:.4f}")
    
    # legal = optimize_soft_macros(legal, nets, benchmark)
    
    # costs_post_soft = compute_proxy_cost(legal, benchmark, plc)
    # print(f"  After soft: den={costs_post_soft['density_cost']:.4f}")    
    costs_final = compute_proxy_cost(legal, benchmark, plc)
    elapsed = time.time() - start

    print(
        f"  {benchmark_name} FINAL: pc={costs_final['proxy_cost']:.4f} "
        f"wl={costs_final['wirelength_cost']:.4f} den={costs_final['density_cost']:.4f} "
        f"cong={costs_final['congestion_cost']:.4f} ovlp={costs_final['overlap_count']} "
        f"time={elapsed:.0f}s"
    )

    return costs_final, legal


BENCHMARKS = [
    "ibm01",
    "ibm02",
    "ibm03",
    "ibm04",
    "ibm06",
    "ibm07",
    "ibm08",
    "ibm09",
    "ibm10",
    "ibm11",
    "ibm12",
    "ibm13",
    "ibm14",
    "ibm15",
    "ibm17",
    "ibm18",
]

print("=" * 70)


def run_one(name):
    try:
        costs, placement = run_placer(name, num_steps=1500)
        return name, costs
    except Exception as e:
        print(f"  {name} FAILED: {e}")
        return name, {"proxy_cost": float("inf")}


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Run specific benchmark: python gradesc.py ibm01
        bench_name = sys.argv[1]
        name, costs = run_one(bench_name)
        print(f"\n{name}: pc={costs.get('proxy_cost', 'FAIL')}")
    else:
        results = {}
        with Pool(6) as pool:
            for name, costs in pool.imap_unordered(run_one, BENCHMARKS):
                results[name] = costs
                print(f"  ✓ {name} done: pc={costs.get('proxy_cost', 'FAIL'):.4f}")

        # Summary
        print(f"\n\n{'='*70}")
        print(f"{'SUMMARY':^70}")
        print(f"{'='*70}")
        print(f"{'Bench':<8} {'Proxy':>8} {'WL':>8} {'Den':>8} {'Cong':>8} {'Ovlp':>6}")
        print(f"{'-'*70}")

        total = 0
        count = 0
        for name in BENCHMARKS:
            r = results[name]
            pc = r.get("proxy_cost", float("inf"))
            if pc == float("inf"):
                print(f"{name:<8} {'FAILED':>8}")
                continue
            print(
                f"{name:<8} {pc:>8.4f} {r['wirelength_cost']:>8.4f} "
                f"{r['density_cost']:>8.4f} {r['congestion_cost']:>8.4f} "
                f"{r.get('overlap_count', '?'):>6}"
            )
            total += pc
            count += 1

        if count > 0:
            avg = total / count
            print(f"{'-'*70}")
            print(f"{'AVG':<8} {avg:>8.4f}")
            print(f"\nRePlAce avg: 1.4578")
            print(f"SA avg:      2.1251")
            print(f"Your avg:    {avg:.4f}")
