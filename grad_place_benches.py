from sklearn.cluster import KMeans
import traceback
import random
import time
import sys
from multiprocessing import Pool
import torch
from scipy.fft import dctn, idctn
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost, _set_placement
from gradient_place import (
    legalize,
    precompute_net_tensors,
    differentiable_wirelength,
    differentiable_overlap_penalty,
)

BENCH_CACHE = {}


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


def compute_density_grid_fast(placement, benchmark):
    num_macros = benchmark.num_hard_macros + benchmark.num_soft_macros
    sizes = benchmark.macro_sizes[:num_macros]
    grid_rows = benchmark.grid_rows
    grid_cols = benchmark.grid_cols
    bin_w = benchmark.canvas_width / grid_cols
    bin_h = benchmark.canvas_height / grid_rows

    # Macro edges
    cx = placement[:num_macros, 0]
    cy = placement[:num_macros, 1]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    left = cx - half_w
    right = cx + half_w
    bottom = cy - half_h
    top = cy + half_h

    # Bin edges
    bin_left = torch.arange(grid_cols, dtype=torch.float32) * bin_w
    bin_right = bin_left + bin_w
    bin_bottom = torch.arange(grid_rows, dtype=torch.float32) * bin_h
    bin_top = bin_bottom + bin_h

    # Overlap in x: (num_macros, grid_cols)
    overlap_x = torch.clamp(
        torch.min(right.unsqueeze(1), bin_right.unsqueeze(0))
        - torch.max(left.unsqueeze(1), bin_left.unsqueeze(0)),
        min=0,
    )

    # Overlap in y: (num_macros, grid_rows)
    overlap_y = torch.clamp(
        torch.min(top.unsqueeze(1), bin_top.unsqueeze(0))
        - torch.max(bottom.unsqueeze(1), bin_bottom.unsqueeze(0)),
        min=0,
    )

    # Density grid: (grid_rows, grid_cols) = sum of overlap areas
    # overlap_area[macro, row, col] = overlap_y[macro, row] * overlap_x[macro, col]
    # Sum over macros: density[row, col] = sum_macro overlap_y[:,row].T @ overlap_x[:,col]
    density = torch.mm(overlap_y.t(), overlap_x) / (bin_w * bin_h)

    return density


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


def compute_net_weights(placement, nets, benchmark, plc):
    from macro_place.objective import _set_placement

    _set_placement(plc, placement, benchmark)
    plc.FLAG_UPDATE_WIRELENGTH = False

    h_cong = plc.get_horizontal_routing_congestion()
    v_cong = plc.get_vertical_routing_congestion()

    rows, cols = benchmark.grid_rows, benchmark.grid_cols
    h_grid = torch.tensor(h_cong, dtype=torch.float32).reshape(rows, cols)
    v_grid = torch.tensor(v_cong, dtype=torch.float32).reshape(rows, cols)
    cong_grid = (h_grid + v_grid) / 2

    bin_w = benchmark.canvas_width / cols
    bin_h = benchmark.canvas_height / rows

    weights = torch.ones(len(nets))

    for i, net in enumerate(nets):
        xs = [placement[m, 0].item() for m in net]
        ys = [placement[m, 1].item() for m in net]
        min_c = max(0, int(min(xs) / bin_w))
        max_c = min(cols - 1, int(max(xs) / bin_w))
        min_r = max(0, int(min(ys) / bin_h))
        max_r = min(rows - 1, int(max(ys) / bin_h))

        region = cong_grid[min_r : max_r + 1, min_c : max_c + 1]
        if region.numel() > 0:
            avg_cong = region.mean().item()
            weights[i] = 1.0 + max(0, avg_cong - 0.5) * 2.0

    return weights


def compute_density_force_fast(potential, placement, benchmark):
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


def reduce_congestion(placement, benchmark, plc, nets, num_iterations=500):
    """Move macros out of congested cells using TILOS-guided local search."""
    print("Reducing congestion with local search...")
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]
    rows, cols = benchmark.grid_rows, benchmark.grid_cols
    bin_w = benchmark.canvas_width / cols
    bin_h = benchmark.canvas_height / rows

    _set_placement(plc, placement, benchmark)
    plc.FLAG_UPDATE_WIRELENGTH = False
    best_cong = plc.get_congestion_cost()
    best_den = plc.get_density_cost()
    best_score = 0.5 * best_den + 0.5 * best_cong

    for iteration in range(num_iterations):
        # Find most congested cells
        h_cong = plc.get_horizontal_routing_congestion()
        v_cong = plc.get_vertical_routing_congestion()
        cong_grid = torch.tensor([(h + v) / 2 for h, v in zip(h_cong, v_cong)]).reshape(rows, cols)

        # Find macros in congested cells
        macro_cong = torch.zeros(num_hard)
        for i in range(num_hard):
            c = min(cols - 1, max(0, int(placement[i, 0].item() / bin_w)))
            r = min(rows - 1, max(0, int(placement[i, 1].item() / bin_h)))
            macro_cong[i] = cong_grid[r, c]

        # Pick a macro from the top 20% most congested
        top_k = max(1, num_hard // 5)
        _, worst_indices = torch.topk(macro_cong, top_k)
        idx = worst_indices[random.randint(0, top_k - 1)].item()

        if benchmark.macro_fixed[idx]:
            continue

        # Try moving it slightly
        old_x, old_y = placement[idx, 0].item(), placement[idx, 1].item()
        shift = max(sizes[idx, 0].item(), sizes[idx, 1].item()) * 0.3
        new_x = old_x + random.gauss(0, shift)
        new_y = old_y + random.gauss(0, shift)
        new_x = max(
            sizes[idx, 0].item() / 2, min(benchmark.canvas_width - sizes[idx, 0].item() / 2, new_x)
        )
        new_y = max(
            sizes[idx, 1].item() / 2, min(benchmark.canvas_height - sizes[idx, 1].item() / 2, new_y)
        )

        # Quick overlap check
        has_overlap = False
        for j in range(num_hard):
            if j == idx:
                continue
            if (
                abs(new_x - placement[j, 0].item())
                < (sizes[idx, 0] + sizes[j, 0]).item() / 2 + 0.01
                and abs(new_y - placement[j, 1].item())
                < (sizes[idx, 1] + sizes[j, 1]).item() / 2 + 0.01
            ):
                has_overlap = True
                break
        if has_overlap:
            continue

        # Evaluate
        placement[idx, 0] = new_x
        placement[idx, 1] = new_y
        _set_placement(plc, placement, benchmark)
        plc.FLAG_UPDATE_WIRELENGTH = False
        new_den = plc.get_density_cost()
        new_cong = plc.get_congestion_cost()
        new_score = 0.5 * new_den + 0.5 * new_cong

        if new_score < best_score:
            best_score = new_score
            best_den = new_den
            best_cong = new_cong
        else:
            placement[idx, 0] = old_x
            placement[idx, 1] = old_y

    return placement


def run_placer_multiseed(benchmark_name, num_steps=800, lr=1.0, momentum=0.9, seeds=[-3,-2,-1,0,42]):
    best_costs = None
    best_legal = None
    best_pc = float("inf")
    best_seed = None
    for seed in seeds:
        try:
            costs, legal = run_placer(
                benchmark_name, num_steps=num_steps, lr=lr, momentum=momentum, seed=seed
            )
            pc = costs.get("proxy_cost", float("inf"))
            overlaps = costs.get("overlap_count", 999)
            print(f"  {benchmark_name} seed={seed}: pc={pc:.4f} ovlp={overlaps}", flush=True)

            # Simple: prefer zero overlaps, then lowest proxy cost
            better = False
            if best_costs is None:
                better = True
            elif overlaps == 0 and best_costs.get("overlap_count", 999) > 0:
                better = True
            elif overlaps == 0 and best_costs.get("overlap_count", 999) == 0 and pc < best_pc:
                better = True
            elif overlaps > 0 and best_costs.get("overlap_count", 999) > 0 and pc < best_pc:
                better = True

            if better:
                best_pc = pc
                best_costs = costs
                best_legal = legal
                best_seed = seed
        except Exception as e:
            print(f"  {benchmark_name} seed={seed} FAILED: {e}", flush=True)

    print(
        f"  {benchmark_name} BEST: pc={best_pc:.4f} ovlp={best_costs.get('overlap_count', '?')} seed={best_seed}",
        flush=True,
    )
    best_costs["best_seed"] = best_seed
    return best_costs, best_legal

def connectivity_init(placement, nets, benchmark, num_clusters=6):
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]
    
    # Build adjacency matrix
    adj = torch.zeros(num_hard, num_hard)
    for net in nets:
        hard_members = [m for m in net if m < num_hard]
        for i in range(len(hard_members)):
            for j in range(i+1, len(hard_members)):
                adj[hard_members[i], hard_members[j]] += 1
                adj[hard_members[j], hard_members[i]] += 1
    
    # Spectral clustering
    # Degree matrix
    degree = adj.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / (degree.sqrt() + 1e-8))
    # Normalized Laplacian
    L = torch.eye(num_hard) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Smallest eigenvectors (skip first which is constant)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    features = eigenvectors[:, 1:num_clusters+1]  # use eigenvectors 1..k
    
    # K-means on the features (simple version)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(features.numpy())
    
    # Assign each cluster to a region of the canvas
    cols = int(num_clusters ** 0.5) + 1
    rows_grid = (num_clusters + cols - 1) // cols
    region_w = benchmark.canvas_width / cols
    region_h = benchmark.canvas_height / rows_grid
    
    for cluster_id in range(num_clusters):
        members = [i for i in range(num_hard) if labels[i] == cluster_id]
        if not members:
            continue
        
        # Region center
        col = cluster_id % cols
        row = cluster_id // cols
        region_cx = (col + 0.5) * region_w
        region_cy = (row + 0.5) * region_h
        
        # Place members around region center
        n = len(members)
        side = int(n ** 0.5) + 1
        for idx, macro_id in enumerate(members):
            if benchmark.macro_fixed[macro_id]:
                continue
            local_r = idx // side
            local_c = idx % side
            # Spread within region
            spacing = min(region_w, region_h) / (side + 1)
            x = region_cx + (local_c - side/2) * spacing
            y = region_cy + (local_r - side/2) * spacing
            # Clamp
            hw = sizes[macro_id, 0].item() / 2
            hh = sizes[macro_id, 1].item() / 2
            x = max(hw, min(benchmark.canvas_width - hw, x))
            y = max(hh, min(benchmark.canvas_height - hh, y))
            placement[macro_id, 0] = x
            placement[macro_id, 1] = y
    
    return placement

def optimize_soft_macros_gentle(placement, nets, benchmark, alpha=0.3):
    """Move each soft macro partially toward centroid of connected macros."""
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

        target_x = sum(placement[j, 0].item() for j in connected) / len(connected)
        target_y = sum(placement[j, 1].item() for j in connected) / len(connected)

        # Move alpha fraction toward target (0.3 = 30% of the way)
        old_x = placement[soft_idx, 0].item()
        old_y = placement[soft_idx, 1].item()
        new_x = old_x + alpha * (target_x - old_x)
        new_y = old_y + alpha * (target_y - old_y)

        w = benchmark.macro_sizes[soft_idx, 0].item()
        h = benchmark.macro_sizes[soft_idx, 1].item()
        placement[soft_idx, 0] = max(w/2, min(benchmark.canvas_width - w/2, new_x))
        placement[soft_idx, 1] = max(h/2, min(benchmark.canvas_height - h/2, new_y))

    return placement

def run_placer(benchmark_name, num_steps=800, lr=1.0, momentum=0.9, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if not BENCH_CACHE.get(benchmark_name):
        benchmark, plc = load_benchmark_from_dir(
            f"external/MacroPlacement/Testcases/ICCAD04/{benchmark_name}"
        )
        BENCH_CACHE[benchmark_name] = (benchmark, plc)
    else:
        benchmark, plc = BENCH_CACHE[benchmark_name]
    nets = extract_nets(plc, benchmark)

    net_indices, net_mask = precompute_net_tensors(nets)
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]
    
    if seed == -1:
        start = time.time()
        # Gentle density-only optimization from initial placement
        placement = benchmark.macro_positions.clone()
        placement.requires_grad_(False)

        for step in range(500):
            grid = compute_density_grid_fast(placement, benchmark)
            potential = solve_poisson(grid, benchmark)
            forces = compute_density_force_fast(potential, placement, benchmark)

            # Very small steps, density only, no wirelength
            placement[:num_hard] += 0.05 * forces

            hw = sizes[:, 0] / 2
            hh = sizes[:, 1] / 2
            placement[:num_hard, 0].clamp_(min=hw, max=benchmark.canvas_width - hw)
            placement[:num_hard, 1].clamp_(min=hh, max=benchmark.canvas_height - hh)
            if benchmark.macro_fixed.any():
                placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

        legal = legalize_fast(placement, benchmark, gap=0.01, max_iters=1000)
        costs_final = compute_proxy_cost(legal, benchmark, plc)
        costs_final["seed"] = seed
        elapsed = time.time() - start
        print(
            f"  {benchmark_name} FINAL (gentle density): pc={costs_final['proxy_cost']:.4f} "
            f"wl={costs_final['wirelength_cost']:.4f} den={costs_final['density_cost']:.4f} "
            f"cong={costs_final['congestion_cost']:.4f} ovlp={costs_final['overlap_count']}",
            flush=True,
        )
        return costs_final, legal

    if seed == 0:
        # Start from initial placement
        placement = benchmark.macro_positions.clone()
        placement.requires_grad_(False)
        
        den_weight = 0.01
        for step in range(200):
            grid = compute_density_grid_fast(placement, benchmark)
            potential = solve_poisson(grid, benchmark)
            forces = compute_density_force_fast(potential, placement, benchmark)
            placement[:num_hard] += 0.02 * forces
            
            # Clamp
            hw = sizes[:, 0] / 2
            hh = sizes[:, 1] / 2
            placement[:num_hard, 0].clamp_(min=hw, max=benchmark.canvas_width - hw)
            placement[:num_hard, 1].clamp_(min=hh, max=benchmark.canvas_height - hh)
            if benchmark.macro_fixed.any():
                placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        
        legal = legalize_fast(placement, benchmark, gap=0.01, max_iters=1000)
        costs_final = compute_proxy_cost(legal, benchmark, plc)
        
    elif seed > 0:
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

    if seed == -2:
        placement = benchmark.macro_positions.clone()
        cx = benchmark.canvas_width / 2
        cy = benchmark.canvas_height / 2
        scale = 1.05  # gentler scaling
        for i in range(num_hard):
            if not benchmark.macro_fixed[i]:
                placement[i, 0] = cx + (placement[i, 0] - cx) * scale
                placement[i, 1] = cy + (placement[i, 1] - cy) * scale
                hw = sizes[i, 0].item() / 2
                hh = sizes[i, 1].item() / 2
                placement[i, 0] = max(hw, min(benchmark.canvas_width - hw, placement[i, 0].item()))
                placement[i, 1] = max(hh, min(benchmark.canvas_height - hh, placement[i, 1].item()))

        legal = legalize_fast(placement, benchmark, gap=0.01, max_iters=1000)
        costs_final = compute_proxy_cost(legal, benchmark, plc)
        costs_final["seed"] = seed
        print(
            f"  {benchmark_name} FINAL (scaled 1.05): pc={costs_final['proxy_cost']:.4f} "
            f"wl={costs_final['wirelength_cost']:.4f} den={costs_final['density_cost']:.4f} "
            f"cong={costs_final['congestion_cost']:.4f} ovlp={costs_final['overlap_count']}",
            flush=True,
        )
        return costs_final, legal

    if seed == -3:
        placement = benchmark.macro_positions.clone()
        placement = connectivity_init(placement, nets, benchmark, num_clusters=6)
        # Then fall through to the normal optimization loop below

    velocity = torch.zeros_like(placement)
    den_weight = 0.001
    best_proxy = float("inf")
    best_placement = placement.clone()

    start = time.time()

    net_weights = None
    for step in range(1300):
        # Update net weights every 100 steps
        if step % 100 == 0 and step > 0:
            # t0=time.time()
            # with torch.no_grad():
            # placement.data = optimize_soft_macros(placement.detach(), nets, benchmark).data
            _set_placement(plc, placement.detach(), benchmark)
            plc.FLAG_UPDATE_WIRELENGTH = False
            net_weights = compute_net_weights(placement.detach(), nets, benchmark, plc)
            # print("  {} step {}: computed net weights in {:.1f}ms".format(benchmark_name, step, (time.time() - t0) * 1000))

        placement.requires_grad_(True)
        # t0=time.time()
        wl = differentiable_wirelength(
            placement, nets, benchmark, net_indices=net_indices, net_mask=net_mask
        )

        wl.backward()
        # print("  {} step {}: computed wirelength in {:.1f}ms".format(benchmark_name, step, (time.time() - t0) * 1000))

        # t0=time.time()
        wl_grad = placement.grad[:num_hard].detach().clone()
        wl_grad_full = placement.grad.detach().clone()
        wl_grad = wl_grad_full[:num_hard]
        # Scale down soft macro gradient
        soft_grad = wl_grad_full[num_hard:] * 0.3
        # print("  {} step {}: computed wirelength gradient in {:.1f}ms".format(benchmark_name, step, (time.time() - t0) * 1000))
        placement.requires_grad_(False)

        # t0=time.time()
        grid = compute_density_grid_fast(placement, benchmark)
        # print("  {} step {}: computed density grid in {:.1f}ms".format(benchmark_name, step, (time.time() - t0) * 1000))
        # t0=time.time()
        potential = solve_poisson(grid, benchmark)
        # print("  {} step {}: solved Poisson in {:.1f}ms".format(benchmark_name, step, (time.time() - t0) * 1000))
        # t0=time.time()
        density_forces = compute_density_force_fast(potential, placement, benchmark)
        # print("  {} step {}: computed density forces in {:.1f}ms".format(benchmark_name, step, (time.time() - t0) * 1000))

        if step < 1000:
            # Pure wirelength
            total_grad = wl_grad
        else:
            # Density spreading to prepare for legalization
            total_grad = wl_grad - den_weight * density_forces
            
        if benchmark.macro_fixed.any():
            total_grad[benchmark.macro_fixed[:num_hard]] = 0.0

        velocity[:num_hard] = momentum * velocity[:num_hard] - lr * total_grad
        placement[:num_hard] += velocity[:num_hard]

        # Soft macro update (simple gradient descent, no density force)
        placement.data[num_hard:] -= 0.003 * soft_grad
        # Clamp all macros to canvas
        num_all = num_hard + benchmark.num_soft_macros
        all_sizes = benchmark.macro_sizes[:num_all]
        hw = all_sizes[:, 0] / 2
        hh = all_sizes[:, 1] / 2
        placement.data[:num_all, 0].clamp_(min=hw, max=benchmark.canvas_width - hw)
        placement.data[:num_all, 1].clamp_(min=hh, max=benchmark.canvas_height - hh)

        if benchmark.macro_fixed.any():
            placement.data[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

        if benchmark.macro_fixed.any():
            placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

        if step % 10 == 0 and step > 0:
            _set_placement(plc, placement.detach(), benchmark)
            plc.FLAG_UPDATE_WIRELENGTH = False
            tilos_den = plc.get_density_cost()
            if tilos_den > 0.85:
                den_weight = min(0.1, den_weight * 1.1)
            elif tilos_den < 0.75:
                den_weight = max(0.001, den_weight * 0.95)  # reduce if overspreading

        if step % 200 == 0:
            _set_placement(plc, placement.detach(), benchmark)
            plc.FLAG_UPDATE_WIRELENGTH = False
            den = plc.get_density_cost()
            cong = plc.get_congestion_cost()
            wl_val = wl.item()  # already computed this step
            proxy_est = wl_val + 0.5 * den + 0.5 * cong

            # Count overlaps fast
            with torch.no_grad():
                p = placement[:num_hard].detach()
                dx = torch.abs(p.unsqueeze(0)[:, :, 0] - p.unsqueeze(1)[:, :, 0])
                dy = torch.abs(p.unsqueeze(0)[:, :, 1] - p.unsqueeze(1)[:, :, 1])
                sx = (sizes[:, 0].unsqueeze(0) + sizes[:, 0].unsqueeze(1)) / 2
                sy = (sizes[:, 1].unsqueeze(0) + sizes[:, 1].unsqueeze(1)) / 2
                tri = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
                olaps = ((dx < sx) & (dy < sy) & tri).sum().item()

            print(
                f"  {benchmark_name}.{seed}.{step}: pc={proxy_est:.4f} wl={wl_val:.4f} "
                f"den={den:.4f} cong={cong:.4f} ovlp={olaps} dw={den_weight:.4f}",
                flush=True,
            )

            if proxy_est < best_proxy:
                best_proxy = proxy_est
                best_placement = placement.detach().clone()

    # After optimization, before legalization, add:
    hard_only = [n for n in nets if all(m < num_hard for m in n)]
    mixed = [n for n in nets if any(m < num_hard for m in n) and any(m >= num_hard for m in n)]
    soft_only = [n for n in nets if all(m >= num_hard for m in n)]

    print(f"Hard-only nets: {len(hard_only)}")
    print(f"Mixed nets: {len(mixed)}")
    print(f"Soft-only nets: {len(soft_only)}")
    print(f"Total: {len(nets)}")
    
    # Legalize
    first_legalize_start = time.time()
    print(f"  {benchmark_name} starting legalization with best proxy cost {best_proxy:.4f}")
    legal = legalize_fast(best_placement, benchmark, gap=0.01, max_iters=1000)
    first_legalize_end = time.time()
    print(f"  First legalization pass took {first_legalize_end - first_legalize_start:.0f}s")
    legal = optimize_soft_macros_gentle(legal, nets, benchmark, alpha=.05)
    costs_final = compute_proxy_cost(legal, benchmark, plc)
    costs_final["seed"] = seed
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
    "ibm16",
    "ibm15",
    "ibm17",
    "ibm18",
]

print("=" * 70)


def run_one(name, seedcount=None):
    try:
        if seedcount is not None and seedcount != "multi":
            seeds = [int(seedcount)]
            costs, _ = run_placer_multiseed(name, seeds=seeds)
        elif seedcount == "multi":
            costs, _ = run_placer_multiseed(name)
        else:
            costs, _ = run_placer_multiseed(name)
        return name, costs
    except Exception as e:
        traceback.print_exc()
        print(f"  {name} FAILED: {e}", flush=True)
        return name, {"proxy_cost": float("inf")}


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # Run specific benchmark: python gradesc.py ibm01
        bench_name = sys.argv[1]
        seedcount = sys.argv[2] if len(sys.argv) > 2 else "multi"
        name, costs = run_one(bench_name, seedcount)
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
        print(f"{'Bench':<8} {'Proxy':>8} {'WL':>8} {'Den':>8} {'Cong':>8} {'Ovlp':>6} {'Seed':>5}")
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
                f"{r.get('overlap_count', '?'):>6} {r.get('best_seed', '?'):>5}"
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
