from pathlib import Path

import numpy as np
import torch
from in_tfm.density_peaks import DensityPeakForest, density_ranks, forest_trees, log_density, merge_copies

from neural_interp.lib.forest import peak_pull_layout
from neural_interp.lib.geometry import local_intrinsic_dim, mutual_knn, other_tree, tree_percentile, unit

K_DENSITY = 10
K_STORED = 20
BLOCK = 2048
N_PAIRS = 400
PATH_STEPS = 16
DEVICE = "cuda"


def to_distance(sim):
    return torch.sqrt(torch.clamp(2 - 2 * sim, min=0))


def blocks(R):
    for start in range(0, len(R), BLOCK):
        sim = R[start:start + BLOCK] @ R.T
        yield sim, torch.arange(start, start + len(sim), device=DEVICE)


def gpu_forest(R, counts):
    n = len(R)
    counts_t = torch.from_numpy(counts).to(DEVICE)
    neighbours = torch.empty((n, K_STORED), dtype=torch.long, device=DEVICE)
    neighbour_sim = torch.empty((n, K_STORED), device=DEVICE)
    for sim, own in blocks(R):
        sim[torch.arange(len(sim), device=DEVICE), own] = -float("inf")
        top = sim.topk(K_STORED, dim=1)
        neighbours[own], neighbour_sim[own] = top.indices, top.values
    neighbour_distance = to_distance(neighbour_sim)
    print("knn done", flush=True)

    hits_so_far = (counts_t[:, None] - 1) + torch.cumsum(counts_t[neighbours], dim=1)
    reached = torch.clamp((hits_so_far < K_DENSITY).sum(1), max=K_STORED - 1)
    kth = neighbour_distance.gather(1, reached[:, None])[:, 0]
    kth = torch.where(counts_t - 1 >= K_DENSITY, torch.zeros_like(kth), kth).cpu().numpy().astype(np.float64)
    ranks = density_ranks(kth, counts)
    ranks_t = torch.from_numpy(ranks).to(DEVICE)

    parent = torch.full((n,), -1, dtype=torch.long, device=DEVICE)
    delta = torch.zeros(n, device=DEVICE)
    for sim, own in blocks(R):
        denser = ranks_t[None, :] > ranks_t[own, None]
        best = torch.where(denser, sim, torch.full_like(sim, -float("inf"))).max(1)
        has_denser = denser.any(1)
        parent[own[has_denser]] = best.indices[has_denser]
        delta[own[has_denser]] = to_distance(best.values[has_denser])
        if (~has_denser).any():
            delta[own[~has_denser]] = to_distance(sim[~has_denser].min(1).values)
    print("nearest denser done", flush=True)

    forest = DensityPeakForest(
        distinct_of_hit=np.arange(n), counts=counts, kth_distance=kth, density_rank=ranks,
        parent=parent.cpu().numpy(), delta=delta.cpu().numpy().astype(np.float64),
        neighbours=neighbours[:, :K_DENSITY].cpu().numpy(), neighbour_distance=neighbour_distance[:, :K_DENSITY].cpu().numpy(),
    )
    return forest, neighbour_distance.cpu().numpy()


def halo_mask(tree, rho, nb, crossing, n_trees):
    border_rho = np.full(n_trees, -np.inf)
    i_idx, j_idx = np.nonzero(crossing)
    pair_rho = (rho[i_idx] + rho[nb[i_idx, j_idx]]) / 2
    np.maximum.at(border_rho, tree[i_idx], pair_rho)
    np.maximum.at(border_rho, tree[nb[i_idx, j_idx]], pair_rho)
    return rho < border_rho[tree]


def geodesic_paths(R, tree, halo, tree_pct, rng):
    big_trees = np.flatnonzero(np.bincount(tree) >= 50)
    pairs = []
    for kind, pool in [("core", ~halo & (tree_pct >= 0.8)), ("halo", halo)]:
        for _ in range(N_PAIRS):
            members = np.flatnonzero(pool & (tree == rng.choice(big_trees)))
            if len(members) >= 2:
                pairs.append((kind, *rng.choice(members, 2, replace=False)))
    a, b = R[[p[1] for p in pairs]], R[[p[2] for p in pairs]]
    omega = torch.arccos(torch.clamp((a * b).sum(1), -1 + 1e-6, 1 - 1e-6))[:, None]
    steps = torch.linspace(0, 1, PATH_STEPS, device=DEVICE)[None, :]
    weight_a = torch.sin((1 - steps) * omega) / torch.sin(omega)
    weight_b = torch.sin(steps * omega) / torch.sin(omega)
    queries = (weight_a[:, :, None] * a[:, None, :] + weight_b[:, :, None] * b[:, None, :]).reshape(-1, R.shape[1])
    queries = queries / queries.norm(dim=1, keepdim=True)
    best = [(queries[s:s + BLOCK] @ R.T).max(1) for s in range(0, len(queries), BLOCK)]
    path_token = torch.cat([x.indices for x in best]).reshape(len(pairs), PATH_STEPS).cpu().numpy()
    path_gap = to_distance(torch.cat([x.values for x in best])).reshape(len(pairs), PATH_STEPS).cpu().numpy()
    kind = np.array([p[0] for p in pairs])
    print("paths", len(pairs), "mean interior gap core", round(float(path_gap[kind == "core", 1:-1].mean()), 4),
          "halo", round(float(path_gap[kind == "halo", 1:-1].mean()), 4))
    return dict(
        path_kind=kind, path_ends=np.array([[p[1], p[2]] for p in pairs]),
        path_end_distance=to_distance((a * b).sum(1)).cpu().numpy(), path_token=path_token, path_gap=path_gap,
    )


def run(out, mode):
    out = Path(out)
    rng = np.random.default_rng(0)
    centred = np.load(out / "emb.npy").astype(np.float64)
    centred = centred - centred.mean(0)
    if mode == "null":
        cov = np.cov(centred, rowvar=False)
        root_cov = np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))
        centred = rng.standard_normal(centred.shape) @ root_cov.T
        print("null: gaussian with the embedding covariance", centred.shape)
    rows = unit(centred)
    assert len(merge_copies(rows)[0]) == len(rows), "copies would need mapping back to tokens"
    n = len(rows)
    print(mode, "tokens", n, "distinct", n)
    R = torch.from_numpy(rows).float().to(DEVICE)

    forest, all_neighbour_distance = gpu_forest(R, np.ones(n, dtype=int))
    trees = forest_trees(forest)
    tree, nb = trees.tree, forest.neighbours
    rho = log_density(forest.kth_distance)
    is_root = np.zeros(n, dtype=bool)
    is_root[trees.roots] = True
    print("trees", len(trees.roots), "cuts", trees.cuts, flush=True)

    crossing = other_tree(tree, nb)
    halo = halo_mask(tree, rho, nb, crossing, len(trees.roots))
    tree_pct = tree_percentile(tree, rho)
    dist_root = to_distance((R * R[torch.from_numpy(trees.roots[tree]).to(DEVICE)]).sum(1)).cpu().numpy()
    local_id = local_intrinsic_dim(all_neighbour_distance)
    mutual = mutual_knn(nb)
    k_occurrence = np.bincount(nb.ravel(), minlength=n)
    skew = float(((k_occurrence - k_occurrence.mean()) ** 3).mean() / k_occurrence.std() ** 3)
    print("halo fraction", round(float(halo.mean()), 3), "local id median", round(float(np.median(local_id)), 2),
          "mutual mean", round(float(mutual.mean()), 3), "hub skew", round(skew, 2))

    result = dict(
        distinct_of_token=np.arange(n), tree=tree, depth=trees.depth, parent=forest.parent, roots=trees.roots,
        is_root=is_root, rho=rho, delta=forest.delta, halo=halo, tree_pct=tree_pct, dist_root=dist_root,
        local_id=local_id, mutual=mutual, agree=1 - crossing.mean(1), k_occurrence=k_occurrence, hub_skew=skew,
        neighbours=nb, neighbour_distance=forest.neighbour_distance, cut_rho=trees.cuts.rho, cut_delta=trees.cuts.delta,
    )
    if mode == "real":
        result.update(geodesic_paths(R, tree, halo, tree_pct, rng))
        result["xy"], _ = peak_pull_layout(rows, forest, trees, per_tree=2)
    np.savez(out / f"forest_{mode}.npz", **result)
    print("saved", out / f"forest_{mode}.npz")
