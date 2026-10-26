from dataclasses import dataclass

import numpy as np

from neural_interp.distance import signed_split
from neural_interp.selection import elbow_index


@dataclass
class Forest:
    rho: np.ndarray
    delta: np.ndarray
    parent: np.ndarray
    roots: np.ndarray
    tree: np.ndarray
    depth: np.ndarray
    rho_cut: float
    delta_cut: float


def collapse_copies(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.round(signed_split(rows), 6)
    _, first, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return rows[first[order]], counts[order], remap[inverse.ravel()]


def log_density(dist: np.ndarray, counts: np.ndarray, k: int) -> np.ndarray:
    radius = np.empty(len(dist))
    for i, row in enumerate(dist):
        order = np.argsort(row, kind="stable")
        mass = counts[order].copy()
        mass[order == i] -= 1
        reached = np.searchsorted(np.cumsum(mass), k)
        radius[i] = row[order[min(reached, len(order) - 1)]]
    positive = radius[radius > 0]
    offset = positive.min() / 2 if positive.size else 1.0
    return -np.log10(radius + offset)


def build_forest(dist: np.ndarray, counts: np.ndarray, k: int = 5) -> Forest:
    n = len(dist)
    rho = log_density(dist, counts, k)
    rank_order = np.lexsort((np.arange(n), -counts, -rho))
    rank = np.empty(n, dtype=np.int64)
    rank[rank_order] = np.arange(n)

    parent = np.full(n, -1, dtype=np.int64)
    delta = np.full(n, np.inf)
    for i in range(n):
        denser = rank < rank[i]
        if denser.any():
            candidates = np.where(denser, dist[i], np.inf)
            parent[i] = int(np.argmin(candidates))
            delta[i] = candidates[parent[i]]
    finite = np.isfinite(delta)
    delta[~finite] = delta[finite].max() if finite.any() else 1.0

    rho_cut = np.sort(rho)[::-1][elbow_index(np.sort(rho)[::-1])]
    delta_cut = np.sort(delta)[::-1][elbow_index(np.sort(delta)[::-1])]
    is_root = (rho >= rho_cut) & (delta >= delta_cut)
    is_root[rank_order[0]] = True
    roots = np.flatnonzero(is_root)
    roots = roots[np.argsort(rank[roots])]

    tree = np.full(n, -1, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    tree[roots] = np.arange(len(roots))
    for i in rank_order:
        if tree[i] < 0:
            tree[i] = tree[parent[i]]
            depth[i] = depth[parent[i]] + 1
    parent[roots] = -1
    by_size = np.argsort(-np.bincount(tree, weights=counts, minlength=len(roots)), kind="stable")
    renumber = np.empty_like(by_size)
    renumber[by_size] = np.arange(len(by_size))
    tree, roots = renumber[tree], roots[by_size]
    return Forest(rho, delta, parent, roots, tree, depth, float(rho_cut), float(delta_cut))
