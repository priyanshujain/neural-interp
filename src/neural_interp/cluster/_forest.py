from typing import Self

import numpy as np

from neural_interp.selection import elbow_index
from neural_interp.typing import FloatArray, IntArray


class DensityPeakForest:
    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def log_density(self, distances: FloatArray, counts: IntArray) -> FloatArray:
        radius = np.empty(len(distances))
        for i in range(len(distances)):
            row = np.asarray(distances[i])
            order = np.argsort(row, kind="stable")
            mass = counts[order].copy()
            mass[order == i] -= 1
            reached = int(np.searchsorted(np.cumsum(mass), self.n_neighbors))
            radius[i] = row[order[min(reached, len(order) - 1)]]
        positive = radius[radius > 0]
        offset = positive.min() / 2 if positive.size else 1.0
        return -np.log10(radius + offset)

    def fit(self, distances: FloatArray, counts: IntArray | None = None) -> Self:
        n = len(distances)
        counts = np.ones(n, dtype=np.int64) if counts is None else counts
        rho = self.log_density(distances, counts)
        rank_order = np.lexsort((np.arange(n), -counts, -rho))
        rank = np.empty(n, dtype=np.int64)
        rank[rank_order] = np.arange(n)

        parent = np.full(n, -1, dtype=np.int64)
        delta = np.full(n, np.inf)
        for i in range(n):
            denser = rank < rank[i]
            if denser.any():
                candidates = np.where(denser, np.asarray(distances[i]), np.inf)
                parent[i] = int(np.argmin(candidates))
                delta[i] = candidates[parent[i]]
        finite = np.isfinite(delta)
        delta[~finite] = delta[finite].max() if finite.any() else 1.0

        rho_sorted = np.sort(rho)[::-1]
        delta_sorted = np.sort(delta)[::-1]
        rho_cut = float(rho_sorted[elbow_index(rho_sorted)])
        delta_cut = float(delta_sorted[elbow_index(delta_sorted)])
        is_root = (rho >= rho_cut) & (delta >= delta_cut)
        is_root[rank_order[0]] = True
        roots = np.flatnonzero(is_root)
        roots = roots[np.argsort(rank[roots])]

        labels = np.full(n, -1, dtype=np.int64)
        depth = np.zeros(n, dtype=np.int64)
        labels[roots] = np.arange(len(roots))
        for i in rank_order:
            if labels[i] < 0:
                labels[i] = labels[parent[i]]
                depth[i] = depth[parent[i]] + 1
        parent[roots] = -1
        by_size = np.argsort(
            -np.bincount(labels, weights=counts, minlength=len(roots)), kind="stable"
        )
        renumber = np.empty_like(by_size)
        renumber[by_size] = np.arange(len(by_size))

        self.rho_: FloatArray = rho
        self.delta_: FloatArray = delta
        self.parent_: IntArray = parent
        self.roots_: IntArray = roots[by_size].astype(np.int64)
        self.labels_: IntArray = renumber[labels].astype(np.int64)
        self.depth_: IntArray = depth
        self.rho_cut_: float = rho_cut
        self.delta_cut_: float = delta_cut
        return self
