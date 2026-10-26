from typing import Self

import numpy as np
import torch

from neural_interp._device import default_device
from neural_interp.cluster._graph import nearest
from neural_interp.selection import elbow_index
from neural_interp.typing import FloatArray, IntArray


class DensityPeakForest:
    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def log_density(self, distances: FloatArray, counts: IntArray) -> FloatArray:
        values, indices = nearest(distances, self.n_neighbors + 1)
        mass = counts[indices]
        own = np.maximum(counts - 1, 0)
        reached = np.argmax(own[:, None] + np.cumsum(mass, axis=1) >= self.n_neighbors, axis=1)
        radius = np.where(own >= self.n_neighbors, 0.0, values[np.arange(len(values)), reached])
        positive = radius[radius > 0]
        offset = positive.min() / 2 if positive.size else 1.0
        return -np.log10(radius + offset)

    def nearest_denser(
        self, distances: FloatArray, rank: IntArray, chunk: int = 2048
    ) -> tuple[IntArray, FloatArray]:
        device = default_device()
        n = len(distances)
        ranks = torch.as_tensor(rank, device=device)
        parent = np.full(n, -1, dtype=np.int64)
        delta = np.full(n, np.inf)
        for start in range(0, n, chunk):
            block = torch.as_tensor(
                np.array(distances[start : start + chunk]), device=device
            ).float()
            denser = ranks[None, :] < ranks[start : start + chunk, None]
            masked = torch.where(denser, block, torch.full_like(block, torch.inf))
            found = masked.min(dim=1)
            delta[start : start + chunk] = found.values.cpu().numpy()
            parent[start : start + chunk] = found.indices.cpu().numpy()
        parent[~np.isfinite(delta)] = -1
        return parent, delta

    def fit(self, distances: FloatArray, counts: IntArray | None = None) -> Self:
        n = len(distances)
        counts = np.ones(n, dtype=np.int64) if counts is None else counts
        rho = self.log_density(distances, counts)
        rank_order = np.lexsort((np.arange(n), -counts, -rho))
        rank = np.empty(n, dtype=np.int64)
        rank[rank_order] = np.arange(n)

        parent, delta = self.nearest_denser(distances, rank)
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
