from typing import Self

import numpy as np
from scipy.sparse.csgraph import dijkstra

from neural_interp.cluster._forest import DensityPeakForest
from neural_interp.cluster._graph import knn_graph, knn_indices
from neural_interp.typing import FloatArray, IntArray


def entropy_bits(probabilities: FloatArray) -> FloatArray:
    p = np.clip(probabilities, 1e-300, 1)
    return -(probabilities * np.log2(p)).sum(1)


class SoftMembership:
    def __init__(self, n_neighbors: int = 5, peak_share: float = 0.8) -> None:
        self.n_neighbors = n_neighbors
        self.peak_share = peak_share

    def peaks(self, forest: DensityPeakForest) -> list[IntArray]:
        out: list[IntArray] = []
        for t, root in enumerate(forest.roots_):
            members = np.flatnonzero(forest.labels_ == t)
            out.append(
                members[forest.rho_[members] >= self.peak_share * forest.rho_[root]].astype(
                    np.int64
                )
            )
        return out

    def fit(self, distances: FloatArray, forest: DensityPeakForest) -> Self:
        graph = knn_graph(distances, self.n_neighbors)
        peaks = self.peaks(forest)
        geodesic = np.stack(
            [dijkstra(graph, directed=False, indices=p, min_only=True) for p in peaks], axis=1
        )
        nearest = np.asarray(distances)[np.arange(len(distances)), knn_indices(distances, 1)[:, 0]]
        temperature = float(np.median(nearest[np.isfinite(nearest) & (nearest > 0)]))
        logits = -geodesic / temperature
        finite = np.isfinite(logits)
        top = np.where(finite, logits, -np.inf).max(1, keepdims=True)
        weights = np.where(finite, np.exp(logits - np.where(np.isfinite(top), top, 0)), 0.0)
        stranded = weights.sum(1) == 0
        weights[stranded] = 0.0
        weights[stranded, forest.labels_[stranded]] = 1.0
        probabilities = weights / weights.sum(1, keepdims=True)
        labels = probabilities.argmax(1).astype(np.int64)

        mutual = knn_graph(distances, self.n_neighbors, mutual=True).tocoo()
        touching = sorted(
            {
                (int(min(a, b)), int(max(a, b)))
                for a, b in zip(labels[mutual.row], labels[mutual.col])
                if a != b
            }
        )

        self.peaks_: list[IntArray] = peaks
        self.geodesic_: FloatArray = geodesic
        self.probabilities_: FloatArray = probabilities
        self.entropy_: FloatArray = entropy_bits(probabilities)
        self.labels_: IntArray = labels
        self.temperature_: float = temperature
        self.touching_: list[tuple[int, int]] = touching
        return self
