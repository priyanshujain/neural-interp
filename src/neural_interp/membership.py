from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from neural_interp.density_peaks import Forest


@dataclass
class Membership:
    peaks: list[np.ndarray]
    geodesic: np.ndarray
    probs: np.ndarray
    entropy: np.ndarray
    temperature: float
    touching: list[tuple[int, int]]


def knn_indices(dist: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(dist), k), dtype=np.int64)
    for i, row in enumerate(dist):
        row = row.copy()
        row[i] = np.inf
        nearest = np.argpartition(row, k)[:k]
        out[i] = nearest[np.argsort(row[nearest], kind="stable")]
    return out


def knn_graph(dist: np.ndarray, k: int, mutual: bool = False) -> csr_matrix:
    n = len(dist)
    nbrs = knn_indices(dist, k)
    rows = np.repeat(np.arange(n), k)
    cols = nbrs.ravel()
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    linked = adj.multiply(adj.T) if mutual else adj.maximum(adj.T)
    linked = linked.tocoo()
    weights = np.maximum(dist[linked.row, linked.col], 1e-9)
    return csr_matrix((weights, (linked.row, linked.col)), shape=(n, n))


def peaks(forest: Forest, share: float = 0.8) -> list[np.ndarray]:
    density = np.exp(forest.rho - forest.rho.max())
    out = []
    for t, root in enumerate(forest.roots):
        members = np.flatnonzero(forest.tree == t)
        out.append(members[density[members] >= share * density[root]])
    return out


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-300, 1)
    return -(probs * np.log2(p)).sum(1)


def soft_membership(dist: np.ndarray, forest: Forest, k: int = 5, peak_share: float = 0.8) -> Membership:
    graph = knn_graph(dist, k)
    tops = peaks(forest, peak_share)
    geodesic = np.stack(
        [dijkstra(graph, directed=False, indices=top, min_only=True) for top in tops], axis=1
    )
    stranded = ~np.isfinite(geodesic).any(1)
    if stranded.any():
        direct = np.stack([dist[np.ix_(np.flatnonzero(stranded), top)].min(1) for top in tops], axis=1)
        geodesic[stranded] = direct
    nearest = dist[np.arange(len(dist)), knn_indices(dist, 1)[:, 0]]
    temperature = float(np.median(nearest[np.isfinite(nearest) & (nearest > 0)]))
    logits = -geodesic / temperature
    logits -= np.nanmax(np.where(np.isfinite(logits), logits, -np.inf), axis=1, keepdims=True)
    weights = np.where(np.isfinite(logits), np.exp(logits), 0.0)
    probs = weights / weights.sum(1, keepdims=True)

    mutual = knn_graph(dist, k, mutual=True).tocoo()
    pairs = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(forest.tree[mutual.row], forest.tree[mutual.col])
        if a != b
    }
    return Membership(tops, geodesic, probs, entropy_bits(probs), temperature, sorted(pairs))
