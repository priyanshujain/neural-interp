import networkx as nx
import numpy as np

from neural_interp.buckets import Bucket, ranked_sample


def map_nodes(buckets: list[Bucket], probs: np.ndarray, root_hits: np.ndarray, per_bucket: int = 8) -> np.ndarray:
    picked = [ranked_sample(b, probs, per_bucket) for b in buckets]
    return np.unique(np.concatenate(picked + [root_hits]))


def hit_map(
    dist: np.ndarray,
    probs: np.ndarray,
    nodes: np.ndarray,
    root_hits: np.ndarray,
    ks=(6, 10, 15, 25),
    pull_min: float = 0.1,
    pull_weight: float = 3.0,
    seed: int = 0,
) -> dict[int, tuple[float, float]]:
    sub = dist[np.ix_(nodes, nodes)]
    masked = sub + np.diag(np.full(len(nodes), np.inf))
    for k in ks:
        g = nx.Graph()
        g.add_nodes_from(nodes.tolist())
        nbrs = np.argsort(masked, axis=1)[:, :k]
        for a, row in enumerate(nbrs):
            for b in row:
                g.add_edge(int(nodes[a]), int(nodes[b]), weight=float(1 - sub[a, b]))
        if nx.is_connected(g):
            break
    for node in nodes:
        for c, p in enumerate(probs[node]):
            peak = int(root_hits[c])
            if p >= pull_min and peak != node:
                g.add_edge(int(node), peak, weight=pull_weight * float(p))
    return nx.spring_layout(g, weight="weight", iterations=300, seed=seed)
