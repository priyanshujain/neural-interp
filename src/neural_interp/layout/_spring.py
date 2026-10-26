from typing import Self

import networkx as nx
import numpy as np

from neural_interp.buckets import Bucket, ranked_sample
from neural_interp.typing import FloatArray, IntArray


def sample_nodes(
    buckets: list[Bucket], probabilities: FloatArray, anchors: IntArray, per_bucket: int = 8
) -> IntArray:
    picked = [ranked_sample(b, probabilities, per_bucket) for b in buckets]
    return np.unique(np.concatenate(picked + [anchors])).astype(np.int64)


class PulledSpringLayout:
    def __init__(
        self,
        ks: tuple[int, ...] = (6, 10, 15, 25),
        pull_min: float = 0.1,
        pull_weight: float = 3.0,
        iterations: int = 300,
        seed: int = 0,
    ) -> None:
        self.ks = ks
        self.pull_min = pull_min
        self.pull_weight = pull_weight
        self.iterations = iterations
        self.seed = seed

    def fit(
        self, distances: FloatArray, probabilities: FloatArray, nodes: IntArray, anchors: IntArray
    ) -> Self:
        sub = np.asarray(distances)[np.ix_(nodes, nodes)]
        masked = sub + np.diag(np.full(len(nodes), np.inf))
        graph = nx.Graph()
        for k in self.ks:
            graph = nx.Graph()
            graph.add_nodes_from(int(n) for n in nodes)
            for a, row in enumerate(np.argsort(masked, axis=1)[:, :k]):
                for b in row:
                    graph.add_edge(int(nodes[a]), int(nodes[b]), weight=float(1 - sub[a, b]))
            if nx.is_connected(graph):
                break
        for node in nodes:
            for c, p in enumerate(probabilities[node]):
                anchor = int(anchors[c])
                if p >= self.pull_min and anchor != node:
                    graph.add_edge(int(node), anchor, weight=self.pull_weight * float(p))
        layout = nx.spring_layout(
            graph, weight="weight", iterations=self.iterations, seed=self.seed
        )
        self.positions_: dict[int, tuple[float, float]] = {
            int(n): (float(xy[0]), float(xy[1])) for n, xy in layout.items()
        }
        return self
