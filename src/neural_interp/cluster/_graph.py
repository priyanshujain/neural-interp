import numpy as np
from scipy.sparse import csr_matrix

from neural_interp.typing import FloatArray, IntArray


def knn_indices(distances: FloatArray, k: int) -> IntArray:
    out = np.empty((len(distances), k), dtype=np.int64)
    for i in range(len(distances)):
        row = np.array(distances[i], dtype=np.float64)
        row[i] = np.inf
        nearest = np.argpartition(row, k)[:k]
        out[i] = nearest[np.argsort(row[nearest], kind="stable")]
    return out


def knn_graph(distances: FloatArray, k: int, mutual: bool = False) -> csr_matrix:
    n = len(distances)
    rows = np.repeat(np.arange(n), k)
    cols = knn_indices(distances, k).ravel()
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    linked = (adjacency.multiply(adjacency.T) if mutual else adjacency.maximum(adjacency.T)).tocoo()
    weights = np.maximum(np.asarray(distances)[linked.row, linked.col], 1e-9)
    return csr_matrix((weights, (linked.row, linked.col)), shape=(n, n))
