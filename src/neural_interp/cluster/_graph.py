import numpy as np
import torch
from scipy.sparse import csr_matrix

from neural_interp._device import default_device
from neural_interp.typing import FloatArray, IntArray


def nearest(
    distances: FloatArray, k: int, device: str | None = None, chunk: int = 2048
) -> tuple[FloatArray, IntArray]:
    device = device or default_device()
    n = len(distances)
    values = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk):
        block = torch.as_tensor(np.array(distances[start : start + chunk]), device=device).float()
        rows = torch.arange(len(block), device=device)
        block[rows, rows + start] = torch.inf
        found = torch.topk(block, k, dim=1, largest=False, sorted=True)
        values[start : start + chunk] = found.values.cpu().numpy()
        indices[start : start + chunk] = found.indices.cpu().numpy()
    return values, indices


def knn_indices(distances: FloatArray, k: int) -> IntArray:
    return nearest(distances, k)[1]


def knn_graph(distances: FloatArray, k: int, mutual: bool = False) -> csr_matrix:
    n = len(distances)
    rows = np.repeat(np.arange(n), k)
    cols = knn_indices(distances, k).ravel()
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    linked = (adjacency.multiply(adjacency.T) if mutual else adjacency.maximum(adjacency.T)).tocoo()
    weights = np.maximum(np.asarray(distances)[linked.row, linked.col], 1e-9)
    return csr_matrix((weights, (linked.row, linked.col)), shape=(n, n))
