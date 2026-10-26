import numpy as np
import torch

from neural_interp.typing import Float32Array, FloatArray


def signed_split(rows: FloatArray) -> FloatArray:
    return np.concatenate([np.maximum(rows, 0), np.maximum(-rows, 0)], axis=1)


def weighted_jaccard(
    rows: FloatArray, device: str | None = None, chunk: int = 1024
) -> Float32Array:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(signed_split(rows), dtype=torch.float32, device=device)
    mass = x.sum(1)
    n = len(x)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk):
        block = x[start : start + chunk]
        l1 = torch.cdist(block, x, p=1)
        total = mass[start : start + chunk, None] + mass[None, :]
        shared = (total - l1) / 2
        union = (total + l1) / 2
        distance = 1 - shared / union.clamp_min(1e-12)
        out[start : start + chunk] = distance.clamp(0, 1).cpu().numpy()
    np.fill_diagonal(out, 0)
    return out
