from dataclasses import dataclass

import numpy as np

from neural_interp.metrics import signed_split
from neural_interp.typing import FloatArray, IntArray


@dataclass
class Copies:
    points: FloatArray
    counts: IntArray
    point_of: IntArray


def collapse_copies(rows: FloatArray, decimals: int = 6) -> Copies:
    keys = np.round(signed_split(rows), decimals)
    _, first, inverse, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return Copies(
        rows[first[order]], counts[order].astype(np.int64), remap[inverse.ravel()].astype(np.int64)
    )
