from typing import Literal, Self

import numpy as np

from neural_interp.typing import FloatArray, IntArray


def elbow_index(sorted_desc: FloatArray) -> int:
    y = np.asarray(sorted_desc, dtype=np.float64)
    n = len(y)
    if n < 3:
        return n - 1
    x = np.arange(n, dtype=np.float64)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    gap = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / np.hypot(dx, dy)
    return int(np.argmax(gap))


class ElbowSelector:
    def __init__(self, side: Literal["positive", "all"] = "positive") -> None:
        self.side = side

    def fit(self, values: FloatArray) -> Self:
        pool = np.flatnonzero(values > 0) if self.side == "positive" else np.arange(len(values))
        order = pool[np.argsort(-values[pool], kind="stable")]
        self.indices_: IntArray = order[: elbow_index(values[order]) + 1].astype(np.int64)
        self.threshold_: float = float(values[self.indices_[-1]])
        return self
