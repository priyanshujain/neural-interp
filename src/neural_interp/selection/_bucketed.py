from typing import Self

import numpy as np

from neural_interp.selection._elbow import ElbowSelector
from neural_interp.typing import FloatArray, IntArray


class BucketedSelector:
    def __init__(
        self,
        budget: int,
        n_bands: int = 10,
        floor: float | None = None,
        min_band: int | None = None,
        seed: int = 0,
    ) -> None:
        self.budget = budget
        self.n_bands = n_bands
        self.floor = floor
        self.min_band = min_band
        self.seed = seed

    def fit(self, values: FloatArray) -> Self:
        rng = np.random.default_rng(self.seed)
        floor = self.floor if self.floor is not None else ElbowSelector().fit(values).threshold_
        candidates = np.flatnonzero(values >= floor)
        edges = np.linspace(floor, float(values.max()), self.n_bands + 1)
        band = np.clip(
            np.searchsorted(edges, values[candidates], side="right") - 1, 0, self.n_bands - 1
        )
        min_band = self.min_band or max(1, self.budget // (4 * self.n_bands))
        merged: list[IntArray] = []
        for b in reversed(range(self.n_bands)):
            members = candidates[band == b]
            if merged and len(merged[-1]) < min_band:
                merged[-1] = np.concatenate([merged[-1], members])
            else:
                merged.append(members)
        if len(merged) > 1 and len(merged[-1]) < min_band:
            merged[-2] = np.concatenate([merged[-2], merged.pop()])
        share = self.budget // len(merged)
        picked = [m if len(m) <= share else rng.choice(m, share, replace=False) for m in merged]
        chosen = np.concatenate(picked)
        self.indices_: IntArray = chosen[np.argsort(-values[chosen], kind="stable")].astype(
            np.int64
        )
        self.threshold_: float = float(floor)
        return self
