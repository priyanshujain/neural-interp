import numpy as np


def elbow_index(sorted_desc: np.ndarray) -> int:
    y = np.asarray(sorted_desc, dtype=np.float64)
    n = len(y)
    if n < 3:
        return n - 1
    x = np.arange(n, dtype=np.float64)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    distance = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / np.hypot(dx, dy)
    return int(np.argmax(distance))


def elbow_hits(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    return order[: elbow_index(values[order]) + 1]


def bucketed_hits(
    values: np.ndarray,
    budget: int,
    n_bands: int = 10,
    floor: float | None = None,
    min_band: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if floor is None:
        floor = values[elbow_hits(values)[-1]]
    candidates = np.flatnonzero(values >= floor)
    edges = np.linspace(floor, values.max(), n_bands + 1)
    band = np.clip(np.searchsorted(edges, values[candidates], side="right") - 1, 0, n_bands - 1)
    bands = [candidates[band == b] for b in range(n_bands)]
    min_band = min_band or max(1, budget // (4 * n_bands))
    merged: list[np.ndarray] = []
    for members in reversed(bands):
        if merged and len(merged[-1]) < min_band:
            merged[-1] = np.concatenate([merged[-1], members])
        else:
            merged.append(members)
    if len(merged) > 1 and len(merged[-1]) < min_band:
        merged[-2] = np.concatenate([merged[-2], merged.pop()])
    share = budget // len(merged)
    picked = [m if len(m) <= share else rng.choice(m, share, replace=False) for m in merged]
    chosen = np.concatenate(picked)
    return chosen[np.argsort(-values[chosen], kind="stable")]
