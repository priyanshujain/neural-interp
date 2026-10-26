from dataclasses import dataclass

import numpy as np

from neural_interp.typing import FloatArray, IntArray


@dataclass
class ClusterComposition:
    cluster: int
    size: int
    dominant: str
    purity: float
    labels: dict[str, int]


def adjusted_rand(a: IntArray, b: IntArray) -> float:
    _, a_codes = np.unique(a, return_inverse=True)
    _, b_codes = np.unique(b, return_inverse=True)
    table = np.zeros((int(a_codes.max()) + 1, int(b_codes.max()) + 1))
    np.add.at(table, (a_codes, b_codes), 1)

    def pairs(x: FloatArray) -> float:
        return float((x * (x - 1) / 2).sum())

    both, rows, cols = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    n = pairs(np.array([len(a_codes)], dtype=np.float64))
    expected = rows * cols / n
    return (both - expected) / ((rows + cols) / 2 - expected)


def composition(labels: np.ndarray, cluster: IntArray) -> list[ClusterComposition]:
    out: list[ClusterComposition] = []
    for c in np.unique(cluster):
        names, counts = np.unique(labels[cluster == c], return_counts=True)
        top = int(np.argmax(counts))
        out.append(
            ClusterComposition(
                cluster=int(c),
                size=int(counts.sum()),
                dominant=str(names[top]),
                purity=float(counts[top] / counts.sum()),
                labels={str(k): int(v) for k, v in zip(names, counts)},
            )
        )
    return sorted(out, key=lambda r: -r.size)


def weighted_purity(rows: list[ClusterComposition]) -> float:
    sizes = np.array([r.size for r in rows], dtype=np.float64)
    return float((sizes * np.array([r.purity for r in rows])).sum() / sizes.sum())
