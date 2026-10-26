import numpy as np


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(x: np.ndarray) -> float:
        return float((x * (x - 1) / 2).sum())

    both, rows, cols, n = pairs(table), pairs(table.sum(1)), pairs(table.sum(0)), pairs(np.array([len(a)]))
    expected = rows * cols / n
    return (both - expected) / ((rows + cols) / 2 - expected)


def composition(labels: np.ndarray, cluster: np.ndarray) -> list[dict]:
    rows = []
    for c in np.unique(cluster):
        names, counts = np.unique(labels[cluster == c], return_counts=True)
        top = int(np.argmax(counts))
        rows.append({
            "cluster": int(c), "size": int(counts.sum()), "dominant": str(names[top]),
            "purity": float(counts[top] / counts.sum()), "labels": dict(zip(names.tolist(), counts.tolist())),
        })
    return sorted(rows, key=lambda r: -r["size"])


def weighted_purity(rows: list[dict]) -> float:
    sizes = np.array([r["size"] for r in rows])
    return float((sizes * np.array([r["purity"] for r in rows])).sum() / sizes.sum())
