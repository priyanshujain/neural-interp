from dataclasses import dataclass

import numpy as np

from neural_interp.typing import FloatArray, IntArray


@dataclass
class Bucket:
    name: str
    clusters: list[int]
    hits: IntArray
    threshold: float


def cluster_buckets(
    probabilities: FloatArray, thresholds: tuple[float, ...] = (0.9, 0.8, 0.7)
) -> list[Bucket]:
    out: list[Bucket] = []
    for c in range(probabilities.shape[1]):
        for t in thresholds:
            hits = np.flatnonzero(probabilities[:, c] >= t).astype(np.int64)
            if len(hits):
                out.append(Bucket(f"cluster {c} >= {t}", [c], hits, t))
    return out


def covering_set(p: FloatArray, mass: float) -> tuple[int, ...]:
    order = np.argsort(-p, kind="stable")
    needed = int(np.searchsorted(np.cumsum(p[order]), mass - 1e-12)) + 1
    return tuple(sorted(int(c) for c in order[:needed]))


def set_buckets(probabilities: FloatArray, mass: float = 0.85, min_hits: int = 1) -> list[Bucket]:
    limit = probabilities.shape[1] / 2
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, p in enumerate(probabilities):
        key = covering_set(p, mass)
        if 1 < len(key) <= limit:
            groups.setdefault(key, []).append(i)
    out = [
        Bucket(
            "clusters " + " + ".join(map(str, key)), list(key), np.array(hits, dtype=np.int64), mass
        )
        for key, hits in groups.items()
        if len(hits) >= min_hits
    ]
    return sorted(out, key=lambda b: -len(b.hits))


def leading_cluster(bucket: Bucket, probabilities: FloatArray) -> int:
    mass = probabilities[np.ix_(bucket.hits, bucket.clusters)].sum(0)
    return int(bucket.clusters[int(np.argmax(mass))])


def ranked_sample(bucket: Bucket, probabilities: FloatArray, n: int) -> IntArray:
    lead = probabilities[bucket.hits, leading_cluster(bucket, probabilities)]
    ranked = bucket.hits[np.argsort(-lead, kind="stable")]
    if len(ranked) <= n:
        return ranked
    return ranked[np.unique(np.linspace(0, len(ranked) - 1, n).round().astype(np.int64))]
