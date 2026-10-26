from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bucket:
    name: str
    clusters: list[int]
    hits: np.ndarray
    threshold: float = field(default=0.0)


def cluster_buckets(probs: np.ndarray, thresholds=(0.9, 0.8, 0.7)) -> list[Bucket]:
    out = []
    for c in range(probs.shape[1]):
        for t in thresholds:
            hits = np.flatnonzero(probs[:, c] >= t)
            if len(hits):
                out.append(Bucket(f"cluster {c} >= {t}", [c], hits, t))
    return out


def covering_set(p: np.ndarray, mass: float) -> tuple[int, ...]:
    order = np.argsort(-p, kind="stable")
    needed = int(np.searchsorted(np.cumsum(p[order]), mass - 1e-12)) + 1
    return tuple(sorted(int(c) for c in order[:needed]))


def set_buckets(probs: np.ndarray, mass: float = 0.85, min_hits: int = 1) -> list[Bucket]:
    limit = probs.shape[1] / 2
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, p in enumerate(probs):
        key = covering_set(p, mass)
        if 1 < len(key) <= limit:
            groups.setdefault(key, []).append(i)
    out = [
        Bucket("clusters " + " + ".join(map(str, key)), list(key), np.array(hits), mass)
        for key, hits in groups.items()
        if len(hits) >= min_hits
    ]
    return sorted(out, key=lambda b: -len(b.hits))


def leading_cluster(bucket: Bucket, probs: np.ndarray) -> int:
    return int(bucket.clusters[np.argmax(probs[np.ix_(bucket.hits, bucket.clusters)].sum(0))])


def ranked_sample(bucket: Bucket, probs: np.ndarray, n: int) -> np.ndarray:
    lead = probs[bucket.hits, leading_cluster(bucket, probs)]
    ranked = bucket.hits[np.argsort(-lead, kind="stable")]
    if len(ranked) <= n:
        return ranked
    return ranked[np.unique(np.linspace(0, len(ranked) - 1, n).round().astype(int))]
