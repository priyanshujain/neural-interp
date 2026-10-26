import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from neural_interp.buckets import Bucket
from neural_interp.cluster import knn_graph
from neural_interp.report._hits import HitView
from neural_interp.report._render import PALETTE, color
from neural_interp.report._views import ClusterRef, HitPattern, Part
from neural_interp.space import HitSpace
from neural_interp.typing import FloatArray

Positions = dict[int, tuple[float, float]]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bucket"


def ref(cluster: int) -> ClusterRef:
    return ClusterRef(int(cluster), color(cluster))


def source_of(hit_id: str) -> str:
    return hit_id.rsplit(":", 1)[0]


def pattern_png(row: np.ndarray, path: Path, cols: int = 24, cell: int = 6) -> None:
    n = len(row)
    rows = -(-n // cols)
    grid = np.zeros(rows * cols)
    grid[:n] = row
    scale = np.percentile(np.abs(grid[:n]), 99) or 1.0
    v = grid.reshape(rows, cols) / scale
    v = np.sign(v) * np.sqrt(np.clip(np.abs(v), 0, 1))
    base = np.array([0.15, 0.16, 0.19])
    rgb = base + np.clip(v, 0, 1)[..., None] * (np.array([0.35, 0.95, 0.4]) - base)
    rgb = rgb + np.clip(-v, 0, 1)[..., None] * (np.array([0.95, 0.3, 0.3]) - base)
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    image.resize((cols * cell, rows * cell), Image.Resampling.NEAREST).save(path)


@dataclass(frozen=True)
class LinkedBucket:
    bucket: Bucket
    href: str


def link_buckets(buckets: list[Bucket]) -> list[LinkedBucket]:
    linked: list[LinkedBucket] = []
    used: set[str] = set()
    for b in buckets:
        name = slug(b.name)
        while name in used:
            name += "-x"
        used.add(name)
        linked.append(LinkedBucket(b, f"{name}.html"))
    return linked


@dataclass
class ReportContext:
    space: HitSpace
    hit_view: HitView
    unit: str
    all_activations: FloatArray | None
    groups: np.ndarray | None
    group_colors: dict[str, str]
    probabilities: np.ndarray
    clusters: np.ndarray
    activation: np.ndarray
    entropy: np.ndarray
    sizes: np.ndarray
    contacts: np.ndarray
    buckets: list[LinkedBucket] = field(default_factory=lambda: [])
    positions: Positions = field(default_factory=lambda: {})
    out: Path = Path(".")

    @classmethod
    def build(
        cls,
        space: HitSpace,
        hit_view: HitView,
        group_of: Callable[[int], str] | None = None,
        group_colors: dict[str, str] | None = None,
        all_activations: FloatArray | None = None,
        unit: str = "unit",
    ) -> "ReportContext":
        probabilities = np.asarray(space.probabilities)
        clusters = np.asarray(space.cluster)
        n_clusters = probabilities.shape[1]
        groups = np.array([group_of(h) for h in range(len(space.hit_ids))]) if group_of else None
        names = sorted(set(groups.tolist())) if groups is not None else []
        modes = np.asarray(space.membership.probabilities_).argmax(1)
        edges = knn_graph(np.asarray(space.distances), 5, mutual=True).tocoo()
        contacts = np.zeros((n_clusters, n_clusters))
        np.add.at(contacts, (modes[edges.row], modes[edges.col]), 1)
        return cls(
            space=space,
            hit_view=hit_view,
            unit=unit,
            all_activations=all_activations,
            groups=groups,
            group_colors={n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
            | (group_colors or {}),
            probabilities=probabilities,
            clusters=clusters,
            activation=np.asarray(space.activation),
            entropy=np.asarray(space.entropy),
            sizes=np.bincount(clusters, minlength=n_clusters),
            contacts=contacts,
        )

    @property
    def n_clusters(self) -> int:
        return int(self.probabilities.shape[1])

    def clusters_by_size(self) -> list[int]:
        return [int(c) for c in np.argsort(-self.sizes, kind="stable")]

    def group_color(self, name: str) -> str:
        return self.group_colors.get(name, "#aaaaaa")

    def composition(self, members: np.ndarray) -> list[Part]:
        if self.groups is None or not len(members):
            return []
        names, counts = np.unique(self.groups[members], return_counts=True)
        share = counts / counts.sum()
        return [Part(str(n), float(s), self.group_color(str(n))) for n, s in zip(names, share)]

    def pattern(self, hit: int) -> str:
        path = self.out / "patterns" / f"hit_{hit}.png"
        if not path.exists():
            pattern_png(np.asarray(self.space.rows[hit]), path)
        return f"patterns/{path.name}"

    def cluster_pattern(self, cluster: int) -> str:
        path = self.out / "patterns" / f"cluster_{cluster}.png"
        if not path.exists():
            pattern_png(np.asarray(self.space.rows[self.clusters == cluster]).mean(0), path)
        return f"patterns/{path.name}"

    def hit_patterns(self, hits: list[int]) -> list[HitPattern]:
        return [HitPattern(self.space.hit_ids[h], self.pattern(h)) for h in hits]
