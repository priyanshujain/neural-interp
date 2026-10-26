from dataclasses import dataclass

from markupsafe import Markup


@dataclass(frozen=True)
class ClusterRef:
    id: int
    color: str


@dataclass(frozen=True)
class Part:
    name: str
    share: float
    color: str


@dataclass(frozen=True)
class HitPattern:
    id: str
    pattern: str


@dataclass(frozen=True)
class TileView:
    number: int
    id: str
    media: Markup
    label: str
    activation: float
    entropy: float
    pattern: str
    probabilities: list[tuple[ClusterRef, float]]


@dataclass(frozen=True)
class ClusterCard:
    cluster: ClusterRef
    size: int
    sources: int
    composition: list[Part]
    low: float
    high: float
    median: float
    confident: float
    self_contained: float
    neighbours: list[tuple[ClusterRef, float]]
    pattern: str
    bucket_links: list[tuple[float, str]]
    strip: Markup
    patterns: list[HitPattern]


@dataclass(frozen=True)
class GroupCard:
    name: str
    href: str
    size: int
    clusters: list[ClusterRef]
    composition: list[Part]
    strip: Markup
    patterns: list[HitPattern]


@dataclass(frozen=True)
class BucketRow:
    name: str
    href: str
    size: int


@dataclass(frozen=True)
class TreePick:
    id: int
    color: str
    on: bool


@dataclass(frozen=True)
class UpClose:
    per_tree: int
    picks: list[TreePick]
    figure: Markup
