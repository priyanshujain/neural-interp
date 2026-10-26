import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from neural_interp.buckets import cluster_buckets, set_buckets
from neural_interp.density_peaks import build_forest, collapse_copies
from neural_interp.distance import weighted_jaccard
from neural_interp.layout import hit_map, map_nodes
from neural_interp.membership import soft_membership
from neural_interp.report import write_report
from neural_interp.selection import bucketed_hits, elbow_hits
from neural_interp.space import HitSpace, summary
from neural_interp.vision import inception
from neural_interp.vision.cards import overlay, save_card
from neural_interp.vision.labels import group, synset_name


@dataclass
class Config:
    pool: Path
    out: Path
    n_images: int = 2000
    image_list: Path | None = None
    seed: int = 0
    selection: str = "elbow"
    budget: int = 14000
    k: int = 5
    peak_share: float = 0.8
    max_entropy: float = 0.5
    per_bucket: int = 40
    min_set_bucket: int = 10


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hit_id(paths: list[Path], pool: Path, flat: int, grid: tuple[int, int]) -> str:
    image, pos = divmod(int(flat), grid[0] * grid[1])
    return f"{paths[image].relative_to(pool).as_posix()}:{pos}"


def collect(cfg: Config) -> HitSpace:
    dev = device()
    model = inception.load_model(dev)
    probe = inception.Probe(model, inception.MIXED4E_55)
    if cfg.image_list:
        paths = [cfg.pool / line for line in cfg.image_list.read_text().split()]
    else:
        paths = inception.sample_images(cfg.pool, cfg.n_images, cfg.seed)
    activations = inception.collect_activations(probe, paths, dev)
    grid = activations.shape[1:]
    flat = activations.ravel()
    hits = elbow_hits(flat) if cfg.selection == "elbow" else bucketed_hits(flat, cfg.budget, seed=cfg.seed)
    print(f"{len(flat)} activations, {len(hits)} hits ({cfg.selection})")

    rows, preactivation = inception.collect_hadamard(probe, paths, hits, grid, dev)
    error = np.abs(rows.sum(1) - preactivation).max()
    print(f"hadamard rows sum to pre-activation minus bias, max error {error:.2e}")

    unique, counts, point = collapse_copies(rows)
    dist = weighted_jaccard(unique)
    forest = build_forest(dist, counts, cfg.k)
    membership = soft_membership(dist, forest, cfg.k, cfg.peak_share)
    space = HitSpace(
        [hit_id(paths, cfg.pool, h, grid) for h in hits], flat[hits], rows, point, counts, dist, forest, membership
    )
    space_dir = cfg.out / "space"
    space.save(space_dir)
    (space_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=1))
    (space_dir / "summary.json").write_text(json.dumps(summary(space), indent=1))
    print(json.dumps({k: v for k, v in summary(space).items() if k != "cluster_sizes"}))
    return space


def wnid(hit: str) -> str:
    return hit.split("/")[0]


def composition(space: HitSpace) -> list[dict]:
    groups = np.array([group(wnid(h)) for h in space.hit_ids])
    rows = []
    for c in range(len(space.forest.roots)):
        members = space.cluster == c
        names, counts = np.unique(groups[members], return_counts=True)
        rows.append({"cluster": c, "size": int(members.sum()), **dict(zip(names.tolist(), counts.tolist()))})
    return rows


def report(cfg: Config) -> None:
    space = HitSpace.load(cfg.out / "space")
    dist_hits = np.asarray(space.dist)[np.ix_(space.point, space.point)]
    confident = space.entropy < cfg.max_entropy
    buckets = cluster_buckets(space.probs)
    buckets += set_buckets(space.probs, min_hits=cfg.min_set_bucket)
    for b in buckets:
        b.hits = b.hits[confident[b.hits]] if len(b.clusters) == 1 else b.hits

    peak_hits = np.array([int(np.flatnonzero(space.point == r)[0]) for r in space.forest.roots])
    nodes = map_nodes(dist_hits, buckets, peak_hits, cfg.per_bucket)
    positions = hit_map(dist_hits, space.probs, nodes, peak_hits)

    dev = device()
    model = inception.load_model(dev)
    cards_dir = cfg.out / "report" / "cards"
    attribution_of = inception.NeuronAttribution(model, inception.MIXED4E_55)

    def card(hit: int) -> str:
        path, pos = space.hit_ids[hit].rsplit(":", 1)
        grid = 14
        y, x = divmod(int(pos), grid)
        pixels = inception.preprocess(cfg.pool / path)
        attribution = attribution_of.map(pixels.to(dev), y, x)
        src = save_card(cards_dir, f"hit_{hit}", pixels.numpy(), overlay(pixels.numpy(), attribution, y, x, grid))
        return f'<img src="{src}" width="448"><div>{synset_name(wnid(space.hit_ids[hit]))}</div>'

    write_report(
        cfg.out / "report", space, buckets, positions, card, group_of=lambda h: group(wnid(space.hit_ids[h]))
    )
    (cfg.out / "report" / "composition.json").write_text(json.dumps(composition(space), indent=1))
