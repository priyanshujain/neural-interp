import json
from pathlib import Path

import numpy as np

from neural_interp.space import HitSpace
from neural_interp.vision.labels import group


def reference_hits(space_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    hits = np.load(space_dir / "hits.npz", allow_pickle=True)
    distinct = np.load(space_dir / "forest.npz")["distinct_of_hit"]
    probs = np.load(space_dir / "probabilities.npy")
    ids = [f"{hits['sample_ids'][b]}:{t}" for b, t in zip(hits["batch_idx"], hits["token_idx"])]
    return ids, probs[distinct].argmax(1), hits["activations"]


def worlds(ids: list[str], cluster: np.ndarray) -> dict:
    groups = np.array([group(h.split("/")[0]) for h in ids])
    clusters = []
    for c in np.unique(cluster):
        members = groups[cluster == c]
        names, counts = np.unique(members, return_counts=True)
        top = int(np.argmax(counts))
        clusters.append({
            "cluster": int(c), "size": int(len(members)),
            "dominant": str(names[top]), "purity": float(counts[top] / len(members)),
            "groups": dict(zip(names.tolist(), counts.tolist())),
        })
    sizes = np.array([c["size"] for c in clusters])
    purity = np.array([c["purity"] for c in clusters])
    mixed_cat_vehicle = [
        c["cluster"] for c in clusters
        if min(c["groups"].get("cat", 0), c["groups"].get("vehicle", 0)) > 0.1 * c["size"]
    ]
    return {
        "hits": len(ids),
        "clusters": len(clusters),
        "group_share": {g: float((groups == g).mean()) for g in np.unique(groups)},
        "weighted_purity": float((sizes * purity).sum() / sizes.sum()),
        "clusters_mixing_cat_and_vehicle": mixed_cat_vehicle,
        "per_cluster": sorted(clusters, key=lambda c: -c["size"]),
    }


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda x: (x * (x - 1) / 2).sum()
    both, rows, cols, n = pairs(table), pairs(table.sum(1)), pairs(table.sum(0)), pairs(np.array([len(a)]))
    expected = rows * cols / n
    return float((both - expected) / ((rows + cols) / 2 - expected))


def compare(ours_dir: Path, reference_dir: Path, out: Path) -> dict:
    ours = HitSpace.load(ours_dir)
    ref_ids, ref_tree, _ = reference_hits(reference_dir)
    shared = sorted(set(ours.hit_ids) & set(ref_ids))
    ours_index = {h: i for i, h in enumerate(ours.hit_ids)}
    ref_index = {h: i for i, h in enumerate(ref_ids)}
    a = ours.cluster[[ours_index[h] for h in shared]]
    b = ref_tree[[ref_index[h] for h in shared]]
    result = {
        "shared_hits": len(shared),
        "ari_on_shared": adjusted_rand(a, b) if shared else None,
        "ours": worlds(ours.hit_ids, ours.cluster),
        "reference": worlds(ref_ids, ref_tree),
    }
    out.write_text(json.dumps(result, indent=1))
    return result
