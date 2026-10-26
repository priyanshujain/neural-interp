from pathlib import Path

import numpy as np
import torch
from in_tfm.density_peaks import log_density
from in_tfm.hit_selection.strategies.elbow import find_elbow_index_in_sorted_data

from neural_interp.lib import gemma
from neural_interp.lib.forest import cluster, peak_pull_layout
from neural_interp.lib.geometry import nan_free, other_tree, tree_percentile, unit
from neural_interp.lib.report import save_json

K = 10


def run(dim, sign, out):
    dim, out = int(dim), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok = gemma.tokenizer()
    _, W = gemma.embedding()
    W = W[:len(tok)]
    pieces, is_unused, is_special = gemma.token_masks(tok, len(W))
    kept = np.flatnonzero(~is_unused & ~is_special)

    values = W[kept, dim] * (1 if sign == "pos" else -1)
    order = np.argsort(-values)
    n_positive = int((values > 0).sum())
    curve = values[order[:n_positive]]
    n_hits = find_elbow_index_in_sorted_data(torch.from_numpy(curve.copy())) + 1
    curve_ranks = np.unique(np.concatenate([np.arange(min(n_positive, 3 * n_hits)), np.linspace(0, n_positive - 1, 2000).astype(int)]))
    hits = kept[order[:n_hits]]
    print(f"dim {dim} {sign}: {n_hits} hits above the elbow of {n_positive} positive of {len(kept)} tokens, "
          f"value from {values[order[0]]:.4f} to {values[order[n_hits - 1]]:.4f}", flush=True)

    rows = unit(W[hits] - W[kept].mean(0))
    forest, trees = cluster(rows, K, input_order=True)
    print("trees", len(trees.roots), "cuts", trees.cuts, flush=True)
    xy, probabilities = peak_pull_layout(rows, forest, trees, per_tree=len(rows), max_nodes=4000)

    tree = trees.tree
    rho = log_density(forest.kth_distance)
    side = "highest" if sign == "pos" else "lowest"
    save_json(out / "single_dim.json", {
        "intro": f"Dimension {dim} of Gemma 3 270M's W_E, {side} values. The values form a signed bell curve with no ReLU, "
                 f"so the elbow is taken over the {n_positive:,} tokens on this side of zero, as a ReLU would leave them. "
                 f"The hits are the {n_hits:,} tokens above it, as with mixed4e:55. Each hit is its full embedding row, centred on the vocabulary "
                 f"mean and L2-normalised, so distance is cosine. In-tfm density-peak forest (k={K}, elbow root cuts) gives "
                 f"{len(trees.roots)} trees. The map is in-tfm's bucket-graph layout: nearest-neighbour edges in the real "
                 f"distance plus pulls toward the peaks of the trees each hit has probability for, spring-laid out. "
                 f"Numbered badges are peaks, points are coloured by their most probable cluster.",
        "n_points": int(n_hits),
        "elbow": {"rank": curve_ranks.tolist(), "value": curve[curve_ranks].round(5).tolist(), "n_hits": int(n_hits),
                  "n_positive": n_positive, "n_tokens": int(len(kept)), "cut": round(float(curve[n_hits - 1]), 5)},
        "cuts": {"rho": trees.cuts.rho, "delta": trees.cuts.delta},
        "distinct": {
            "token": [gemma.show(pieces[i]) for i in hits], "count": [1] * n_hits, "value": values[order[:n_hits]].round(4).tolist(),
            "tree": tree.tolist(), "label": probabilities.argmax(1).tolist(), "parent": forest.parent.tolist(),
            "depth": trees.depth.tolist(), "rho": rho.round(4).tolist(), "delta": forest.delta.round(4).tolist(),
            "tree_pct": tree_percentile(tree, rho).round(4).tolist(), "lid": [0.0] * n_hits, "mutual": [0.0] * n_hits,
            "agree": (1 - other_tree(tree, forest.neighbours).mean(1)).round(3).tolist(), "neighbours": forest.neighbours.tolist(),
            "x": nan_free(xy[:, 0]), "y": nan_free(xy[:, 1]),
        },
        "roots": trees.roots.tolist(), "tree_hits": np.bincount(tree).tolist(),
        "occurrence": {"distinct": [], "window": [], "position": [], "context": []},
    })
    for t in range(min(20, len(trees.roots))):
        members = np.flatnonzero(tree == t)
        densest = members[np.argsort(-rho[members])][:14]
        print(f"tree {t}: {len(members)} hits:", " ".join(gemma.show(pieces[hits[i]]) for i in densest))
