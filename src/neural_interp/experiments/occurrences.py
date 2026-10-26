from pathlib import Path

import numpy as np
from in_tfm.density_peaks import log_density
from sklearn.neighbors import NearestNeighbors

from neural_interp.lib import gemma
from neural_interp.lib.forest import cluster, peak_pull_layout
from neural_interp.lib.geometry import local_intrinsic_dim, mutual_knn, nan_free, other_tree, tree_percentile, unit
from neural_interp.lib.report import save_json

N_WINDOWS = 10
WINDOW = 5_000
N_POINTS = 14_240
K = 10
K_LID = 20
CONTEXT = 10


def sample_windows(tok, rng):
    text = gemma.wikitext()
    starts = np.sort(rng.choice(len(text) - 20_000, N_WINDOWS, replace=False))
    windows = []
    for s in starts:
        ids, line = [], s
        while len(ids) < WINDOW:
            ids += tok(text[line], add_special_tokens=False)["input_ids"]
            line += 1
        windows.append(np.array(ids[:WINDOW]))
        print("window at line", s, "starts:", repr(tok.decode(ids[:30])), flush=True)
    return starts, np.stack(windows)


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tok = gemma.tokenizer()
    _, W = gemma.embedding()
    starts, stream = sample_windows(tok, rng)

    flat = rng.choice(N_WINDOWS * WINDOW, N_POINTS, replace=False)
    window_of, position = flat // WINDOW, flat % WINDOW
    token_ids = stream[window_of, position]
    print("occurrences", N_POINTS, "distinct tokens", len(np.unique(token_ids)), flush=True)

    points = W[token_ids] - W.mean(0)
    forest, trees = cluster(points, K)
    tree = trees.tree
    print("distinct points", len(forest.counts), "trees", len(trees.roots), "cuts", trees.cuts, flush=True)

    distinct = unit(points[np.unique(forest.distinct_of_hit, return_index=True)[1]])
    distance, _ = NearestNeighbors(n_neighbors=K_LID + 1).fit(distinct).kneighbors(distinct)
    local_id = local_intrinsic_dim(distance[:, 1:])
    mutual = mutual_knn(forest.neighbours)
    rho = log_density(forest.kth_distance)
    sizes = np.bincount(tree, weights=forest.counts).astype(int)
    xy, _ = peak_pull_layout(distinct, forest, trees, per_tree=12)

    pieces = tok.convert_ids_to_tokens(list(range(len(W))))
    distinct_token = np.empty(len(tree), dtype=int)
    distinct_token[forest.distinct_of_hit] = token_ids
    contexts = []
    for w, p in zip(window_of, position):
        lo, hi = max(0, p - CONTEXT), min(WINDOW, p + CONTEXT + 1)
        contexts.append([tok.decode(stream[w, lo:p]), tok.decode(stream[w, p:p + 1]), tok.decode(stream[w, p + 1:hi])])

    save_json(out / "occurrences.json", {
        "windows": [int(s) for s in starts],
        "n_points": N_POINTS,
        "cuts": {"rho": trees.cuts.rho, "delta": trees.cuts.delta},
        "distinct": {
            "token": [gemma.show(pieces[i]) for i in distinct_token], "count": forest.counts.tolist(),
            "tree": tree.tolist(), "parent": forest.parent.tolist(), "depth": trees.depth.tolist(),
            "rho": rho.round(4).tolist(), "delta": forest.delta.round(4).tolist(),
            "tree_pct": tree_percentile(tree, rho).round(4).tolist(), "lid": local_id.round(2).tolist(),
            "mutual": mutual.round(3).tolist(), "agree": (1 - other_tree(tree, forest.neighbours).mean(1)).round(3).tolist(),
            "neighbours": forest.neighbours.tolist(), "x": nan_free(xy[:, 0]), "y": nan_free(xy[:, 1]),
        },
        "roots": trees.roots.tolist(), "tree_hits": sizes.tolist(),
        "occurrence": {"distinct": forest.distinct_of_hit.tolist(), "window": window_of.tolist(),
                       "position": position.tolist(), "context": contexts},
    })
    print("median local id", round(float(np.median(local_id)), 1), "mutual", round(float(mutual.mean()), 3))
    for t in range(min(15, len(trees.roots))):
        members = np.flatnonzero(tree == t)
        densest = members[np.argsort(forest.kth_distance[members])][:12]
        print(f"tree {t}: {sizes[t]} hits, {len(members)} distinct:", " ".join(gemma.show(pieces[distinct_token[i]]) for i in densest))
