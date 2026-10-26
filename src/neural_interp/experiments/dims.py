import json
from pathlib import Path

import numpy as np
from in_tfm.density_peaks import log_density
from scipy.stats import kurtosis

from neural_interp.lib.forest import cluster, peak_pull_layout
from neural_interp.lib.geometry import unit
from neural_interp.lib.report import save_json

K = 5
TOP = 20


def off_diagonal(matrix):
    return np.abs(matrix[~np.eye(len(matrix), dtype=bool)])


def cluster_columns(matrix, label):
    columns = (matrix - matrix.mean(0)).T
    forest, trees = cluster(columns, K, input_order=True)
    corr = unit(columns) @ unit(columns).T
    off = off_diagonal(corr)
    kurt = kurtosis(matrix, axis=0)
    print(f"{label}: trees {len(trees.roots)}, cuts {trees.cuts}, mean |corr| {off.mean():.4f}, "
          f"max |corr| {off.max():.3f}, median kurtosis {np.median(kurt):.2f}, max kurtosis {kurt.max():.1f}", flush=True)
    return forest, trees, corr, kurt


def tree_rows(W, tokens, trees, corr):
    rows = []
    for t in range(len(trees.roots)):
        members = np.flatnonzero(trees.tree == t)
        signs = np.sign(corr[trees.roots[t], members])
        direction = ((W[:, members] - W[:, members].mean(0)) / W[:, members].std(0)) @ signs
        rows.append({
            "tree": t, "size": len(members), "root": int(trees.roots[t]), "members": members.tolist(),
            "mean_corr": round(float(off_diagonal(corr[np.ix_(members, members)]).mean()), 3) if len(members) > 1 else None,
            "pos": [tokens[i] for i in np.argsort(-direction)[:TOP]],
            "neg": [tokens[i] for i in np.argsort(direction)[:TOP]],
        })
    return rows


def run(out):
    out = Path(out)
    rng = np.random.default_rng(0)
    W = np.load(out / "emb.npy")
    tokens = json.loads((out / "tokens.json").read_text())
    n_vocab, d_model = W.shape
    rotation = np.linalg.qr(rng.standard_normal((d_model, d_model)))[0]

    forest, trees, corr, kurt = cluster_columns(W, "raw")
    _, trees_rot, corr_rot, kurt_rot = cluster_columns(W @ rotation, "rotated")

    rho = log_density(forest.kth_distance)
    order = np.lexsort((-rho, trees.tree))
    top_pos = np.argsort(-W, axis=0)[:TOP].T
    top_neg = np.argsort(W, axis=0)[:TOP].T
    xy, probabilities = peak_pull_layout(unit((W - W.mean(0)).T), forest, trees, per_tree=d_model, max_nodes=10**6)
    rows = tree_rows(W, tokens, trees, corr)
    result = {
        "n_vocab": n_vocab, "d_model": d_model, "k": K,
        "summary": {
            "trees (raw / rotated)": f"{len(trees.roots)} / {len(trees_rot.roots)}",
            "mean |correlation| between dimensions (raw / rotated)":
                f"{off_diagonal(corr).mean():.4f} / {off_diagonal(corr_rot).mean():.4f}",
            "median excess kurtosis (raw / rotated)": f"{np.median(kurt):.2f} / {np.median(kurt_rot):.2f}",
            "max excess kurtosis (raw / rotated)": f"{kurt.max():.1f} / {kurt_rot.max():.1f}",
            "dimensions with kurtosis above rotated max": int((kurt > kurt_rot.max()).sum()),
        },
        "dims": {
            "tree": trees.tree.tolist(), "label": probabilities.argmax(1).tolist(), "parent": forest.parent.tolist(),
            "depth": trees.depth.tolist(), "rho": rho.round(4).tolist(), "delta": forest.delta.round(4).tolist(),
            "neighbours": forest.neighbours.tolist(), "kurtosis": kurt.round(2).tolist(),
            "norm": np.linalg.norm(W, axis=0).round(3).tolist(), "mean": W.mean(0).round(5).tolist(),
            "x": xy[:, 0].round(3).tolist(), "y": xy[:, 1].round(3).tolist(),
            "pos": [[tokens[i] for i in r] for r in top_pos], "neg": [[tokens[i] for i in r] for r in top_neg],
            "pos_val": np.take_along_axis(W.T, top_pos, 1).round(4).tolist(),
            "neg_val": np.take_along_axis(W.T, top_neg, 1).round(4).tolist(),
        },
        "order": order.tolist(),
        "corr": np.round(corr[np.ix_(order, order)], 3).tolist(),
        "corr_rot": np.round(corr_rot, 3).tolist(),
        "kurtosis_rot": kurt_rot.round(2).tolist(),
        "trees": rows,
        "trees_rot_sizes": np.bincount(trees_rot.tree).tolist(),
    }
    save_json(out / "dims.json", result)
    for row in rows[:12]:
        print(f"tree {row['tree']} ({row['size']} dims, mean |corr| {row['mean_corr']}):", " ".join(row["pos"][:10]), "|", " ".join(row["neg"][:10]))
    print("rotated tree sizes", sorted(result["trees_rot_sizes"], reverse=True)[:15])
