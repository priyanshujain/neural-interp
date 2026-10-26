import base64
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from neural_interp.lib.report import render

MIN_TREE = 20
N_NEIGHBOURS = 10


def percentile(values):
    return (np.argsort(np.argsort(values)) + 0.5) / len(values)


def decile_curves(pct, mask, columns):
    bins = np.minimum((pct * 10).astype(int), 9)
    return {
        name: [float(fn(values[mask & (bins == b)])) if (mask & (bins == b)).any() else None for b in range(10)]
        for name, (values, fn) in columns.items()
    }


def continuity(forest, with_agree=True):
    columns = {"local intrinsic dim (median)": (forest["local_id"], np.median), "mutual kNN": (forest["mutual"], np.mean)}
    if with_agree:
        columns["neighbours in own tree"] = (forest["agree"], np.mean)
    return columns


def b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def build_curves(real, null, feat, freq, big, rho_pct):
    n = len(real["tree"])
    everyone = np.ones(n, bool)
    meaning = {
        "whole word": (feat["is_word"], np.mean),
        "in WordNet": (feat["in_wordnet"], np.mean),
        "undertrained": (feat["is_undertrained"], np.mean),
        "byte token": (feat["is_byte"], np.mean),
        "seen in WikiText": (freq > 0, np.mean),
    }
    null_big = np.bincount(null["tree"])[null["tree"]] >= MIN_TREE
    curves = {
        "within_meaning": decile_curves(real["tree_pct"], big, meaning),
        "within_continuity": decile_curves(real["tree_pct"], big, continuity(real)),
        "global_meaning": decile_curves(rho_pct, everyone, meaning),
        "global_continuity": decile_curves(rho_pct, everyone, continuity(real)),
        "null_within_continuity": decile_curves(null["tree_pct"], null_big, continuity(null)),
        "null_global_continuity": decile_curves(percentile(null["rho"]), everyone, continuity(null, with_agree=False)),
    }
    bands = {"unseen": freq == 0, "1 to 100": (freq > 0) & (freq <= 100), "over 100": freq > 100}
    for key, values, fn in [("word_by_freq_band", feat["is_word"], np.mean), ("lid_by_freq_band", real["local_id"], np.median)]:
        curves[key] = {name: decile_curves(real["tree_pct"], big & band, {"w": (values, fn)})["w"] for name, band in bands.items()}
    return curves


def build_summary(real, null, feat, freq, sizes, rho_pct, boundary):
    undertrained, rho, seen = feat["is_undertrained"], real["rho"], freq > 0
    return {
        "tokens": len(rho),
        "trees (real)": int(len(real["roots"])),
        "trees (null)": int(len(null["roots"])),
        f"trees with at least {MIN_TREE} tokens": int((sizes >= MIN_TREE).sum()),
        "Rodriguez-Laio halo fraction (real / null)": f"{real['halo'].mean():.3f} / {null['halo'].mean():.3f}",
        "boundary fraction, most neighbours in other trees (real / null)":
            f"{boundary.mean():.3f} / {(null['agree'] < 0.5).mean():.3f}",
        "median local intrinsic dim (real / null)": f"{np.median(real['local_id']):.1f} / {np.median(null['local_id']):.1f}",
        "mean mutual kNN (real / null)": f"{real['mutual'].mean():.3f} / {null['mutual'].mean():.3f}",
        "hubness skew of 10-occurrence (real / null)": f"{float(real['hub_skew']):.1f} / {float(null['hub_skew']):.1f}",
        "Spearman density vs log frequency, seen tokens": f"{spearmanr(rho[seen], np.log10(freq + 1)[seen]).statistic:.3f}",
        "undertrained tokens": int(undertrained.sum()),
        "undertrained median global density percentile": f"{np.median(rho_pct[undertrained]):.3f}",
        "undertrained median within-tree percentile": f"{np.median(real['tree_pct'][undertrained]):.3f}",
    }


def build_tree_rows(real, feat, tokens, sizes, boundary, log_freq):
    rows = []
    for t in np.flatnonzero(sizes >= 5):
        members = np.flatnonzero(real["tree"] == t)
        densest = members[np.argsort(-real["rho"][members])]
        scripts, script_counts = np.unique(feat["script"][members], return_counts=True)
        top = np.argmax(script_counts)
        rows.append({
            "tree": int(t), "size": int(len(members)), "root": tokens[int(real["roots"][t])],
            "core": [tokens[i] for i in densest[:14]], "edge": [tokens[i] for i in densest[-8:][::-1]],
            "script": f"{scripts[top]} {script_counts[top] / len(members):.0%}",
            "word": round(float(feat["is_word"][members].mean()), 3),
            "undertrained": round(float(feat["is_undertrained"][members].mean()), 3),
            "boundary": round(float(boundary[members].mean()), 3),
            "lid": round(float(np.median(real["local_id"][members])), 1),
            "freq": round(float(np.median(log_freq[members])), 2),
        })
    return rows


def build_paths(real, tokens):
    kind, gap, end_d = real["path_kind"], real["path_gap"][:, 1:-1].mean(1), real["path_end_distance"]
    edges = np.quantile(end_d, np.linspace(0, 1, 7))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = {"range": f"{lo:.2f} to {hi:.2f}"}
        for k in ("core", "halo"):
            sel = (kind == k) & (end_d >= lo) & (end_d <= hi)
            row[k] = round(float(gap[sel].mean()), 4) if sel.any() else None
            row[k + "_n"] = int(sel.sum())
        bins.append(row)
    examples = []
    for k in ("core", "halo"):
        for p in np.flatnonzero(kind == k)[:10]:
            seq = [tokens[i] for i in real["path_token"][p]]
            examples.append({"kind": k, "gap": round(float(gap[p]), 3), "end": round(float(end_d[p]), 3),
                             "tokens": [s for j, s in enumerate(seq) if j == 0 or s != seq[j - 1]]})
    return bins, examples


def run(out, html_path):
    out = Path(out)
    tokens = json.loads((out / "tokens.json").read_text())
    feat = np.load(out / "features.npz")
    freq = np.load(out / "freq.npy")
    real = np.load(out / "forest_real.npz")
    null = np.load(out / "forest_null.npz")
    assert len(real["tree"]) == len(tokens), "copies were merged, map through distinct_of_token"

    tree = real["tree"]
    boundary = real["agree"] < 0.5
    log_freq = np.log10(freq + 1)
    sizes = np.bincount(tree)
    rho_pct = percentile(real["rho"])
    summary = build_summary(real, null, feat, freq, sizes, rho_pct, boundary)
    path_bins, examples = build_paths(real, tokens)
    script_names, script_code = np.unique(feat["script"], return_inverse=True)
    flags = (feat["is_word"] * 1 + feat["in_wordnet"] * 2 + feat["is_undertrained"] * 4 + feat["is_byte"] * 8
             + boundary * 16 + real["is_root"] * 32)
    data = {
        "tokens": tokens, "scripts": script_names.tolist(), "summary": summary,
        "curves": build_curves(real, null, feat, freq, sizes[tree] >= MIN_TREE, rho_pct),
        "trees": build_tree_rows(real, feat, tokens, sizes, boundary, log_freq),
        "path_bins": path_bins, "examples": examples, "roots": real["roots"].astype(int).tolist(),
        "arrays": {
            "x": b64(real["xy"][:, 0], np.float32), "y": b64(real["xy"][:, 1], np.float32),
            "tree": b64(tree, np.int32), "parent": b64(real["parent"], np.int32),
            "rho": b64(real["rho"], np.float32), "rho_pct": b64(rho_pct, np.float32),
            "tree_pct": b64(real["tree_pct"], np.float32), "lid": b64(real["local_id"], np.float32),
            "agree": b64(real["agree"], np.float32), "log_freq": b64(log_freq, np.float32),
            "script": b64(script_code, np.int32), "flags": b64(flags, np.int32),
            "neighbours": b64(real["neighbours"][:, :N_NEIGHBOURS], np.int32),
            "k_occ": b64(real["k_occurrence"], np.int32),
        },
    }
    render("worlds", data, html_path)
    for k, v in summary.items():
        print(f"{k}: {v}")
