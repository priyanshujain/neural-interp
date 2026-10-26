import numpy as np
from in_tfm.bucket_graph import sample_map
from in_tfm.density_peaks import DensityPeakForest, ForestTrees, calculate_probs, density_peak_forest, forest_trees
from in_tfm.pair_mixed_report import spread_by_share

PEAK_SHARE = 0.8


def cluster(rows, k, input_order=False):
    forest = density_peak_forest(rows, k=k, distance="euclidean", normalise=True)
    trees = forest_trees(forest)
    return in_input_order(forest, trees) if input_order else (forest, trees)


def peak_pull_layout(unit_rows, forest, trees, per_tree, max_nodes=3000):
    """in-tfm's bucket-graph map: a kNN graph on real distances plus pulls to cluster peaks, laid out by spring_layout.

    Returns full-length x and y, NaN for points outside the sample."""
    probabilities = calculate_probs(forest, trees, PEAK_SHARE)
    sampled = set(trees.roots.tolist())
    budget = max(1, (max_nodes - len(sampled)) // len(trees.roots))
    for t in range(len(trees.roots)):
        members = np.flatnonzero(trees.tree == t)
        sampled.update(spread_by_share(members, probabilities[members, t], min(per_tree, budget)).tolist())
    sample = np.array(sorted(sampled))
    rows = unit_rows[sample]
    local = np.clip(1.0 - rows @ rows.T, 0.0, 1.0)
    index = {int(p): i for i, p in enumerate(sample)}
    result = sample_map(local, probabilities[sample], np.arange(len(sample)), [index[int(r)] for r in trees.roots])
    xy = np.full((len(unit_rows), 2), np.nan)
    xy[sample] = result.points
    print("graph layout", len(sample), "nodes, k", result.neighbours, flush=True)
    return xy, probabilities


def in_input_order(forest, trees):
    """The forest and trees indexed like the input rows. `merge_copies` sorts the rows, so without copies this is a relabelling."""
    assert (forest.counts == 1).all(), "only for inputs without copies"
    to_distinct = forest.distinct_of_hit
    to_input = np.empty_like(to_distinct)
    to_input[to_distinct] = np.arange(len(to_distinct))
    parent = forest.parent[to_distinct]
    return (
        DensityPeakForest(
            distinct_of_hit=np.arange(len(to_distinct)), counts=forest.counts[to_distinct],
            kth_distance=forest.kth_distance[to_distinct], density_rank=forest.density_rank[to_distinct],
            parent=np.where(parent >= 0, to_input[np.maximum(parent, 0)], -1), delta=forest.delta[to_distinct],
            neighbours=to_input[forest.neighbours[to_distinct]],
            neighbour_distance=forest.neighbour_distance[to_distinct],
        ),
        ForestTrees(cuts=trees.cuts, roots=to_input[trees.roots], tree=trees.tree[to_distinct], depth=trees.depth[to_distinct]),
    )
