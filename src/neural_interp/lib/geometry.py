import numpy as np


def unit(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def tree_percentile(tree, rho):
    sizes = np.bincount(tree)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    pct = np.empty(len(tree))
    pct[np.lexsort((rho, tree))] = (np.arange(len(tree)) - np.repeat(offsets, sizes) + 0.5) / np.repeat(sizes, sizes)
    return pct


def local_intrinsic_dim(neighbour_distance):
    d = np.maximum(neighbour_distance.astype(np.float64), 1e-9)
    return 1.0 / np.mean(np.log(d[:, -1:] / d[:, :-1]), axis=1)


def mutual_knn(neighbours):
    return (neighbours[neighbours] == np.arange(len(neighbours))[:, None, None]).any(2).mean(1)


def other_tree(tree, neighbours):
    return tree[neighbours] != tree[:, None]


def nan_free(values):
    return [None if np.isnan(v) else round(float(v), 4) for v in values]

