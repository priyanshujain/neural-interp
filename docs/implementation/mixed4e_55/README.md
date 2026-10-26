# mixed4e:55 reproduction

Implementation notes for [the thesis](../../thesis/index.md). The method is [method.md](../../thesis/method.md); the reference run it reproduces is described in [in_tfm_neuron_clustering.md](in_tfm_neuron_clustering.md). No in-tfm code is used.

## Running

The notebook [mixed4e_55.ipynb](mixed4e_55.ipynb) is the run itself, one step per cell with its output and wall time on a Colab T4: setup and data, model, images, activations, hits, hit vectors, distances, forest, soft membership, save, buckets and map, report, comparison with the reference, upload. It installs `neural-interp` from GitHub and pulls the image pool (`gs://pj-okc/inception_pool.tar`) and reference run (`gs://pj-okc/mixed4e_55/mixed4e_55_full.tar`); the full run is stored at `gs://pj-okc/neural_interp/mixed4e_55/same_images.tar`. Compute after the download takes about 3 minutes.

## Code

The library in `src/neural_interp/` knows nothing about InceptionV1 or images: `selection` (`ElbowSelector`), `decomposition` (`linear_contributions`), `metrics` (`weighted_jaccard`, purity, ARI), `cluster` (`DensityPeakForest`, `SoftMembership`), `buckets`, `layout` (`PulledSpringLayout`), `space` (`HitSpace`), `report` (sections and hit views) and `torch` (`Capture`, `UnitDeepLift`). Everything specific to this experiment (the lucent model, preprocessing, WordNet class groups, image cards, reading the reference run) is in the notebook.

## Choices

- **Model.** Lucid's InceptionV1 (inception5h) through `torch-lucent`, ReLUs swapped for `nn.ReLU` so DeepLift recognises them. Input: shorter side resized to 256, centre crop 224, `x * 255 - 117`. This reproduces the reference activations at the same image positions (260.5, 295.2, 377.4 in both).
- **Hit vector.** `mixed4e_1x1` is a 1x1 conv, so `mixed4d[:, y, x] ⊙ W[55]` is the exact decomposition; the gradient the reference multiplies by is the same weight row. Rows sum to the pre-activation minus bias (max error 0.05 on the T4, 4e-4 on CPU) and match the reference hadamards (median cosine 1.0).
- **Distance.** Signed split, weighted Jaccard, computed as `(Σx + Σy ∓ |x - y|₁) / 2` for the shared and union mass. Identical to the reference on its own vectors.
- **Elbow.** Kneedle over the positive activations. This keeps 13,527 hits (cut 209); the reference kept 14,240 (cut 203), and about 14,200 of our activations are at or above 203, so the reference cut comes from a curve or rule not yet described. Rescaling the axes to [0, 1] does not change the cut.
- **Forest.** Copies after the sign split rounded to 6 decimals, rho = `-log10(d_k + c)`, ties to more copies, trees numbered by size. On the reference's own distance matrix this gives the same rho cut, delta cut 0.5217 against 0.5219, 40 of 41 roots and tree ARI 0.96.
- **Soft membership.** A peak is the tree's points with rho at least 0.8 times the root's rho; the docs leave the scale open, and of the readings tried this one matches the reference best. Geodesics run on the union kNN graph, temperature is the median nearest-neighbour distance, and a hit that reaches no peak gets probability 1 for its own tree. On the reference's distances and trees the most probable cluster agrees with the reference's for 90% of hits (mean probability difference 0.004); the rest differ in peak sets or graph details the docs do not pin down.

## Results, same 2,000 images as the reference

Hit classes come from each image's ImageNet synset, grouped by WordNet into cat (felines), dog, vehicle (wheeled vehicles) and other. Weighted purity is the size-weighted share of each cluster's dominant group.

| | hits | clusters | weighted purity | clusters with >10% cat and >10% vehicle |
|---|---|---|---|---|
| reference | 14,240 | 41 | 0.82 | 0 |
| this repo | 13,527 | 37 | 0.82 | 0 |

A hit's cluster is its most probable cluster, for both runs. The largest clusters line up: a vehicle cluster of about 3,500 hits at 0.91 to 0.93 vehicle, a cat cluster of about 1,400 at 0.80 cat, a second vehicle cluster and a second cat cluster, and a mixed "other" cluster. Cat and vehicle hits never share a cluster. On the 13,341 hits both runs keep, the adjusted Rand index is 0.61. A CPU run of the same notebook gives 35 clusters: about a dozen roots sit just above the delta cut, so tiny numerical differences between devices move a couple of them across it.

Report (maps, a page per bucket, DeepLift cards): <https://interp.pjay.in/mixed4e_55/same_images/report/index.html>. Comparison numbers: <https://interp.pjay.in/mixed4e_55/same_images/compare.json>.

A fresh-sample run (different images from the same pool) has not been done yet.
