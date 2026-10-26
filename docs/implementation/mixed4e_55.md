# mixed4e:55 reproduction

Implementation notes for [the thesis](../thesis/index.md). The method is [method.md](../thesis/method.md); the reference run it reproduces is described in [in_tfm_neuron_clustering.md](in_tfm_neuron_clustering.md). No in-tfm code is used.

## Running

The notebook [mixed4e_55.ipynb](../../notebooks/mixed4e_55.ipynb) installs this package from GitHub on a Colab GPU, pulls the image pool (`gs://pj-okc/inception_pool.tar`) and reference run (`gs://pj-okc/mixed4e_55/mixed4e_55_full.tar`), and writes results to `gs://pj-okc/neural_interp/mixed4e_55/`. The same steps run locally:

```
nint collect --pool <pool> --image-list <ids.txt> --out <run>
nint report  --pool <pool> --out <run>
nint compare --ours <run>/space --reference <reference>/runs/mixed4e_55/space --out <run>/compare.json
```

`collect` writes the hit space to `<run>/space/`: `space.npz` (activations, hit vectors, forest, probabilities), `distances.npy`, `space.json` (hit ids), `summary.json`. `report` writes `<run>/report/index.html`, a page per bucket and DeepLift cards.

## Code

`selection`, `distance`, `density_peaks`, `membership`, `buckets`, `layout` and `report` in `src/neural_interp/` are model-agnostic and take arrays. The InceptionV1 specifics are in `vision/`: model, preprocessing, the exact hit vector and DeepLift (`inception.py`), cards (`cards.py`), WordNet class groups (`labels.py`). `experiments/mixed4e_55.py` wires them together; `experiments/compare.py` reads the reference run's saved arrays.

## Choices

- **Model.** Lucid's InceptionV1 (inception5h) through `torch-lucent`, ReLUs swapped for `nn.ReLU` so DeepLift recognises them. Input: shorter side resized to 256, centre crop 224, `x * 255 - 117`. This reproduces the reference activations at the same image positions (260.5, 295.2, 377.4 in both).
- **Hit vector.** `mixed4e_1x1` is a 1x1 conv, so `mixed4d[:, y, x] ⊙ W[55]` is the exact decomposition; the gradient the reference multiplies by is the same weight row. Rows sum to the pre-activation minus bias (max error 4e-4) and match the reference hadamards (median cosine 1.0).
- **Distance.** Signed split, weighted Jaccard, computed as `(Σx + Σy ∓ |x - y|₁) / 2` for the shared and union mass. Identical to the reference on its own vectors.
- **Elbow.** Kneedle over all 392,000 activations. This keeps 18,747 hits (cut 172); the reference kept 14,240 (cut 203). The reference's exact rule is unknown; positives only gives 13,529 (cut 209).
- **Unreachable hits.** Hits in a kNN component that reaches no peak use direct distance to the peaks instead of geodesic.

## Results, same 2,000 images as the reference

Hit classes come from each image's ImageNet synset, grouped by WordNet into cat (felines), dog, vehicle (wheeled vehicles) and other. Weighted purity is the size-weighted share of each cluster's dominant group.

| | hits | clusters | weighted purity | clusters with >10% cat and >10% vehicle |
|---|---|---|---|---|
| reference | 14,240 | 41 | 0.78 | 0 |
| this repo | 18,747 | 45 | 0.77 | 0 |

The largest clusters split the same way in both: one large cat-world cluster (with dogs and no vehicles), several vehicle-world clusters (with no cats). Cat and vehicle hits never share a cluster. On the 14,183 hits both runs keep, the adjusted Rand index is 0.54; the difference comes from the hit set, since this repo's forest run on the reference's distance matrix gives the same density (correlation 1.0), 98.9% of parents, 39 of 41 roots and ARI 0.90.

The fresh-sample run (seed 1) is in the notebook and not yet recorded here.
