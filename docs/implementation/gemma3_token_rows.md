# Side runs: clustering Gemma 3's tokens

Implementation notes for [the thesis](../thesis/index.md).

These two runs cluster Gemma 3's tokens instead of its dimensions. They came before Hariom's dimension proposal and are a side branch: the main line is [gemma3_embedding_dims.md](gemma3_embedding_dims.md), which follows his method from [in_tfm_neuron_clustering.md](in_tfm_neuron_clustering.md). Read those two first for the method, the language-model background (tokens, `W_E`, kept tokens) and the cosine distance used here. [glossary.md](../thesis/glossary.md) is a short lookup table.

## The thesis tested

> Inside a density cluster, points near the dense core carry continuous meaning (move a little and the meaning shifts a little), while points at the sparse edge are discrete and carry little meaning.

Two parts can fail separately. **Geometry**: is the core really smoother and better connected than the edge? **Meaning**: do core tokens actually mean more?

## Measurements

### Where a token sits in its tree

- **Within-tree percentile.** The token's density rank inside its own tree, from 0 (sparsest) to 1 (densest). **Core** means the dense top of a tree, **edge** the sparse bottom.
- **Halo.** Rodriguez and Laio's definition of a cluster's edge. A tree's **border density** is the highest average density of any neighbour pair that crosses into another tree, and halo points are less dense than that. For tokens it covers 99% of points and is useless, so the within-tree percentile is used instead.
- **Boundary point.** A token with most of its neighbours in other trees.

### How smooth the space is

- **Neighbours in own tree.** The share of a token's k nearest neighbours in its own tree.
- **Mutual kNN.** The share of a token's neighbours that also list it among their own neighbours. Smooth, filled regions have symmetric neighbour relations; noise and edges do not.
- **Local intrinsic dimension (LID).** How many dimensions the data actually spans near a point. A sheet of paper in 3D has intrinsic dimension 2. It is estimated with Levina and Bickel's maximum-likelihood estimator from how fast distances to the 1st to 20th neighbours grow: `1 / mean_j log(d_k / d_j)`. Low LID means a smooth, low-dimensional surface.
- **Hubness.** In high dimensions a few points end up in almost everyone's neighbour list. The skew of each point's **k-occurrence** (how often it appears in others' lists) measures this.
- **Path gap.** Take two tokens of one tree and walk the shortest arc between them on the unit sphere in 16 steps (spherical interpolation, "slerp"). At each step, measure the distance to the nearest real token. A small gap means the space between them is filled (continuous), a large one means it is empty (discrete).

### What a token means

- **Whole word.** Starts a word (`␣`), is alphabetic, has at least two letters.
- **In WordNet.** The lowercased token is in WordNet, an English dictionary database.
- **Byte token.** `<0xNN>`, one raw byte.
- **Frequency.** Occurrences in the first 400,000 non-empty lines of WikiText-103 (a corpus of Wikipedia articles).
- **Undertrained.** A token whose embedding barely moved in training. Detected as in Land and Bartolo 2024: remove the first principal component, then flag tokens whose cosine to the average unused-token embedding is at least the 5th percentile of the unused tokens' own scores.

### Control

**Gaussian null.** Random points drawn from a Gaussian with the real embeddings' covariance, run through the same pipeline. Structure beyond the null is not explained by covariance alone.

## Run A: every token row

**Setup.** Notebook `notebooks/gemma3_embedding_worlds.ipynb`, commands `embed`, `freq`, `forest`, `worlds-report`. All 255,893 kept tokens are points, centred and normalised. The forest uses k = 10 and runs on a GPU, with exact nearest-neighbour and nearest-denser search in torch, because in-tfm's CPU version is too slow at this size.

**Result.**

| | Real | Gaussian null |
|---|---|---|
| Trees | 1,007 | 2 |
| Median local intrinsic dim | 26.7 | 127.8 |
| Mean mutual kNN | 0.405 | 0.029 |
| Hubness skew | 76 | 35 |

The **geometry** part holds. From the sparse edge to the dense core of a tree, LID falls from 71 to 18, and the share of neighbours in the same tree rises from 0.46 to 0.89.

The **meaning** part does not. Whole-word share is flat at about 0.44 across density. WordNet coverage falls slightly toward the core. Undertrained tokens sit at median density, not at the edge. Density is unrelated to frequency (Spearman -0.05).

**Known bug.** `outputs/reports/gemma3_embedding_worlds/` reads the forest's internally sorted points in input order, so its tree contents and within-tree meaning curves are wrong. Counts, the null comparison and global curves are unaffected. The corrected rerun was stopped before it finished.

## Run B: WikiText occurrences

**Setup.** Notebook `notebooks/gemma3_wikitext_occurrences.ipynb`, command `occurrences`, report `outputs/reports/gemma3_wikitext_occurrences_graph/`. Ten random 5,000-token windows of WikiText-103, with 14,240 occurrences sampled from them (mixed4e:55's hit count). Each occurrence is its token's row of `W_E`. Repeats merge as copies with counts, giving 3,752 distinct tokens.

**Result.** 206 trees. Common tokens become peaks by repetition (`the`, `and`, `to`), as repeated patches do in images. Semantic trees appear: speech verbs (mentioned, described, stated), making verbs (made, built, designed), digits, punctuation. The input has no context, so this is a frequency-weighted view of the vocabulary, not of meaning in context.

## Running

Setup as in [gemma3_embedding_dims.md](gemma3_embedding_dims.md). `forest` needs a CUDA GPU, so both runs go on Colab.

```
nint embed out
nint freq out
nint forest out real
nint forest out null
nint worlds-report out out/report.html

nint occurrences occ
nint render occ occ/occurrences.json occ/report.html
```
