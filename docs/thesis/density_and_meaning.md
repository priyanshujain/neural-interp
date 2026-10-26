# Density and meaning: Gemma 3 token rows

[Index](index.md)

These runs cluster Gemma 3's tokens rather than its dimensions. They came before the dimension work and are a side branch, but they bear directly on the sparse-hits problem: if density tracked meaning, sparse hits could be dropped as noise.

## Hypothesis tested

> Inside a density cluster, points near the dense core carry continuous meaning (move a little and the meaning shifts a little), while points at the sparse edge are discrete and carry little meaning.

Two parts can fail separately. Geometry: is the core smoother and better connected than the edge? Meaning: do core tokens mean more?

## Measurements

**Position in tree.** Within-tree percentile: a token's density rank inside its own tree, 0 sparsest to 1 densest. Core is the top, edge the bottom. Rodriguez and Laio's halo (points less dense than the tree's border density) covers 99% of tokens here and is useless, so the percentile is used instead. A boundary point has most of its neighbours in other trees.

**Smoothness.** Share of neighbours in own tree. Mutual kNN (share of neighbours that list the point back). Local intrinsic dimension by the Levina-Bickel estimator, `1 / mean_j log(d_k / d_j)` over the 1st to 20th neighbours; low means a smooth, low-dimensional surface. Hubness, the skew of how often each point appears in others' neighbour lists. Path gap: interpolate along the great circle between two tokens of one tree in 16 steps and measure the distance from each step to the nearest real token; small means filled and continuous, large means empty and discrete.

**Meaning.** Whole word (starts a word, alphabetic, at least two letters). In WordNet. Byte token. Frequency in the first 400,000 non-empty lines of WikiText-103. Undertrained, as in Land and Bartolo 2024: remove the first principal component, then flag tokens whose cosine to the mean unused-token embedding is at least the 5th percentile of the unused tokens' own scores.

**Control.** Gaussian null: random points with the real embeddings' covariance, through the same pipeline.

## Run A: every token row

All 255,893 kept tokens, centred and normalised, forest with k = 10.

| | Real | Gaussian null |
|---|---|---|
| Trees | 1,007 | 2 |
| Median local intrinsic dim | 26.7 | 127.8 |
| Mean mutual kNN | 0.405 | 0.029 |
| Hubness skew | 76 | 35 |

**Geometry holds.** From the sparse edge to the dense core of a tree, local intrinsic dimension falls from 71 to 18, and the share of neighbours in the same tree rises from 0.46 to 0.89. The structure is far beyond what covariance explains.

**Meaning does not.** Whole-word share is flat at about 0.44 across density. WordNet coverage falls slightly toward the core. Undertrained tokens sit at median density, not at the edge. Density is unrelated to frequency (Spearman -0.05).

**Known bug.** The report for this run read the forest's internally sorted points in input order, so its tree contents and within-tree meaning curves are wrong. Tree counts, the null comparison and the global curves are unaffected. The corrected rerun was stopped before it finished. The meaning result above must be rechecked on the corrected run before it is cited.

## Run B: WikiText occurrences

Ten random 5,000-token windows of WikiText-103, with 14,240 occurrences sampled (mixed4e:55's hit count). Each occurrence is its token's row of `W_E`. Repeats merge as copies, giving 3,752 distinct tokens.

206 trees. Common tokens become peaks by repetition (`the`, `and`, `to`), as repeated patches do in images. Semantic trees appear: speech verbs (mentioned, described, stated), making verbs (made, built, designed), digits, punctuation. The input has no context, so this is a frequency-weighted view of the vocabulary, not of meaning in context.

## What it adds up to

Density predicts geometry but not meaning, and where density moves with anything it is repetition. This is the same pattern as on the induction heads, where density tracks token frequency and sparse hits are real, used behaviour. It argues against dropping sparse hits.

## Artifacts

[../implementation/gemma3_token_rows.md](../implementation/gemma3_token_rows.md).
