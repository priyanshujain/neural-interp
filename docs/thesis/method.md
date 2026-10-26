# Method

[Index](index.md)

The method collects many moments when a neuron fires strongly, describes each by exactly which inputs drove it, and clusters those descriptions. If the claim holds, the descriptions fall into separate groups, and each group is one mode with one meaning. Steps 1 to 3 turn the neuron's behaviour into points with distances. Steps 4 to 6 cluster them. Steps 7 to 9 use the modes. Step 10 draws them.

## 1. Neuron

One weight row in a dot product. In InceptionV1, a layer outputs a stack of 2D feature maps (channels), and one channel at one spatial position is one number. `mixed4e:55` is channel 55 of the 1x1 conv inside block `mixed4e`. Its output at a position is a weighted sum of the 528 upstream channels at that position, plus a bias. It is read pre-ReLU, before negatives are clamped to zero, so how far below zero a position is is kept.

## 2. Hits

A hit is an input on which the neuron is strongly active. Most activations are weak and say little about what the neuron is for, so only the strongest are kept.

**Elbow cut.** Sort the positive activations from high to low. The curve drops steeply, then flattens into a long tail. The elbow is found with the Kneedle rule: draw a straight line from the first point to the last, and take the point furthest from it (perpendicular distance). Everything before it is a hit. It is automatic, so the cut is not tuned per neuron. Only the positive side is used even when the unit is read pre-ReLU: the negative tail would otherwise pull the line and move the elbow.

**Bucketed selection.** The alternative. Split the activation range between a floor and the maximum into equal-width bands, merge bands with too few activations into a neighbour, and give each band an equal share of the hit budget. The elbow is dominated by the many moderate activations, since high ones are rare; bucketing guarantees the top is represented.

**Units without a ReLU.** A unit whose values form a symmetric bell curve (an embedding dimension, for example) has no privileged side, and an elbow over all values lands in the wrong tail. There the elbow is taken over one side of zero at a time, as a ReLU would leave it, and both sides are run. Flipping the sign of a unit and every weight that reads it gives an identical model, so neither side is privileged.

## 3. Hit vector: the exact decomposition

The Hadamard product of the input and the weight row, entry by entry: `(a ⊙ w)_d = a_d × w_d`. It sums exactly to the neuron's pre-activation (minus the bias), and entry d is the exact contribution of input dim d.

Neither activation nor weight alone measures contribution. A channel can be strongly on but ignored, or heavily weighted but off. Their product captures both: on and used. Example with three inputs: activations `[0.5, 2.0, 0.0]`, weights `[1.0, -0.3, 4.0]`, product `[0.5, -0.6, 0.0]`. Input 1 pushed the neuron up by 0.5, input 2 pushed it down by 0.6, input 3 would matter but was off.

Two hits with similar hit vectors were driven by the same inputs in the same proportions, so they share a route.

In the mixed4e:55 run the per-channel factor is the raw weight of the 1x1 conv. For a linear unit this equals the gradient of the pre-activation with respect to the input, so the hit vector is also gradient times input.

## 4. Distance

Weighted Jaccard, also called Ruzicka distance. For non-negative x and y:

```
distance = 1 - Σ min(x_i, y_i) / Σ max(x_i, y_i)
```

The top sum is the mass the two vectors share, the bottom the mass they have between them. Identical vectors give 0, vectors with no input in common give 1.

Hit vectors have negative entries, so each is first split into a positive part `max(x, 0)` and a negative part `max(-x, 0)`, laid side by side. Pushing up and pushing down are compared separately and never cancel. Example: x = `[2, -1, 0]`, y = `[1, -1, 3]` become `[2, 0, 0 | 0, 1, 0]` and `[1, 0, 3 | 0, 1, 0]`. Sum of minimums 2, sum of maximums 6, distance 0.67.

Why not Euclidean or cosine: Jaccard asks which inputs carry the mass and how much, which is exactly the question, and it is not dominated by one large entry the way squared distances are. Vectors are not normalised first, so a strong hit and a weak hit with the same pattern still differ.

## 5. Modes

**History.** The first draft clustered with HDBSCAN. HDBSCAN needs a low-density gap between clusters, so a continuous cloud with two bumps stays one cluster, and in practice it split modes at the base of the activation ramp even where Hadamard products looked alike (see [open_problems.md](open_problems.md)). It also labelled 50 to 95% of hits as noise.

**Current: density-peak forest** (Rodriguez and Laio, 2014). Clusters are hills of density; each hill has a peak; every hit belongs to the hill it climbs.

- **Copies.** Identical hit vectors (image patches and repeated tokens repeat exactly) merge into one point with a count. Vectors are compared after the sign split and rounded to 6 decimals, so float noise does not separate copies.
- **Density.** For each point, the distance at which its k nearest neighbours, copies included, add up to k. Smaller is denser. Used as log density, rho = -log10(d_k + c), with c half the smallest nonzero d_k so points at distance 0 stay finite. mixed4e:55 uses k = 5.
- **Parent and delta.** Points are ranked by density; ties, common at d_k = 0, go to the point with more copies. Every point links to its nearest denser point (its parent). The distance is its delta. Typical points have small delta; a hilltop must cross a valley, so its delta is large.
- **Roots.** Sort rho and find its elbow, sort delta and find its elbow, and keep points above both. Every other point follows parents to exactly one root.
- **Modes.** All points reaching the same root form one tree, and each tree is one mode. Trees are numbered by size, largest first, so mode ids are not stable across runs; compare runs by membership, not by id.

Picture two hills joined by a low ridge: points on each hill climb to its own summit, both summits become roots, and the ridge splits where the climbing direction changes.

The trade-off against HDBSCAN is explicit. The forest can split a continuous cloud, which matters because routes into a neuron may blend. It cannot tell two genuinely separate hills from two bumps on one hill, so it can over-split.

## 6. Soft membership

The forest gives each hit one mode, but some hits sit between two.

- **Peak** of a mode: all its points at least 0.8 as dense as its root.
- **Geodesic distance:** distance walking along kNN graph edges, so crossing a sparse gap is expensive.
- **Probability:** a softmax over each hit's geodesic distance to every mode's peak, with temperature one typical step (the median nearest-neighbour distance). A hit that reaches no peak along the graph gets probability 1 for its own tree.
- **Mode of a hit:** from here on, a hit belongs to its most probable mode, not the tree it grew in. The two disagree often (5,218 of 14,240 hits for mixed4e:55), so touching modes, composition and reports all use the most probable mode.
- **Entropy** of those probabilities, in bits. Under 0.5 bits is a confident hit; above is a mixed hit.
- **Buckets:** hits where one mode has at least 0.9, 0.8 or 0.7 probability (nested), and set buckets of hits that need exactly the same set of modes to reach 0.85 of their probability (for example "modes 1 + 8"), up to half of all modes. Buckets hold every qualifying hit, confident or not; entropy only marks hits in the report. Only buckets with at least 10 hits get a page. Modes are said to touch when a mutual kNN edge joins hits whose most probable modes differ.

- **Mixed-hits sheet:** the mode report shows only confident hits. The rest go on a separate sheet, sampled across entropy levels so it runs from mildly to heavily mixed. Each hit is shown next to the mean hit vector of each of its likely modes, faded by its probability for that mode.
- **Pair report:** a hit belongs to the pair of its two most probable modes when the second has at least 0.1. Pairs with at least 30 hits get a page, binned by entropy (edges 0.5, 1, 1.5, 2 bits), with each bin running from mostly one mode to mostly the other. This shows what changes along the border between two modes.

This turns the boundary between modes into a measurement rather than an assumption, and gives the mixed hits a place in the record.

## 7. Dependencies

For each mode, the input dims covering 90% of its Hadamard mass. These dims are neurons of the previous layer. Following them gives the circuit, one mode at a time.

## 8. Runtime assignment

A new input's Hadamard product is assigned to the nearest mode. This is what makes the meaning available per activation at inference.

## 9. Bilinear units

Attention scores are bilinear in query and key: `q^T W_Q^T W_K k`. Head dims are an arbitrary basis, since rotating q and k together changes every entry and no score. So the score is decomposed in the singular basis of `W_Q^T W_K`, where each singular direction contributes an exact term, and inputs are split further into token and position parts.

## 10. Maps

The usual approach squashes points into 2D with UMAP or t-SNE, which distorts distances. The map here projects nothing.

1. Sample 8 hits from every bucket of at least 10, at evenly spaced ranks of the bucket's leading-mode probability (so a set bucket runs from one side to the other), plus each mode's root.
2. Build a kNN graph on the real distances, edge weight one minus distance, using the smallest k from 6, 10, 15, 25 that connects the graph.
3. Add a pull: an edge from each hit to the peak of every mode where its probability is at least 0.1, weight 3 times that probability.
4. Lay out with a force-directed spring layout (300 iterations, fixed seed).

Without the pull the picture is one blob, since a mixed hit's neighbours belong to many modes. With it, confident hits gather around their peak and a hit split between two modes sits between their peaks. Up to 6 hits per bucket, chosen the same way, are numbered and shown as cards: a DeepLift map (vision) or its token and context (text), next to the mean hit vector of its top 3 modes faded by probability, so a mode's meaning can be read off.

## Substitutions in specific experiments

Where an experiment cannot follow the method exactly, its page says so. The Gemma embedding runs ([gemma_embedding_dims.md](gemma_embedding_dims.md), [density_and_meaning.md](density_and_meaning.md)) have nothing upstream to decompose, so they use the token's embedding row as the hit vector and cosine distance (centre on the vocabulary mean, L2-normalise, Euclidean between unit vectors, which orders pairs as cosine distance does). Those runs support the worlds picture but do not test exactness.
