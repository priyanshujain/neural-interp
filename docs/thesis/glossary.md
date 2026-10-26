# Glossary

Definitions for the terms used in the thesis. Each one says what the term means here, which can be narrower than its general use. [Index](index.md)

Definitions for the terms used in [in_tfm_neuron_clustering.md](in_tfm_neuron_clustering.md) and [gemma3_embedding_dims.md](gemma3_embedding_dims.md). Each one says what the term means here, which can be narrower than its general use.

## Models and units

**Neuron.** One scalar unit whose activation we study. In InceptionV1 it is one output channel of a conv layer, read at one spatial position. In Gemma there is no neuron yet: a single embedding dimension (one column of `W_E`) stands in for one.

**mixed4e:55.** Channel 55 of the pre-ReLU `mixed4e_1x1` conv in InceptionV1 (GoogLeNet). Known to respond to both cats and cars, which is why it is the worked example.

**Pre-ReLU.** The value before the ReLU clamps negatives to zero. Using it keeps information about how far below zero a position is.

**Channel.** One feature map of a conv layer. A channel at a given spatial position is one number.

**Upstream channel.** A channel in the layer that feeds the neuron. `mixed4e_1x1` reads 528 of them.

**W_E, W_U.** The token embedding matrix (vocabulary x model width, 262,144 x 640 for Gemma 3 270M) and the unembedding matrix that maps back to logits. Gemma ties them: they are the same matrix.

**Token row / embedding row.** The 640-number row of `W_E` for one token, the vector the model starts from for that token.

**Embedding dimension / column.** One of the 640 columns of `W_E`: one number per token.

**Unused and special tokens.** `<unusedN>` placeholders and control tokens such as `<bos>`. They are dropped, leaving 255,893 kept tokens.

**Byte token.** A token of the form `<0xNN>`, standing for one raw byte, used when text cannot be covered by normal pieces.

**Whole word.** A token that starts a word (begins with `▁`), is alphabetic and has at least two letters.

**Undertrained token.** A token whose embedding barely moved in training. Detected as in Land and Bartolo 2024: remove the first principal component, then flag tokens whose cosine to the mean unused-token embedding is at least the 5th percentile of the unused tokens' own scores.

**Superposition.** A model storing more features than it has dimensions, so one dimension carries parts of many unrelated features.

## Hits and their vectors

**Activation.** The neuron's value on one input at one position.

**Hit.** An input position kept for analysis because the neuron fires strongly there. For mixed4e:55 that is 14,240 image positions. For a Gemma dimension it is a token whose value in that column is past the elbow.

**Elbow (Kneedle).** On a curve sorted from high to low, the point furthest (perpendicular distance) from the straight line joining the first and last values. Everything before it is kept. Used both to pick hits and to pick forest roots.

**Bucketed selection.** The other way to pick hits: split the activation range into equal-width bands, merge small bands, and give each band an equal share of the hit budget so that rare high activations are not outnumbered.

**Attribution.** How much an upstream channel's value contributed to the neuron's activation, here a gradient-based score per channel.

**Hadamard product.** Elementwise multiplication of two vectors of the same length: `(a ⊙ b)_i = a_i * b_i`. Here it is activation times attribution for each upstream channel.

**Hit vector.** What represents a hit for clustering. In InceptionV1 it is the 528-long Hadamard product of upstream activations and their attributions, so it says how much each upstream channel drove that hit. In the Gemma runs it is the token's centred, normalised embedding row (or one dimension's values over the vocabulary, for the column run).

**Mechanism / route.** One distinct way the neuron gets driven, meaning one recurring pattern of upstream contributions. A cluster of hit vectors is taken as one mechanism.

**Worlds hypothesis.** The claim that one neuron fires for several separate kinds of input (cat and car), each through its own route, and that these show up as separate dense regions ("worlds") in hit-vector space.

## Distances

**Weighted Jaccard (Ruzicka) distance.** For non-negative vectors, `1 - Σ min(x_i, y_i) / Σ max(x_i, y_i)`: one minus the share of mass the two vectors have in common. 0 means identical, 1 means no overlap.

**Signed weighted Jaccard.** Each vector is first split into positive and negative parts (`max(x, 0)` and `max(-x, 0)`, laid side by side), so positive and negative contributions are compared separately. Then the formula above applies.

**Normalise (Jaccard).** Whether each vector is scaled to unit mass first. mixed4e:55 uses `normalise: false`, so a strong hit and a weak hit with the same pattern still differ.

**Cosine distance.** One minus the cosine of the angle between two vectors, which ignores length. Computed here as Euclidean distance between unit vectors, `sqrt(2 - 2 cos)`, which ranks pairs the same way.

**Centring.** Subtracting the mean vector (over the vocabulary for token rows) so distances measure difference from the typical token, not shared offset.

**L2 normalisation.** Dividing a vector by its length so it has length 1.

**Correlation distance.** Cosine distance between centred vectors. Used for embedding columns: two dimensions are close if their values rise and fall together across tokens.

## Density-peak forest

**kNN (k nearest neighbours).** The k points closest to a given point. Density uses k=10 for token rows, 5 for mixed4e:55 and the column run.

**Copies.** Identical hit vectors merged into one point with a count, since image patches and repeated tokens repeat exactly. A point's own copies count towards its k neighbours.

**Density (rho).** How crowded a point's neighbourhood is. Measured by the distance at which its neighbours, copies included, reach k; smaller distance means denser. Reports use log density.

**Nearest denser point.** The closest point with higher density. Every point links to it, which is its parent.

**Delta.** The distance to the nearest denser point. Large delta plus high density marks a peak.

**Density-peak forest (Rodriguez and Laio 2014).** The graph where each point links to its parent. Following links always climbs density, so it ends at a root, and each root grows one tree.

**Root.** A point above both elbows: of the sorted log-density curve and of the sorted delta curve. Every point not a root follows parents to exactly one root.

**Elbow root cuts.** The two elbow values (`cut_rho`, `cut_delta`) that decide which points are roots.

**Tree / cluster.** All points whose parent chain reaches the same root. Each tree is one cluster. Unlike HDBSCAN no low-density gap is needed, so a continuous cloud can split. The cost is that two bumps on one hill can become two clusters.

**Depth.** Number of parent links from a point up to its root.

**Peak.** A tree's points that are at least `peak_share` (0.8) as dense as its root. Used as the target for probabilities and map pulls.

**Geodesic distance.** Distance measured by walking along kNN graph edges, not in a straight line, so it follows the shape of the data and costs more across sparse gaps.

**Cluster probability.** Each point gets one probability per tree, from a softmax over its geodesic distance to each tree's peak. The temperature is the median nearest-neighbour distance (one typical step).

**Entropy.** How spread a point's probabilities are, in bits. Low entropy means one clear cluster.

**Confident hit.** A hit with probability entropy under 0.5 bits. Shown in cluster reports.

**Mixed hit.** A hit above that entropy, split between clusters. Shown on a separate mixed-hits sheet.

**Cluster bucket.** Hits where one cluster has at least 0.9, 0.8 or 0.7 probability. The three levels are nested.

**Set bucket.** Hits that need exactly the same set of clusters to reach 0.85 of their probability, for example "clusters 1 + 8".

**Touching clusters.** Clusters joined by edges of the mutual kNN graph.

## Geometry measures

**Halo (Rodriguez and Laio).** Points of a tree that are less dense than the tree's border density. The border density is the highest average density of any neighbour pair that crosses into another tree.

**Within-tree percentile.** A point's density rank inside its own tree, from 0 (sparsest) to 1 (densest). Used for core versus edge because the halo is degenerate for token rows (99% of tokens).

**Core / edge.** The dense top and the sparse bottom of a tree by within-tree percentile. Core pairs in the path test are at percentile 0.8 or more and not in the halo.

**Boundary point.** A point with most of its neighbours in other trees (own-tree share under 0.5).

**Neighbours in own tree (agree).** The share of a point's k neighbours in the same tree.

**Mutual kNN.** The share of a point's neighbours that also list the point among their own neighbours. High means a tight local cluster; noise gives low values.

**Local intrinsic dimension (Levina-Bickel MLE).** The effective number of dimensions the data spans near a point, estimated from how fast neighbour distances grow: `1 / mean_j log(d_k / d_j)` over the nearer neighbours j. Low means the points lie on a thin, smooth surface.

**k-occurrence and hubness.** How often a point appears in other points' kNN lists. Its skew measures hubness, the high-dimensional effect where a few points become everyone's neighbour.

**Gaussian null.** Random points drawn from a Gaussian with the same covariance as the embeddings and run through the same pipeline. Any structure the real data shows beyond it is not explained by covariance alone.

**Rotated control.** `W_E` multiplied by a random orthogonal matrix. Token geometry is unchanged, only the basis changes, so it tests whether the original axes mean anything.

**Excess kurtosis.** How heavy a distribution's tails are relative to a Gaussian, where 0 is Gaussian. Heavy-tailed dimensions suggest the axis carries specific features.

**Path gap.** For two tokens of one tree, points are interpolated along the great circle between them (slerp, 16 steps). The gap is the distance from each interpolated point to its nearest real token. Small gaps mean the space between them is filled (continuous), large gaps mean it is empty (discrete).

## Maps and reports

**Bucket-graph map.** The 2D picture: a sample of hits from every bucket plus each peak, a kNN graph on real distances, extra edges pulling each hit toward peaks where its probability is at least 0.1 (weight 3 times that probability), laid out with `networkx.spring_layout`. Nothing is projected, so a hit split between two clusters lands between their peaks.

**Spring layout.** A force-directed layout: edges act as springs and all nodes repel, and positions settle where forces balance.

**DeepLift card.** An image showing which pixels drive a hit, from DeepLift attribution back to the input.

**WikiText-103.** A corpus of Wikipedia articles, used for token frequencies and for the occurrence run.
