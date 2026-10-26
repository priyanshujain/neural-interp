# How one neuron fires: the in-tfm clustering method

Implementation notes for [the thesis](../../thesis/index.md).

This doc explains, from the ground up, the method in Hariom's [narang99/in-tfm](https://github.com/narang99/in-tfm) repo and its result on the InceptionV1 neuron `mixed4e:55`. It assumes you know what a neural network is and roughly what a convolution does, and nothing else. [gemma3_embedding_dims.md](gemma3_embedding_dims.md) applies the same method to a language model. [glossary.md](../../thesis/glossary.md) is a short lookup table for every term defined here.

## The question

A neural network is built from many small units, each computing one number from the numbers below it. Interpretability research tries to say what each unit responds to. The simple hope is one unit, one concept: a "cat neuron", a "wheel neuron". In practice many units are **polysemantic**: one unit responds to several unrelated things. The classic example is `mixed4e:55` in InceptionV1, which fires for cats and also for cars.

Polysemanticity raises a sharper question. When the unit fires for a cat and when it fires for a car, does it fire *for the same reason*? Maybe one internal pathway handles both. Or maybe there are separate pathways, one carrying cat evidence and one carrying car evidence, that happen to end at the same unit. The **worlds hypothesis** says the second: a neuron fires through several separate "worlds", each a different kind of input reaching it by a different route.

The method answers this from data. It collects many moments when the neuron fires strongly, describes each moment by *which inputs to the neuron drove it*, and clusters those descriptions. If the worlds hypothesis holds, the descriptions fall into separate groups, and each group is one route.

## Words you need first

**Neuron.** Here, one scalar unit we study. InceptionV1 is a convolutional network, so a layer outputs a stack of 2D feature maps called **channels**. One channel at one spatial position is one number, and "neuron 55" means channel 55, read at any position.

**mixed4e:55.** InceptionV1 (also called GoogLeNet) has blocks named `mixed3a` up to `mixed5b`. Inside block `mixed4e` there is a 1x1 convolution, `mixed4e_1x1`, and channel 55 of its output is our neuron. We read it **pre-ReLU**, meaning before the ReLU that sets negatives to zero, so we keep the full value.

**Activation.** The neuron's value for one image at one position. A high activation means the neuron fires strongly there.

**Upstream channels.** The channels that feed the neuron. `mixed4e_1x1` reads 528 channels from the layer before it, and its output at a position is a weighted sum of those 528 numbers at that position (plus a bias).

## The pipeline

The method has eight steps. Steps 1 to 3 turn the neuron's behaviour into points with distances between them. Steps 4 to 6 cluster those points. Steps 7 and 8 draw them.

### 1. Hits: when does the neuron fire strongly?

The neuron is run over 2,000 images drawn from a pool of cats, cars and dogs. Every spatial position of every image is a candidate, so there are hundreds of thousands of activations. Most are weak and say little about what the neuron is for, so only the strongest are kept. Each kept image position is called a **hit**.

Where to cut is decided automatically. Sort the positive activations from high to low and plot them (the negative tail of the pre-ReLU values is left out, or it would drag the cut): the curve drops steeply, then flattens into a long tail. The **elbow** is where it bends. It is found with the Kneedle rule: draw a straight line from the first point of the curve to the last, and take the point furthest from that line (perpendicular distance). Everything before the elbow is a hit. In code this is `find_elbow_index_in_sorted_data` in `hit_selection/strategies/`. For mixed4e:55 the elbow keeps **14,240 hits**.

There is a second option, **bucketed** selection. The activation range between a floor and the maximum is split into equal-width bands, bands with too few activations are merged into a neighbour, and each band gets an equal share of the hit budget. The elbow keeps whatever lies above the cut, and since high activations are rare, the many moderate ones dominate. Bucketing guarantees the rare top activations are represented.

### 2. Hit vectors: what drove each hit?

Knowing *that* the neuron fired is not enough. We want *why*, so each hit is described by how much each upstream channel contributed.

Two quantities per upstream channel at the hit position:

- the channel's **activation**: how strongly that input was on;
- its **weight** in the neuron's row of the 1x1 conv: how much the neuron's value changes per unit of that channel. For a linear unit this is also the gradient, so the product is gradient times input.

Neither alone measures contribution. A channel can be strongly on but ignored by the neuron, or heavily weighted but switched off at this position. Their product captures both: on *and* used. Multiplying two vectors entry by entry is the **Hadamard product**, written `a ⊙ b`, with `(a ⊙ b)_i = a_i × b_i`.

Example with three upstream channels: activations `[0.5, 2.0, 0.0]`, weights `[1.0, -0.3, 4.0]`. The Hadamard product is `[0.5, -0.6, 0.0]`. Channel 1 pushed the neuron up by 0.5, channel 2 pushed it down by 0.6, channel 3 would matter a lot but was off, so it contributed nothing.

For mixed4e:55 each hit becomes a 528-long vector, its **hit vector** (config `clustering.on: hadamard`). Two hits with similar hit vectors were driven by the same upstream channels in the same proportions, so they share a route.

### 3. Distance: how alike are two hits?

Clustering needs a distance between hit vectors. The method uses **weighted Jaccard**, also called **Ruzicka** distance (`weighted_jaccard/`).

For two non-negative vectors x and y:

```
distance = 1 - Σ min(x_i, y_i) / Σ max(x_i, y_i)
```

The top sum is the mass the two vectors share, the bottom is the mass they have between them. Identical vectors give 0. Vectors with no channel in common give 1. It reads as "one minus the share of contribution they have in common".

Hit vectors have negative entries (channels that pushed the neuron down), and Jaccard needs non-negative numbers. So each vector is first split into a positive part `max(x, 0)` and a negative part `max(-x, 0)`, and the two are laid side by side. Pushing up and pushing down are then compared separately, and never cancel.

Worked example. x = `[2, -1, 0]`, y = `[1, -1, 3]`. Split: x becomes `[2, 0, 0 | 0, 1, 0]`, y becomes `[1, 0, 3 | 0, 1, 0]`. Sum of minimums: 1 + 1 = 2. Sum of maximums: 2 + 3 + 1 = 6. Distance = 1 - 2/6 = 0.67.

Why not plain Euclidean or cosine distance? Jaccard compares *which channels carry the mass and how much*, which is the question we are asking, and it is not dominated by one huge entry the way squared distances are. mixed4e:55 runs with `normalise: false`: vectors are not rescaled first, so a strong hit and a weak hit with the same pattern still count as different. PR 111 computes the distance through an L1 distance, which made it 18 times faster.

### 4. Clustering: the density-peak forest

Now there are 14,240 points and a distance between every pair. The goal is groups of points that sit close together. The method uses **density-peak clustering** (Rodriguez and Laio, 2014), implemented in `density_peaks/`. The idea: clusters are hills of density, each hill has a peak, and every point belongs to the hill it climbs.

**Copies.** Image patches repeat, so some hit vectors are identical. After the sign split, vectors are rounded to 6 decimals and identical ones are merged into one point carrying a count. Everything below treats a point with count 5 as five points stacked in one place.

**Nearest neighbours.** For each point, its **k nearest neighbours** (kNN) are the k points closest to it. mixed4e:55 uses k = 5.

**Density.** A point in a crowded region has its neighbours close by; a point in a sparse region has to reach far. So density is measured by the distance at which a point's neighbours, copies included, add up to k. Smaller distance means denser. The code works with log density, written **rho**: `-log10(d_k + c)`, where c is half the smallest nonzero d_k, so points at distance 0 stay finite.

**Parent and delta.** Points are ranked by density, and ties (common at distance 0) go to the point with more copies. Every point looks for the nearest point that is denser than itself. That point is its **parent**, and the distance to it is its **delta**. A typical point has a denser point right next to it, so its delta is small. A point at the top of a hill has to go far, across a valley to another hill, to find anything denser, so its delta is large.

**Forest.** Draw an arrow from every point to its parent. Following arrows always climbs density, so every chain ends at a point that stops. The arrows form a set of trees: a **forest**.

**Roots.** The points chosen as tops of trees, called **roots**. A good root is both dense (rho high) and far from anything denser (delta high). To pick them automatically, sort all rho values and find the elbow, sort all delta values and find the elbow, and keep points above both. These two thresholds are the **elbow root cuts**. Every other point keeps its parent link, and its chain of parents ends at exactly one root.

**Trees are clusters.** All points whose chain reaches the same root form one **tree**, and each tree is one cluster. Trees are numbered by size, largest first, so a cluster id is only meaningful within one run. A point's **depth** is the number of links up to its root.

A small picture: imagine two hills of points joined by a low ridge. Points on the left hill all climb to the left summit, points on the right hill to the right summit. The two summits are dense and far from anything denser, so both become roots, and the ridge splits where the climbing direction changes.

Why this and not HDBSCAN, the other common density method? HDBSCAN needs a low-density gap between clusters, so a continuous cloud with two bumps stays one cluster. Density peaks can split a continuous cloud, which matters because routes into a neuron may blend into each other. The cost is the reverse mistake: it cannot tell two genuinely separate hills from two bumps on one hill, so it can over-split.

### 5. Soft membership: probabilities

The forest gives each hit exactly one cluster, but some hits genuinely sit between two. PR 106 (`density_peaks/probabilities/`) gives every hit a probability for every cluster.

**Peak of a tree.** Not just the root but all of the tree's points that are at least `peak_share` (0.8) as dense as the root, the flat top of the hill.

**Geodesic distance.** Distance measured by walking along kNN graph edges (each point connected to its neighbours) rather than in a straight line. A straight line can cut across empty space; a walk has to follow the data, so crossing a sparse gap costs a lot.

**Probability.** For each hit, take the geodesic distance to each tree's peak and turn the distances into probabilities with a softmax: closer peaks get higher probability. The softmax temperature, which sets how sharply probability falls with distance, is one typical step, the median distance between a point and its nearest neighbour.

A hit that cannot reach any peak along the graph gets probability 1 for its own tree.

**Most probable cluster.** From here on a hit's cluster is its most probable one, not the tree it grew in. For mixed4e:55 these differ for 5,218 of 14,240 hits, so the reports, the touching pairs and everything after use the most probable cluster.

**Entropy.** How spread out a hit's probabilities are, in bits. A hit with 100% on one cluster has entropy 0. A hit split 50/50 between two has entropy 1.

**Confident and mixed hits** (PRs 107, 108). Hits with entropy under `max_entropy` (0.5 bits) are **confident** and appear in their cluster's report. The rest are **mixed** and go to a separate sheet, sampled across entropy levels so you see mildly and badly mixed hits. PR 113 adds which clusters **touch** (some mutual kNN edge, where two points are linked only if each is in the other's neighbour list, joins hits whose most probable clusters differ) and a report for each pair of clusters that hits are split between.

### 6. Buckets

PR 114 (`hit_buckets.py`) groups hits by how they are split, so you can look at each kind of hit on its own.

- **Cluster buckets.** Hits where one cluster has at least 0.9, 0.8 or 0.7 probability. These are nested: a 0.9 hit is also in the 0.8 and 0.7 buckets.
- **Set buckets.** Hits that need exactly the same set of clusters to reach 0.85 of their probability mass, for example "clusters 1 + 8". These are the hits living on the border between particular clusters. A hit that needs more than half of all clusters is in no set bucket, since it belongs nowhere in particular.

Buckets hold every qualifying hit, confident or not; entropy is only shown. Only buckets with at least 10 hits get a page.

### 7. The map

PR 115 (`bucket_graph.py`) draws the hits in 2D. The common approach is to squash the high-dimensional points into 2D with UMAP or t-SNE, which distorts distances in ways that are hard to read. This map projects nothing.

1. From every bucket of at least 10 hits, take 8 at evenly spaced ranks of the bucket's leading-cluster probability, so a set bucket runs from one side to the other. Add each cluster's root hit.
2. Build a kNN graph on the real distances, with edge weight one minus distance. Use the smallest k from 6, 10, 15, 25 that connects the whole graph.
3. Add a pull: an extra edge from each hit to the peak of every cluster where its probability is at least 0.1, with weight 3 times that probability.
4. Lay the graph out with `networkx.spring_layout` (300 iterations, seed 0), a force-directed layout: edges act like springs, all nodes push each other apart, and positions settle where the forces balance.

Without the pull the picture is one mixed blob, because a mixed hit's neighbours belong to many clusters. With it, confident hits gather around their peak, and a hit split between two clusters sits between their peaks.

### 8. Reports

PR 116 (`bucket_report.py`) writes a page per bucket. Each page shows the map with that bucket's sampled hits starred, and up to 6 of them, picked the same way, numbered with a **card** each. A card has the hit's DeepLift map and the mean Hadamard of its top 3 clusters, faded by probability. DeepLift is an attribution method that traces the activation back to input pixels, so a card shows which region of the image drove that hit. Each map is also drawn at four times the size for detail.

`scripts/collect_hit_space.py` runs steps 1 to 5 once and saves the hits, Hadamard rows, full distance matrix, forest, trees and probabilities to `<out_dir>/space/`. `hit_buckets.py` and `bucket_report.py` then read those saved arrays without rerunning the model.

## Result on mixed4e:55

About 41 clusters. The map shows two large worlds joined by a thin bridge. That is what the worlds hypothesis predicts: the neuron fires for cats and for cars through different sets of upstream channels, with a few hits in between. The full repro tar, including the elbow and bucketed reports, is in `gs://pj-okc/mixed4e_55/`.

The independent reproduction in this repo is [README.md](README.md), run from [mixed4e_55.ipynb](mixed4e_55.ipynb).

## Status

PRs 105 to 111 are merged. 113 to 118 are open and stacked: hit buckets (114), bucket graph (115), bucket report (116), one command for the linked reports from a saved space (117), and a token page per text neuron (118). The Gemma runs in this repo install in-tfm from the `feat/bucket-report` branch.

## Where this goes next

Hariom proposed carrying the method into a language model by treating the 640 dimensions of Gemma 3's embedding matrix as units. [gemma3_embedding_dims.md](gemma3_embedding_dims.md) does that: it first checks that the dimensions are real units, then runs this pipeline on single dimensions as if each were mixed4e:55.
