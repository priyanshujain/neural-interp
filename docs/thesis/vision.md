# Vision: InceptionV1 mixed4e:55

[Index](index.md)

## Why this neuron

`mixed4e:55` is channel 55 of the pre-ReLU `mixed4e_1x1` conv in InceptionV1 (GoogLeNet). It is known to respond to both cats and cars, so it is the cleanest test of whether a polysemantic neuron splits into modes with one meaning each. Original analysis: Hariom's blog.

## Setup

- Inputs: 2,000 images drawn from a pool of cats, cars and dogs. Every spatial position of every image is a candidate activation.
- Hits: elbow cut over the positive activations, giving 14,240 hits. A bucketed run was also made.
- Hit vector: 528-long Hadamard product of upstream activations and their attributions.
- Distance: sign-split weighted Jaccard, not normalised.
- Modes: density-peak forest, k = 5, elbow root cuts, then soft membership, buckets and maps as in [method.md](method.md).

## Result

About 41 modes. The map shows two large worlds joined by a thin bridge. Cat hits and car hits are driven by different sets of upstream channels, with a few hits in between. Per-mode DeepLift cards show which image region drove each hit, and modes read as one concept each.

This is what the claim predicts: the neuron is polysemantic as a whole, and the split by upstream contribution separates its meanings.

## Caveats to record

- The pool contains only cats, cars and dogs, so modes for other content cannot appear. A broader image set would test whether more worlds exist.
- 41 modes for two visible worlds means the forest over-splits each world into sub-modes. Whether those sub-modes are distinct meanings or bumps on one hill has not been checked.
- Mode-level ablation has not been run on this neuron.
- Cammarata ([Neural Networks Are Already Clean](https://already-clean.goodfire.com/), Goodfire, 2026) rereads this neuron as a single V-shape detector, positive on pointed shapes (flames, fireboat jets, fox faces) and negative on round ones. If he is right, the cat and car worlds share one cause and our modes would be sub-shapes of one meaning, not two meanings. We need to show the modes differ in upstream cause, for example by checking whether cat-mode and car-mode hits load on the same V-shape channels.

## Artifacts

Elbow and bucketed reports, and a full reproduction archive. Paths and the reproduction notebook are in [../implementation/mixed4e_55/in_tfm_neuron_clustering.md](../implementation/mixed4e_55/in_tfm_neuron_clustering.md).
