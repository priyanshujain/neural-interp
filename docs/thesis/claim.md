# Claim

[Index](index.md)

## Statement

- Neurons are polysemantic as wholes.
- Each activation of a neuron can still be assigned one meaning, at runtime, exactly.
- The unit of meaning is a (neuron, mode) pair. A mode is a cluster of the neuron's hits, and hits group by which inputs produced the activation, not by how large it is.
- Because the assignment is exact, circuits can be traced neuron by neuron through those modes.

## The question underneath

A network is built from units that each compute one number from the numbers below. The simple hope is one unit, one concept: a cat neuron, a wheel neuron. In practice many units are polysemantic. The standard example is InceptionV1 `mixed4e:55`, which fires for cats and for cars.

Polysemanticity raises a sharper question. When the unit fires for a cat and when it fires for a car, does it fire for the same reason? One internal pathway might handle both. Or separate pathways, one carrying cat evidence and one carrying car evidence, might end at the same unit. The worlds hypothesis says the second: a neuron fires through several separate worlds, each a different kind of input reaching it by a different route. If that is right, splitting each activation by its route recovers one meaning per activation.

## The conflict this addresses

**The neuron view.** The Circuits thread (Cammarata, Olah and others, 2020) shows individual neurons, such as curve detectors in InceptionV1, can be understood and traced into circuits. Neurons that respond to unrelated inputs are acknowledged and set aside as polysemantic.

**The features view.** Superposition (Elhage et al., 2022) and dictionary learning (Bricken et al., 2023) argue that networks represent more features than they have neurons. So neurons are the wrong unit, and features have to be learned as directions, for example with sparse autoencoders. Neel Nanda and others work in this framing. Nanda's team at Google DeepMind has since deprioritised SAE research after negative results on downstream tasks (Smith, Rajamanoharan, Nanda et al., 2025).

**The gap.** The neuron view has no account of polysemantic neurons. The features view gives up on neurons and needs a learned dictionary, which is approximate and is not the model's own computation.

## Our position

Polysemanticity is a property of the neuron as a whole. It disappears once each activation is split by its cause. No dictionary is learned, and every quantity is an exact sum the model computes.

This differs from the features view in two ways that matter for publication. The decomposition is exact, so there is no reconstruction error to explain away. And the unit stays the model's own neuron, so a mode found in one layer points at specific neurons in the previous layer, which is what makes tracing possible.

## What the claim does not yet cover

The claim is about activations that land in a mode. Many hits land in none (see [open_problems.md](open_problems.md)), and the claim has to either cover them or say what they are before it can be called exhaustive.
