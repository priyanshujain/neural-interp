# Positioning

[Index](index.md)

Where the idea stands against the field as of October 2026, what is good about it, what is missing, and what blocks acceptance at a main venue. Sources are in [prior-work.md](prior-work.md), [ongoing-research.md](ongoing-research.md) and [sota.md](sota.md).

## Where it sits

The field has three answers to polysemanticity. Learn a new basis (SAEs, transcoders, cross-layer transcoders and attribution graphs). Learn a decomposition of the weights (APD, SPD, VPD). Or argue neurons are already fine (Transluce's neuron-basis circuits, Cammarata's "already clean", Apple's single refusal neurons). Ours is a fourth: keep the neuron, accept that it is polysemantic, and resolve the mixture per activation by its exact input contributions.

Within that, the nearest work is PURE (2024) and SPICE (ECCV 2026), which already cluster per-activation upstream attributions to split vision neurons. So the honest placement is: an exact, density-based, cross-architecture extension of the PURE and SPICE line, aimed at language models and at mode-level circuit tracing. "First to split neurons into meanings by their causes" is not a claim we can make.

## What is good about it

**Exact and in the real model.** Each hit vector sums to the neuron's pre-activation. No reconstruction error, no dark matter, no replacement model, no error nodes. This answers the faithfulness objection that hangs over SAEs and CLTs, and the current mood (GDM deprioritising SAEs, random SAEs scoring like trained ones, SAE features unstable across seeds) favours it.

**No training.** Modes are read off a forward pass and a clustering. Cheap, deterministic given the hits, and applicable to any already-trained model, unlike weight-sparse training or parameter decomposition.

**Keeps the unit the field is returning to.** Transluce, Goodfire and Apple all now argue the neuron is a meaningful, controllable unit. None of them splits a polysemantic neuron. We are the natural next step for that camp.

**Built for tracing.** A mode names the previous-layer units that drive it, so circuits can be followed one mode at a time. That is the attribution-graph idea without a learned replacement model.

**Covers attention.** The exact bilinear split of attention scores in the singular basis of `W_Q^T W_K` is something PURE and SPICE do not have, and attention is the acknowledged weak spot of attribution graphs.

**Early results are the right kind.** On L1H4 and L1H10, modes differ sharply in downstream use (ablating one mode raises loss by a median 5.6, another is barely read). Mode identity predicting causal effect is the strongest single result we have.

## Gaps

**Novelty.** PURE and SPICE make the core move. Without a direct comparison showing the exact split or the clustering gives better modes, a reviewer will call this incremental.

**No LLM MLP neuron.** All language-model evidence so far is attention heads in a 2-layer toy model, plus Gemma embedding dimensions that do not use the method (cosine on embedding rows, no decomposition). The main claim has not been tested on the unit it is about.

**The exactness claim is narrower than it sounds in transformers.** For an MLP neuron the input is the residual stream after normalisation, not the outputs of previous neurons. A Hadamard split over residual dims is exact but over an arbitrary basis, and weighted Jaccard on it is not rotation invariant, which is the objection our own rotated control was built for. To keep both exactness and meaning, the split has to be over upstream components (each earlier neuron and head's write into the residual, read through the neuron's input row), with the normalisation scale frozen per token. Gated MLPs (Gemma's GeGLU) make each neuron a product of two dot products, so they need the bilinear treatment, not the linear one. The "dependencies are neurons of the previous layer" statement in [method.md](method.md) is true for InceptionV1 and false as written for a transformer.

**Meaning is not exact.** The decomposition is exact, the modes are not: they depend on hit selection, k, the root cut and the clustering algorithm. Results so far changed algorithm midway (HDBSCAN for attention, forest for vision), and the forest over-splits (about 41 modes for two worlds, hundreds per Gemma dimension). There is no stability analysis across seeds, hit samples or k.

**Coverage.** 50 to 95% of hits fell outside every mode under HDBSCAN, and the evidence says those hits are used. Monosemanticity that covers a minority of activations is not monosemanticity.

**No testable definition.** "One mode, one cause, one effect" is a candidate, not yet a measured criterion with a threshold.

**No baselines.** No SAE, transcoder, PURE, SPICE or SPD comparison, and no standard benchmark (SAEBench, MIB). Reviewers at ICML and NeurIPS now expect at least one.

**No causal validation at scale.** Mode-level ablation exists for two heads only. No mode-level circuit end to end, and no recovery of a known circuit (IOI, induction) as a sanity check.

**The flagship example is contested.** Cammarata now reads mixed4e:55 as one V-shape detector. Leading with it invites the reply that our "worlds" are facets of one feature.

**Labelling is manual.** Modes are read by hand, while the field labels with agents and scores labels automatically.

## What stops acceptance at ICML or NeurIPS today

In order of severity:

1. No result on an MLP neuron in a real LLM. This alone is disqualifying for a paper about LLM interpretability.
2. Novelty against PURE and SPICE not established by experiment.
3. No baseline against SAEs or transcoders on the same layer.
4. No quantitative evaluation: coverage, mode stability, and causal effect per mode.
5. The transformer exactness issue above, which a careful reviewer will find.

The current evidence is a credible workshop paper (for example a mechanistic interpretability workshop at ICML or NeurIPS), not a main-track paper.

## Minimum to make it a main-track paper

- **One real LLM, several neurons.** Gemma 3 or Qwen3 MLP neurons, decomposed over upstream components as above. Apple's published refusal neurons are the best first target: the causal effect is already known, so per-mode ablation (does ablating one mode bypass refusal for only its harm category?) is a clean, striking result.
- **Coverage and stability numbers.** Fraction of hits and of downstream effect in confident modes; mode agreement across hit resamples and k.
- **Causal validation.** Mode ablation versus whole-neuron ablation, with targeted effects. One known circuit recovered through modes.
- **Baselines.** An SAE or transcoder on the same layer (Gemma Scope exists), and PURE or SPICE run on our vision neuron and ours run on theirs.
- **Lead with the new results,** keep mixed4e:55 as motivation and address Cammarata's reading directly.

## Our strongest framing

Neurons are back as the unit, but nobody has a method to split the polysemantic ones in a language model exactly and use the split for circuits. SAEs are approximate and unstable, parameter decomposition needs heavy optimisation, and PURE and SPICE are vision only and approximate. We resolve each activation exactly in the real model, without training, and show that the resulting modes predict causal effect.
