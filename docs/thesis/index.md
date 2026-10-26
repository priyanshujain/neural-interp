# Monosemanticity at the neuron level from exact input decompositions

Work in progress. Authors: Hariom Narang, Priyanshu Jain.

This is the running record of the thesis. Every claim, method choice, result, control, negative result and known bug goes here, so that at publication each statement can be traced to the experiment that supports it. How each experiment was run (commands, code, output paths) is in [../implementation/](../implementation/).

## The claim in brief

Neurons are polysemantic as wholes. Each activation of a neuron can still be given one meaning at runtime, exactly. The unit of meaning is a (neuron, mode) pair, where a mode is a cluster of the neuron's hits grouped by which inputs produced the activation. Because the split is an exact sum the model computes, circuits can be traced neuron by neuron through modes, without a learned dictionary.

## Contents

1. [claim.md](claim.md): the claim, the neuron view versus the features view, and our position.
2. [prior-work.md](prior-work.md): the established work we build on or argue against.
3. [ongoing-research.md](ongoing-research.md): active research close to ours, and novelty risks.
4. [sota.md](sota.md): the state of the art in LLM interpretability and where our method fits.
5. [positioning.md](positioning.md): where our idea stands, strengths, gaps, and what blocks acceptance at a main venue.
6. [method.md](method.md): exact decomposition, hits, distance, modes, soft membership, dependencies, bilinear units and maps, with the reason for each choice.
7. [vision.md](vision.md): InceptionV1 mixed4e:55, the cat and car neuron.
8. [attention.md](attention.md): the previous-token head and induction heads in attn-only-2l.
9. [gemma_embedding_dims.md](gemma_embedding_dims.md): Gemma 3 embedding dimensions as units, and single dimensions run as neurons.
10. [density_and_meaning.md](density_and_meaning.md): whether density inside a cluster tracks meaning, on Gemma 3 token rows and WikiText occurrences.
11. [open_problems.md](open_problems.md): sparse hits, exhaustiveness, mode boundaries and scale.
12. [publication.md](publication.md): what would make the claim publishable, and questions to settle.
13. [references.md](references.md): cited work.
14. [glossary.md](glossary.md): every term, as used here.

## Status of the evidence

| Experiment | Unit | Exact decomposition | Status |
|---|---|---|---|
| mixed4e:55 | Vision conv channel | Yes | Done, about 41 modes |
| attn-only-2l L0H7 | Attention score | Yes, bilinear | Done |
| attn-only-2l L1H10, L1H4 | Attention score | Yes, bilinear | Done, modes with ablations |
| Gemma 3 W_E dims | Embedding column | No, stand-in | Done, supporting only |
| Gemma 3 token rows | Token embedding | No | Done, geometry yes, meaning no |
| Gemma 3 MLP neuron | MLP neuron | Yes | Not started, needed |
| SAE comparison | | | Not started, needed |
| Mode-level circuit trace | | | Not started, needed |
