# Open problems

[Index](index.md)

## Sparse hits

Many hits fall outside every mode: 50 to 95% were HDBSCAN noise. The hypothesis was that sparse hits are unused and can be dropped.

Current evidence does not support it.

- Density tracks how frequent a token is, not how central a hit is to a meaning (attention heads).
- The sparse hits on induction heads are real name induction, used a little rather than not at all ([attention.md](attention.md)).
- On Gemma token rows, density predicts geometry but not meaning ([density_and_meaning.md](density_and_meaning.md)).

Next: density within a single token, and density matched on token frequency.

## Exhaustiveness

A claim of monosemanticity has to cover most hits, or say what the rest are. The noise problem threatens this directly. The density-peak forest assigns every hit to a mode, and soft membership marks the ones in between, which moves the question from "what is the noise" to "are the low-density modes and mixed hits meaningful".

## Mode boundaries

HDBSCAN splits modes at the base of the activation ramp, even where Hadamard products look alike. Meaning may be continuous: peaks with shared slopes. The forest with soft membership is the current answer, since it can split a continuous cloud and reports mixed hits explicitly. Its cost is over-splitting: about 41 modes for two worlds on mixed4e:55, and hundreds of trees per Gemma dimension. A criterion for merging modes with the same meaning is needed.

## Scale

So far: one vision neuron, three heads in a 2-layer model, small MNIST nets, and Gemma 3 embedding dimensions as a stand-in. A publishable claim needs an MLP neuron in a real language model, such as Gemma 3 `down_proj` in layer 0 or 1, run with nothing substituted: hits from real WikiText activations, Hadamard hit vectors, weighted Jaccard.

## Candidate first LLM neuron: refusal neurons

Kazemi et al. (Apple, NeurIPS 2026, [ongoing-research.md](ongoing-research.md)) publish single MLP neurons in Qwen3 and Llama-3.1 that are causally sufficient for refusal, with code. Running our method on one of them is the cheapest route to the Scale requirement: a real LLM neuron with a known causal effect already in hand. The question is whether refusal is one mode, or several harm categories gated through separate routes. If modes exist, ablating one mode instead of the whole neuron should bypass refusal for its category only, which is also the mode-level ablation [publication.md](publication.md) asks for.

## Novelty

Clustering per-activation input contributions to split a polysemantic neuron is already published for vision: PURE (Dreyer et al., 2024) and SPICE (Lee et al., ECCV 2026). The thesis has to be positioned as extending that line. What separates us: the exact Hadamard split, density-peak modes under weighted Jaccard, bilinear attention units, language models, and tracing circuits through modes. Details in [ongoing-research.md](ongoing-research.md).

## The flagship example is contested

Cammarata (Goodfire, 2026) reads mixed4e:55 as one V-shape detector, not a cat and car neuron. See [vision.md](vision.md) for what we need to show.

## Records still missing

- The attention results are HDBSCAN-era and not rerun with the forest.
- The Gemma token-row meaning curves need the corrected rerun.
- The MNIST results are not written up here.
