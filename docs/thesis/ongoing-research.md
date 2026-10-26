# Ongoing research

[Index](index.md)

Active work from 2024 to October 2026 that overlaps, competes with or supports the (neuron, mode) claim. Every entry below was checked against its arXiv or publisher page. Novelty risk is rated low, medium or high.

## Splitting polysemantic neurons by input attribution

**PURE** (Dreyer, Purelku, Vielhaben, Samek, Lapuschkin, 2024). [arXiv:2404.06453](https://arxiv.org/abs/2404.06453). Splits a polysemantic ResNet neuron into "virtual" pure neurons by clustering per-example attributions from the previous layer with k-means, one circuit per meaning. This is our method in outline: per-hit upstream contribution vectors, clustered, one meaning per cluster. Differences: they use an attribution method, not the exact Hadamard split; fixed k-means; vision only; no claim that modes are the unit for circuit tracing or a replacement for SAEs. Risk: high, must be cited and compared against directly.

**SPICE** (Lee, Kwon, Lee, Choi, 2026, ECCV 2026). [arXiv:2609.13198](https://arxiv.org/abs/2609.13198). Computes attribution footprints from the preceding layer for each activation of a neuron and clusters them iteratively, choosing the number of clusters automatically, across CNNs and vision transformers. Also studies how polysemanticity forms through distinct computational pathways, which is our worlds hypothesis in other words. Differences: vision only, an attribution method rather than an exact sum, no attention or LLM neurons, no mode-level circuit tracing, no SAE comparison. Risk: high. It is the closest current work and appeared two months ago.

**NeuronScope** (2026). [arXiv:2601.03671](https://arxiv.org/abs/2601.03671). LLM agents decompose a Llama-3.1-8B neuron's exemplars into atomic conditions and cluster them (HDBSCAN on text embeddings). It splits a neuron into meanings by the text of its top activations, not by which inputs caused them. Risk: low, but it shows the "split a neuron into senses" framing is now common.

**Polysemanticity or Polysemy?** (Ait Hou, Hwa, 2026). [arXiv:2604.00443](https://arxiv.org/abs/2604.00443). Argues some neuron polysemanticity is a lexical confound (one word form, several senses) rather than compressed concepts. Supports our point that whole-neuron polysemanticity overstates the problem. Risk: low.

## The neuron basis revival

**Language Model Circuits Are Sparse in the Neuron Basis** (Arora, Wu, Steinhardt, Schwettmann, Transluce, 2026). [arXiv:2601.22594](https://arxiv.org/abs/2601.22594), [blog](https://transluce.org/neuron-circuits). Argues MLP neurons are as sparse a basis as SAE latents and traces circuits over neurons with gradient attribution; about 100 neurons control subject-verb agreement. Strong support for keeping the model's neuron as the unit, and the main competitor on "no dictionary needed" for LLMs. It does not split a neuron's activations into modes, so polysemantic neurons stay polysemantic there. Risk: medium.

**Scaling Automatic Neuron Description** (Choi, Huang, Meng, Johnson, Steinhardt, Schwettmann, Transluce, 2024). [transluce.org/neuron-descriptions](https://transluce.org/neuron-descriptions). Descriptions for every neuron of Llama-3.1-8B-Instruct from a fine-tuned explainer. One description per whole neuron; our modes could give one description per mode instead. Risk: low; useful as a baseline and labelling tool.

**Universal Neurons in GPT2 Language Models** (Gurnee et al., 2024). [arXiv:2401.12181](https://arxiv.org/abs/2401.12181). 1 to 5% of neurons recur across seeds and tend to be interpretable. Support for neuron-level analysis. Risk: low.

**A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models** (Kazemi, Chegini, Safi, Apple, NeurIPS 2026). [arXiv:2605.08513](https://arxiv.org/abs/2605.08513), [code](https://github.com/apple-aiml-research/ml-refusal-neurons). Finds refusal neurons that gate whether harmful knowledge is expressed and concept neurons that encode it. Suppressing one refusal neuron bypasses refusal, and amplifying one concept neuron induces harmful content from innocent prompts, across seven Qwen3 and Llama-3.1 models from 1.7B to 70B with no training. Neurons are found by gradient times activation on a refusal log-odds loss, which per neuron is the same quantity as our Hadamard entry. It narrows Arditi et al. 2024's single refusal direction to a single neuron. The concept-neuron result is shown rigorously only for suicide-related content. Causal evidence at scale that a single MLP neuron is a meaningful, controllable unit. It treats each neuron as one job and never asks whether its activations split into modes. Community follow-ups steer single neurons the same way in Qwen3.5 4B ([example](https://x.com/paahrsa/status/2114442712233681027)). Risk: low; a ready-made target for our method.

## Weight and parameter based interpretability

**Bilinear MLPs** (Pearce, Dooms, Rigg, Oramas, Sharkey, ICLR 2025). [arXiv:2410.08417](https://arxiv.org/abs/2410.08417). Gated units without elementwise nonlinearity are a third-order tensor, so eigendecomposition of weights gives interpretable features and circuits without data. Closest in spirit to our exact bilinear attention decomposition. It decomposes weights globally, we decompose each activation and cluster. Risk: medium for the bilinear part of method.md, low for the claim.

**APD** (Braun, Bushnaq, Heimersheim, Mendel, Sharkey, Apollo, 2025). [arXiv:2501.14926](https://arxiv.org/abs/2501.14926). **SPD** (Bushnaq, Braun, Sharkey, Goodfire, 2025). [arXiv:2506.20790](https://arxiv.org/abs/2506.20790). **VPD, Interpreting Language Model Parameters** (Bushnaq, Braun, Clive-Griffin, Bussmann, Hu, Ivanitskiy, Linsefors, Sharkey, 2026). [arXiv:2610.20748](https://arxiv.org/abs/2610.20748), [Goodfire](https://www.goodfire.com/research/interpreting-lm-parameters). Decompose weight matrices into rank-one subcomponents, few of which are needed per input; VPD scales this to a 67M language model with adversarial ablation. A rank-one subcomponent active on a subset of inputs is close to a (neuron, mode) pair. Differences: theirs is learned by optimisation over parameters and not tied to existing neurons; ours is read off the forward pass with no training. Risk: medium to high conceptually. This is the stronger competing answer to "mechanisms without SAEs", and the comparison must be made explicitly.

**Weight-sparse transformers have interpretable circuits** (Gao, Rajaram, Coxon, Govande, Baker, Mossing, OpenAI, 2025, ICML 2026). [arXiv:2511.13653](https://arxiv.org/abs/2511.13653). Trains models with mostly zero weights so neurons become interpretable and circuits small. Changes the model instead of explaining a dense one. Risk: low; supports the neuron as a unit.

**Interference weights** (Olah, Turner, Conerly, Anthropic, 2025). [A Toy Model of Interference Weights](https://transformer-circuits.pub/2025/interference-weights/index.html). Weights that exist only as a compromise between superposed circuits, which blocks reading global circuits from weights. Relevant risk to our dependencies step: some mode-to-mode links may be interference. Risk: low for novelty, real as an objection.

## Transcoders and attribution graphs

**Transcoders** (Dunefsky, Chlenski, Nanda, NeurIPS 2024). [arXiv:2406.11944](https://arxiv.org/abs/2406.11944). Replace an MLP with a wider sparse MLP so circuits factor into input-dependent and input-invariant terms. **Circuit Tracing** (Ameisen et al., Anthropic, 2025) [methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and **On the Biology of a Large Language Model** (Lindsey et al., 2025) [biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). Cross-layer transcoders build a replacement model, and per-prompt attribution graphs are linear edge sums between features. Their edges are the same kind of exact linear contributions we use, but between learned features in a replacement model with error nodes. We do it in the real model on real neurons. Risk: medium; this is the main thing our circuit tracing claim must beat or match.

## SAE critiques and evaluations

**Negative Results for SAEs on Downstream Tasks** (Smith, Rajamanoharan, Conmy, McDougall, Lieberum, Kramár, Shah, Nanda, GDM, 2025). [LessWrong](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks). SAE probes lose to dense linear probes out of distribution; the team deprioritised SAEs. This is the Nanda citation claim.md marks as needing checking. **A Pragmatic Vision for Interpretability** (Nanda et al., 2025) [Alignment Forum](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) pivots GDM from ambitious reverse engineering to proxy tasks. Support, and a warning: reviewers in this camp will ask what task modes make better.

**SAEs Trained on the Same Data Learn Different Features** (Paulo, Belrose, EleutherAI, ICLR 2026). [arXiv:2501.16615](https://arxiv.org/abs/2501.16615). Only about 30% of latents are shared across seeds. Strong support: our modes are deterministic given the hit set and clustering. Worth showing mode stability across hit samples to make the contrast.

**Decomposing the Dark Matter of SAEs** (Engels, Riggs, Tegmark, 2024). [arXiv:2410.14670](https://arxiv.org/abs/2410.14670). Characterises SAE reconstruction error. Supports our "no reconstruction error" argument.

**SAEBench** (Karvonen et al., ICML 2025). [arXiv:2503.09532](https://arxiv.org/abs/2503.09532). Eight metrics and 200+ SAEs. The likely evaluation suite for the needed SAE comparison. **Matryoshka SAEs** (Bussmann, Nabeshima, Karvonen, Nanda, ICML 2025). [arXiv:2503.17547](https://arxiv.org/abs/2503.17547). Nested dictionaries reduce feature absorption; the strongest SAE baseline to compare against.

## Other related directions

**Monet** (Park, Jin, Kim, Kang, ICLR 2025). [arXiv:2412.04139](https://arxiv.org/abs/2412.04139). Up to 262k experts per layer trained end to end to be monosemantic. Builds the split into the architecture; we find it post hoc. Risk: low.

**What Causes Polysemanticity?** (Lecomte, Thaman, Schaeffer, Bashkansky, Chow, Koyejo, 2023/2024). [arXiv:2312.03096](https://arxiv.org/abs/2312.03096). Polysemanticity can arise incidentally from initialisation even without capacity pressure. Compatible with modes: incidental merging of features into one neuron would leave separable input routes.

**Open Problems in Mechanistic Interpretability** (Sharkey et al., 2025). [arXiv:2501.16496](https://arxiv.org/abs/2501.16496). Field review covering limits of SAEs and parameter decomposition. Useful framing citation.

## Novelty risks

- **SPICE (2026) and PURE (2024)**: same core move, cluster per-activation upstream attributions to split a polysemantic neuron. Ours differs by using the exact Hadamard decomposition (sums to the pre-activation, no attribution rule), density-peak clustering with weighted Jaccard on signed parts, extension to attention scores and LLM units, and mode-level circuit tracing. Both are vision only. The claim must be repositioned as extending and scaling this line, not inventing it.
- **APD, SPD, VPD (2025 to 2026)**: competing dictionary-free mechanism decomposition. Ours needs no optimisation and keeps the model's neurons; theirs is learned and can span neurons. A head-to-head on the same small model would settle which is more useful.
- **Apple single refusal neuron (2026)**: shows single LLM neurons are causally sufficient for refusal, but treats each as one job. Ours asks whether that job splits into modes.
- **Transluce neuron-basis circuits (2026)**: competing "neurons suffice" claim for LLMs. Ours adds the split of polysemantic neurons into modes, which they do not do.
- **Attribution graphs with CLTs (2025)**: the incumbent for circuit tracing. Our edge is no replacement model and no error nodes; this is unproven until the mode-level circuit trace is done.
