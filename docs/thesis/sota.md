# State of the art in LLM interpretability

[Index](index.md)

As of October 2026.

## Where the field stands

Three years after dictionary learning took over the field ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features); [Templeton et al., 2024, Scaling Monosemanticity](https://www.anthropic.com/research/mapping-mind-language-model), millions of SAE features from Claude 3 Sonnet), the field has split. One camp still pursues complete reverse engineering, now mostly through replacement models (transcoders, cross-layer transcoders, attribution graphs), parameter decomposition and models trained to be interpretable. The other camp, led by Google DeepMind, has moved to "pragmatic" interpretability: pick a safety problem, measure progress on a proxy task, start with probes and steering, and add complexity only when simple baselines fail. Meanwhile there is a growing return to the model's own neurons as a unit, which matters directly for us.

The shared starting point is still [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496) (Sharkey, Chughtai et al., 2025, TMLR), which lists the open problems in methods, applications and socio-technical work from about 30 researchers across 18 organisations.

## Leading methods

**Sparse autoencoders and variants.** SAEs learn an overcomplete dictionary whose sparse codes reconstruct an activation. Current practice is JumpReLU or TopK activations with Matryoshka losses to reduce feature absorption. The case against them has hardened. GDM's [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks) (Smith, Rajamanoharan, Conmy et al., March 2025) found SAE probes lost to dense linear probes on out-of-distribution harmful-intent detection. [Sanity Checks for Sparse Autoencoders](https://arxiv.org/abs/2602.14111) (Korznikov et al., February 2026) reports that on synthetic data with known features SAEs recover only 9% of true features at 71% explained variance, and that SAEs with random directions or activation patterns score about the same as trained ones on interpretability (0.87 vs 0.90), sparse probing and causal editing. The unexplained residual, "dark matter", is studied in [Engels, Riggs and Tegmark (2024)](https://arxiv.org/abs/2410.14670): over 90% of the error norm is linearly predictable from the activation, and larger SAEs fail on the same contexts as smaller ones.

**Transcoders and cross-layer transcoders.** A transcoder replaces an MLP with a sparse, wider MLP trained to imitate its input-output map ([Dunefsky, Chlenski and Nanda, 2024](https://arxiv.org/abs/2406.11944)). Cross-layer transcoders (CLTs) let a feature read from one layer and write to all later ones. They are the substrate for attribution graphs.

**Attribution graphs and circuit tracing.** Anthropic's [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Ameisen et al., March 2025) swaps the model's MLPs for a CLT replacement model, freezes attention and normalisation for one prompt, and draws a linear graph of feature-to-feature effects, with error nodes for what the replacement misses. The companion [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Lindsey et al., 2025) applied it to Claude 3.5 Haiku (planning in rhymes, multi-step reasoning, refusals). Attention was the known gap; [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html) (Kamath et al., July 2025) started on QK circuits. Graphs only partially explain a prompt, and each needs manual reading.

**Parameter decomposition.** Instead of decomposing activations, decompose weights into components that sum to the original parameters, of which few are needed per input. [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790) (Bushnaq, Braun and Sharkey, 2025) made this trainable on small models. [Interpreting Language Model Parameters](https://arxiv.org/abs/2610.20748) (Bushnaq, Braun, Clive-Griffin et al., Goodfire, 2026) introduces adversarial parameter decomposition (VPD), trains components to stay faithful under adversarial ablation, and uses component interactions as a new kind of circuit analysis.

**Interpretable-by-construction models.** OpenAI's [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653) (Gao, Rajaram, Coxon, Govande, Baker and Mossing, November 2025) trains models with most weights zero. Pruned task circuits are about 16 times smaller than in dense models of equal loss, and are necessary as well as sufficient. Scaling past tens of millions of nonzero parameters while keeping interpretability is unsolved, and an Anthropic Fellows replication argues the circuits [may be interpretable yet unfaithful](https://www.lesswrong.com/posts/sHpZZnRDLg7ccX9aF/weight-sparse-circuits-may-be-interpretable-yet-unfaithful).

**Neurons again.** Transluce's [Language Model Circuits Are Sparse in the Neuron Basis](https://arxiv.org/abs/2601.22594) (Arora, Wu, Steinhardt and Schwettmann, 2026, ICML spotlight) reports MLP neurons are about as sparse a basis as SAEs, and finds causal neuron circuits with gradient attribution and no training (about 100 neurons control subject-verb agreement). Their [ADAG](https://arxiv.org/abs/2604.07615) automates labelling of attribution graph clusters. Goodfire's essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) (Cammarata, October 2026) goes further: it rereads our own example, InceptionV1 mixed4e:55, as a single V-shape detector rather than a cat and car superposition, and estimates about 90% of Qwen3-32B neuron poles have one main job. It is an informal, unreviewed essay with preliminary numbers, and Nanda publicly called it interesting but not a big-picture update ([The Decoder, October 2026](https://the-decoder.com/is-ai-interpretability-close-to-a-breakthrough-a-new-essay-has-researchers-divided/)).

**Probing, steering and patching.** These remain the workhorses and the baselines everything else must beat. Activation patching is the standard causal test ([Heimersheim and Nanda, 2024](https://arxiv.org/abs/2404.15255)). Contrastive activation addition is the standard steering method ([Panickssery et al., 2023](https://arxiv.org/abs/2312.06681)). In 2026 Goodfire shipped probe-based products: [hallucination probes as RL rewards](https://www.goodfire.com/research/rlfr) and [reward-hacking monitors](https://goodfire.com/research/reward-hacking-activation-monitors).

**Verbalised readouts.** Anthropic's newest tools explain activations in language instead of features. [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) (Fraser-Taliente, Kantamneni, Ong et al., May 2026) train a verbaliser and a reconstructor so the explanation text must rebuild the activation; in a hidden-motive audit, auditors with NLAs won 12 to 15% of the time against under 3% without, though explanations can contain invented details. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) (Gurnee et al., July 2026) uses a Jacobian lens to find a small "J-space" of reportable concepts in Claude and tests it by swaps and ablations.

| Method | What it gives | Exact? | Cost | Main weakness |
|---|---|---|---|---|
| SAE | Sparse feature directions per activation | No, reconstruction error | High training, per layer | Misses true features, random baselines comparable, dark matter |
| Transcoder / CLT | Sparse replacement of MLPs, feature-to-feature edges | No, error nodes | Very high for CLTs | Replacement model is not the model |
| Attribution graph | Per-prompt causal graph of features | Linear and exact only for the frozen replacement | Moderate per prompt, needs CLT | Partial, manual reading, attention weak |
| Parameter decomposition | Weight components used per input | Components sum to weights, behaviour approximate | Very high, small models so far | Scaling |
| Weight-sparse training | Small circuits in a new model | Exact within that model | Retrain from scratch | Not the deployed model, capability cost, faithfulness questioned |
| Neuron attribution | Causal neuron sets per task | Gradient approximation | Cheap, no training | Polysemantic neurons left unsplit |
| Probe | Detector for one concept | N/A, supervised | Cheap | Needs labels, says nothing about mechanism |
| Steering | Behaviour control along a direction | N/A | Cheap | Side effects, no explanation |
| Patching | Causal effect of a component | Exact intervention | Cheap per hypothesis | Needs a hypothesis and counterfactual design |
| NLA / J-lens | Text readout of activations | No | High | Confabulation |

## Largest models interpreted

Anthropic works on production Claude models: Claude 3 Sonnet with SAEs (2024), Claude 3.5 Haiku with attribution graphs (2025), and in 2026 [emotion concepts](https://transformer-circuits.pub/2026/emotions/index.html) (Sofroniew et al.), NLAs and the global workspace work on current Claude models. None of these weights are public. For open models, [Gemma Scope 2](https://www.alignmentforum.org/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2) (McDougall, Conmy, Kramár, Lieberum, Rajamanoharan and Nanda, December 2025) gives SAEs on residual, MLP and attention outputs plus transcoders at every layer of every Gemma 3 model from 270M to 27B, pretrained and instruction tuned, with CLTs on 270M and 1B only. [Llama Scope](https://arxiv.org/abs/2410.20526) (He et al., 2024) gives 256 TopK SAEs on every layer of Llama-3.1-8B. Note the pattern: full CLTs, the basis for circuit tracing, exist publicly only for small models.

## Automated interpretability and agents

Auto-interp began with [Bills et al. (2023)](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html), where GPT-4 explained GPT-2 neurons and was scored by simulation. Transluce's [neuron description pipeline](https://transluce.org/neuron-descriptions) (Choi et al., 2024) fine-tuned explainers to describe every Llama-3.1-8B MLP neuron, and ADAG extends this to circuits. Anthropic's [automated auditing agents](https://alignment.anthropic.com/2025/automated-auditing/) (2025) use interpretability tools inside agentic audits. Cammarata credits agents for most of the Goodfire neuron work. The trend is that labelling and hypothesis testing are now done by models, and the bottleneck has moved to verifying that labels are faithful.

## Evaluation and benchmarks

[SAEBench](https://arxiv.org/abs/2503.09532) (Karvonen, Rager, Lin et al., 2025, ICML) scores SAEs on eight metrics including probing, disentanglement and unlearning, and found that proxy metrics like reconstruction do not predict practical value. [MIB](https://arxiv.org/abs/2504.13151) (Mueller, Geiger et al., 2025, ICML) has a circuit localisation track and a causal variable track. Attribution and mask optimisation methods win on circuits, and SAEs did not beat standard dimensions or DAS as featurisers for known causal variables. MIB keeps a private leaderboard.

## Tooling

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (created by Nanda, now at 4.0) for hooking open models; [nnsight and NDIF](https://arxiv.org/abs/2407.14561) (Fiotto-Kaufman, Loftus et al., ICLR 2025) for interventions, including remote runs on large models; [Neuronpedia](https://www.neuronpedia.org) for hosted features, SAEBench results and graphs; [circuit-tracer](https://www.anthropic.com/research/open-source-circuit-tracing) (Anthropic Fellows with Decode Research, May 2025) for attribution graphs on Gemma, Llama and Qwen; Transluce's [ADAG library](https://github.com/TransluceAI/circuits) for neuron-level circuits.

## What the labs say

**Anthropic.** Amodei's [The Urgency of Interpretability](https://darioamodei.com/post/the-urgency-of-interpretability) (April 2025) sets the goal that "interpretability can reliably detect most model problems" by 2027 and an "MRI for AI" within 5 to 10 years. The 2026 output shows the agenda shifting toward auditing and monitoring real Claude behaviour (evaluation awareness, hidden goals) with NLAs and the J-lens, alongside continued circuit work.

**Google DeepMind.** [A Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) (Nanda, Engels, Conmy et al., December 2025) says the team was "disappointed" by ambitious reverse engineering, now works from proxy tasks on real safety problems, and recommends prompting, steering, probing and chain-of-thought reading as the first baselines. Fundamental SAE work is deprioritised, though they still released Gemma Scope 2.

**OpenAI.** Leo Gao's [An Ambitious Vision for Interpretability](https://www.alignmentforum.org/posts/Hy6PX43HGgmfiTaKu/an-ambitious-vision-for-interpretability) (December 2025, personal view) argues full understanding remains tractable, measured by feature explainability and necessary-and-sufficient circuits, with weight sparsity as the path.

**Goodfire.** Parameter decomposition, probe-based products, and since October 2026 a public bet that neurons are already mostly clean. Goodfire absorbed much of Apollo's parameter decomposition line (Bushnaq, Braun, Sharkey).

**Transluce.** Open, AI-driven tools: neuron descriptions, neuron-basis circuits, ADAG, behaviour elicitation agents.

## Key unsolved problems

**Completeness.** No method accounts for all of a model's computation. SAEs miss true features and leave dark matter; attribution graphs carry error nodes and explain part of a prompt; Cammarata's estimate leaves 2 to 4% of Qwen neurons unexplained even on his generous threshold.

**Faithfulness.** Replacement models are not the model, NLAs confabulate, weight-sparse circuits may not be what the model computes, and random SAE baselines score like trained ones. Methods that are exact by construction are rare.

**Scaling.** Full CLTs and parameter decompositions exist only for small models; weight-sparse training stalls at tens of millions of nonzero parameters; per-prompt graphs need manual reading. Frontier results come from labs with private weights.

**Evaluation.** Whether a feature or label is "right" is still mostly judged by LLM scoring and steering, which SAEBench, MIB and the random-baseline results show can mislead.

**Unit of analysis.** The neuron versus feature question is open again, with Transluce and Goodfire arguing for neurons and GDM still treating superposition as real.

## Where our method fits

Our approach targets the two weakest points above. It is exact: each activation is split into a sum the model actually computes, so there is no reconstruction error, no dark matter by construction, and no replacement model to be unfaithful to. And it keeps the model's own neuron, which is the unit Transluce and Goodfire are now arguing for, while adding what they lack: a principled split of a polysemantic neuron's activations into modes, instead of either declaring the neuron clean or leaving it unsplit. A mode-level trace would be the exact analogue of an attribution graph without the CLT.

We are behind on almost everything else. Our exactness is about the decomposition, not about the meaning: mode boundaries come from clustering, and many hits land in no mode, which is our own completeness problem ([open_problems.md](open_problems.md)). We have no LLM MLP neuron result yet, no SAE comparison, no mode-level circuit, and no evaluation on SAEBench or MIB, all of which reviewers will expect. Our scale is a vision channel and a two-layer attention-only model, against labs working on Claude and Gemma 3 27B. We have no automated labelling, while the field now labels with agents. And our flagship example, mixed4e:55, has just been publicly reread as monosemantic (a V-shape detector), so we must show that its modes are genuinely distinct causes and not facets of one shape feature, or the example works against us.
