# Prior work

[Index](index.md)

The work below is what the thesis builds on or argues against. Each entry says what the work showed and where we stand relative to it.

## Neurons as units: feature visualization and Circuits

**Simonyan, Vedaldi and Zisserman 2013**, [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034). Gradient saliency maps and class visualization by optimizing the input. The ancestor of both feature visualization and gradient attribution. We use neither directly, but the per-channel factor in our mixed4e:55 run is a gradient.

**Olah, Mordvintsev and Schubert 2017**, [Feature Visualization](https://distill.pub/2017/feature-visualization/). Optimizes inputs to excite a neuron, and shows that some neurons give mixed images because they respond to several unrelated things. It names the problem we address but treats the neuron as one object to visualize. Our modes split a neuron's real hits instead of synthesizing one image for the whole neuron.

**Cammarata, Carter, Goh, Olah et al. 2020**, [Thread: Circuits](https://distill.pub/2020/circuits/), opened by [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) (Olah, Cammarata, Schubert, Goh et al. 2020) and [An Overview of Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/) (Olah et al. 2020). The neuron view: neurons are meaningful units and their weights to earlier neurons form readable circuits. Zoom In names polysemantic neurons, using a cat and car neuron in InceptionV1 as an example, and leaves them as an obstacle. Our thesis is a direct continuation: we keep the neuron as the unit and give polysemantic neurons a circuit account through modes. Our dependency step (input dims covering 90% of a mode's Hadamard mass) is the per-mode version of their weight reading.

**Cammarata, Goh, Carter, Schubert et al. 2020**, [Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/), and **Cammarata, Goh, Carter, Voss et al. 2021**, [Curve Circuits](https://distill.pub/2020/circuits/curve-circuits/). The clearest case that a single neuron family can be understood completely and reimplemented by hand. These are monosemantic neurons. They are the standard we want a (neuron, mode) pair to meet: one mode should be as clean as a curve detector.

**Goh, Cammarata, Voss, Carter et al. 2021**, [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/). CLIP neurons that respond to a concept across photos, drawings and text. A multimodal neuron is a natural test for our method: if it fires for the photo and the word through different upstream routes, it should split into modes that still share one concept. It is not yet run.

**Voss, Goh, Cammarata, Petrov et al. 2021**, [Branch Specialization](https://distill.pub/2020/circuits/branch-specialization/). Features of a kind cluster into the same architectural branch. Supports the premise that different kinds of evidence travel by different routes, which is the worlds hypothesis at the scale of branches rather than single input dims.

Nick Cammarata is an author on all of the Circuits papers above and on Bills et al. 2023 below. We found no separate published work of his on polysemanticity beyond these.

**Bau, Zhou, Khosla, Oliva and Torralba 2017**, [Network Dissection: Quantifying Interpretability of Deep Visual Representations](https://arxiv.org/abs/1704.05796). Scores each unit by overlap with labelled concepts and reports that many units align with one concept. It labels whole units, so a polysemantic unit gets its best single label or none. Our modes would let dissection score each mode separately.

**Mu and Andreas 2020**, [Compositional Explanations of Neurons](https://arxiv.org/abs/2006.14032). Explains neurons as logical formulas over concepts (for example "A or B"), which describes polysemantic neurons as disjunctions. Closest in spirit to us: an "or" is a set of modes. The difference is that their disjuncts come from concept labels, ours from the exact input decomposition, and ours point to the upstream neurons behind each disjunct.

## Features instead of neurons: superposition and dictionary learning

**Elhage, Hume, Olsson et al. 2022**, [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) (also on [transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html)). Shows networks store more sparse features than dimensions, so neurons become polysemantic. We accept the observation and dispute the conclusion that neurons are therefore the wrong unit. Superposition says the neuron's output mixes features; it does not say the routes into the neuron are mixed on each input. If features are sparse, any single activation is driven by few of them, which is what makes per-activation splitting plausible.

**Bricken, Templeton, Batson, Chen, Jermyn et al. 2023**, [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Trains a sparse autoencoder on a one-layer transformer's MLP and finds features far more interpretable than neurons. This is the main alternative to our claim. Differences: the dictionary is learned and approximate (reconstruction error), its features are directions not tied to any neuron, and it is trained per layer. Ours is exact and keeps the neuron. A direct comparison on the same layer is listed as needed in [index.md](index.md).

**Cunningham, Ewart, Riggs, Huben and Sharkey 2023**, [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600). Independent SAE result on Pythia, with features that localize causal behaviour better than neurons. Same relation to us as Bricken et al.

**Templeton, Conerly, Marcus, Lindsey, Bricken et al. 2024**, [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). SAEs at production scale, with abstract, multilingual and multimodal features and steering. Sets the bar for scale that our method has not reached (see [open_problems.md](open_problems.md)).

**Lindsey, Templeton, Marcus, Conerly, Batson and Olah 2024**, [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html). Dictionaries that read and write across layers, partly to make circuits traceable through features. It addresses the same gap we do (features that do not connect layer to layer) by learning more, where we use the model's own weights.

**Lieberum, Rajamanoharan, Conmy et al. 2024**, [Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2](https://arxiv.org/abs/2408.05147). Open SAEs on every layer of Gemma 2, with Neel Nanda as senior author. The practical baseline for our SAE comparison, though our Gemma runs use Gemma 3, so a matching dictionary has to be found or trained.

**Smith, Rajamanoharan, Conmy, McDougall, Kramar, Lieberum, Shah and Nanda 2025**, [Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2)](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks). This is the Nanda SAE skepticism source. SAE probes underperformed linear probes on out-of-distribution harmful intent detection, and the team deprioritised fundamental SAE research. It is outside the 2013 to 2024 window but is the citation [claim.md](claim.md) asks for. It supports our motivation (a learned dictionary is not obviously the right unit) without endorsing neurons.

## Transformer circuits and Neel Nanda's work

**Elhage, Nanda, Olsson et al. 2021**, [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Writes attention as QK and OV circuits and reads them through the low-rank matrix W_Q^T W_K. Our bilinear unit decomposition takes this directly: the score is split in the singular basis of that matrix, because head dimensions are not a privileged basis. We add the per-hit clustering on top.

**Olsson, Elhage, Nanda et al. 2022**, [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895). Identifies previous-token and induction heads and links them to in-context learning. Our attention experiments ([attention.md](attention.md)) use these heads in attn-only-2l as known ground truth for whether modes recover the right mechanism.

**Nanda, Chan, Lieberum, Smith and Steinhardt 2023**, [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217). Fully reverse-engineers modular addition into Fourier features. Shows exact, weight-level understanding is possible in small models, which is the standard of exactness we aim for, though their result came from careful manual analysis rather than a general procedure.

**Nanda and Bloom 2022**, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). The standard library for hooking transformer activations. Relevant as tooling for the attention and Gemma runs.

**Nanda 2023**, [Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching). Approximates activation patching with a first-order gradient term, (patched minus clean activation) times gradient. This is gradient times input with a baseline, the same algebraic object as our hit vector. The difference is use: they aggregate it to rank components for one task, we keep it per hit and cluster it. **Syed, Rager and Conmy 2023**, [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348), makes it a circuit discovery method.

**Gurnee, Nanda, Pauly, Harvey, Troitskii and Bertsimas 2023**, [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610). Finds both monosemantic neurons and polysemantic ones that represent features in superposition, especially in early layers. Supports our starting observation and shows neurons are not uniformly useless, against the strong form of the features view.

**Gurnee, Horsley, Guo et al. 2024**, [Universal Neurons in GPT2 Language Models](https://arxiv.org/abs/2401.12181). Neurons that recur across random seeds tend to be interpretable. Further evidence that neurons are real units of the model, which our approach relies on.

## Causal localization of circuits

**Meng, Bau, Andonian and Belinkov 2022**, [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (ROME). Causal tracing: corrupt the input, restore one activation, measure recovery. Localizes at the level of layers and positions. Our method localizes within a neuron and is purely observational, so causal claims about a mode still need ablation, as done for the attention modes.

**Wang, Variengien, Conmy, Shlegeris and Steinhardt 2022**, [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593). A full head-level circuit found by path patching. Our mode-level circuit trace, still to be done, would be a neuron-level analogue that needs no task-specific corrupted dataset.

**Goldowsky-Dill, MacLeod, Sato and Arora 2023**, [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969). Formalizes path patching as testing hypotheses about which paths carry an effect. Our Hadamard split is a single-edge path decomposition that is exact for a linear layer, so it gives path contributions without patching, but only one layer at a time.

**Conmy, Mavor-Parker, Lynch, Heimersheim and Garriga-Alonso 2023**, [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997) (ACDC). Automates circuit search by pruning edges whose patching does not hurt a task metric. Task-driven and edge-level; ours is neuron-driven and finds what a neuron does across all its hits rather than for one task.

## Attribution: where the Hadamard product comes from

**Shrikumar, Greenside, Shcherbina and Kundaje 2016**, [Not Just a Black Box: Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1605.01713). Introduces gradient times input as an attribution. For a linear unit, gradient times input is exactly our hit vector: the activation of each input dim times its weight. So the decomposition itself is not new. What is new is treating the attribution vector of each hit as a point and clustering a neuron's hits by it.

**Shrikumar, Greenside and Kundaje 2017**, [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685) (DeepLift). Attributions relative to a reference that satisfy summation to delta. We use DeepLift cards to display what a mode responds to in the input image, not to define modes.

**Sundararajan, Taly and Yan 2017**, [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (Integrated Gradients). Completeness axiom: attributions sum to output minus baseline. Our one-layer Hadamard split satisfies completeness trivially and exactly, which is why we call it exact. Over several nonlinear layers it would not, and that is the boundary of the exactness claim.

**Ancona, Ceolini, Oztireli and Gross 2017**, [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104). Shows epsilon-LRP and DeepLift (Rescale) reduce to gradient times input in ReLU networks under some conditions. Useful for arguing that our choice of attribution is not arbitrary.

## Automated explanation

**Bills, Cammarata, Mossing, Tillman, Gao, Goh, Sutskever, Leike, Wu and Saunders 2023**, [Language models can explain neurons in language models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html). GPT-4 writes and scores explanations of every GPT-2 neuron, and scores are low, partly attributed to polysemanticity. Our modes offer a fix it could use: explain each mode separately, so each explanation has one meaning to capture.

## Clustering

**Rodriguez and Laio 2014**, [Clustering by fast search and find of density peaks](https://www.science.org/doi/10.1126/science.1242072). Science 344(6191). Cluster centres have high density and large distance to any denser point; every other point joins its nearest denser neighbour. Our density-peak forest is this algorithm with kNN density, elbow-chosen roots and soft membership added. We inherit its known weakness, possible over-splitting of one hill with two bumps.

## Where this leaves our claim

The Circuits thread gives us the unit and the goal, and leaves polysemantic neurons unexplained. Superposition explains why polysemantic neurons exist, and the dictionary-learning line answers with learned, approximate features detached from neurons. Our claim sits between: it accepts superposition but argues that the mixture is resolvable per activation, using an attribution (gradient times input) that is decades old in spirit and exact for one linear layer. The decomposition is not the contribution; clustering a neuron's per-hit attributions into modes, and tracing circuits through them, is. Mu and Andreas and the GPT-4 explanation work are the nearest neighbours in describing a neuron as several meanings; attribution patching is the nearest neighbour in computation. To be credible the thesis needs what the SAE papers have and we lack: a head-to-head comparison on the same layer, an MLP neuron in a real language model, and a causal mode-level circuit, ideally checked against a known circuit such as IOI or induction heads.
