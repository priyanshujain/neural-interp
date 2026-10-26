# Gemma 3 embedding dimensions as units

[Index](index.md)

## Why

mixed4e:55 shows a vision neuron firing through separate worlds. The question is whether that is a quirk of vision or a general property. Following Hariom's proposal, the first language-model step treats the dimensions of Gemma 3's embedding matrix as units, before moving to real MLP neurons.

## Background

A language model reads text cut into tokens from a fixed vocabulary: whole words (`▁the`), word parts (`ize`), characters, punctuation and fragments of every script it was trained on. `▁` marks a token that starts a word. Its first step looks each token up in the embedding matrix `W_E`. For `google/gemma-3-270m`, `W_E` has 262,144 rows (tokens) and 640 columns (dimensions). Gemma ties the unembedding to it, so `W_E` is shaped both by how tokens are read and by how they are predicted. Reserved `<unusedN>` placeholders and special tokens such as `<bos>` are dropped, leaving 255,893 kept tokens. Byte tokens `<0xNN>` stand for one raw byte.

## Mapping onto the method

| mixed4e:55 | Gemma |
|---|---|
| Neuron: channel 55 | One column of `W_E` |
| Activation | The column's value for a token |
| Hits: elbow of sorted activations | Tokens past the elbow, one side of zero |
| Hit vector: Hadamard over 528 channels | The token's 640-number embedding row |
| Distance: signed weighted Jaccard | Cosine distance |

A dimension is the simplest unit-like thing available without running the model: a fixed axis every token has a value on. An embedding has nothing upstream, so there is no exact decomposition, and the hit vector is the embedding row itself. Rows are dense and signed, so they are centred on the vocabulary mean, L2-normalised and compared by cosine distance. A weighted-Jaccard rerun has not been done.

## Step 1: are the dimensions real units?

**Question.** A network's axes can be arbitrary: rotate the whole space and the model computes the same thing in a different basis. If Gemma's 640 axes are arbitrary, treating one as a neuron is meaningless.

**Setup.** Each of the 640 dimensions is a point described by its 255,893 values. Values are centred and normalised, so cosine distance between dimensions is correlation distance. Forest with k = 5. Measures: correlation between dimensions, and excess kurtosis (heavy tails mean near zero for most tokens and large for a few, which is how a feature detector behaves). Control: the same pipeline on `W_E` times a random orthogonal matrix, which keeps every token distance and changes only the axes.

**Result.**

| | Raw | Rotated |
|---|---|---|
| Trees | 28 | 2 |
| Max correlation between dims | 0.74 | 0.43 |
| Median / max excess kurtosis | 0.21 / 2.1 | 0.05 / 0.2 |

302 of the 640 dimensions are more heavy-tailed than any rotated dimension. The axes are not arbitrary: the model uses its basis, and single dimensions behave like feature detectors.

Families are weak. The 28 trees are loose (mean within-tree correlation 0.03 to 0.10), and their top tokens are multilingual fragments. With only 640 points the elbow root cut is fragile, so the tree count should not be relied on.

Single dimensions are clear and mostly encode script and form: dim 421 separates Vietnamese capitals from CJK nouns, 163 separates byte tokens from Bengali, 239 marks digits, 218 marks code identifiers. Dims 421 and 218 go to step 2.

## Step 2: one dimension as a neuron

**Setup.** Hits are the tokens past the elbow on one side of zero; both sides are run (`high`, `low`). Each hit is its centred, normalised embedding row, clustered with k = 10.

**Result.**

| | Hits | Trees |
|---|---|---|
| dim 421 high | 5,629 | 298 |
| dim 421 low | 16,475 | 614 |
| dim 218 high | 8,822 | 589 |
| dim 218 low | 10,142 | 937 |

One dimension fires through many unrelated worlds. Dim 421 low holds Indonesian verbs, Japanese particles, Portuguese nouns, rare CJK characters, and medical professions (physicians, psychiatrist, cardiologist). Most worlds are a language or a script. Next come word parts: -ize, -ist, Russian adjective endings, Slavic -ský. A minority are semantic, some across languages: "known" in several languages, Romance subjunctives.

Compared with mixed4e:55 (about 41 modes from 14,240 hits), these hits break into hundreds of small islands. With 640 dimensions shared by 262,144 tokens, each dimension carries parts of many unrelated features, so some fragmentation is superposition rather than separate mechanisms.

## What it adds up to

Gemma's embedding axes are real units: heavy-tailed, unlike a rotated basis, each detecting something specific, mostly script and form. Run as neurons, single dimensions fire through many separate worlds organised by language and script, so the worlds picture carries over to a language model.

It is not a fair test of the thesis. A dimension has no nonlinearity, no upstream route and no exact decomposition, and superposition fragments its worlds far more than a neuron's. The fair test is an MLP neuron ([open_problems.md](open_problems.md), Scale).

## Known bugs

An earlier dims report read the forest's internally sorted points in input order, so its tree contents were wrong. All numbers above are from the corrected run.

## Artifacts

Implementation notes move to `docs/implementation/` when this experiment is rerun on the current library.
