# Hariom's method on Gemma 3: the dimensions of W_E

Implementation notes for [the thesis](../thesis/index.md).

This doc continues [in_tfm_neuron_clustering.md](in_tfm_neuron_clustering.md). That doc explains Hariom's method on the vision neuron mixed4e:55: pick a neuron, take its strongest hits, describe each hit, cluster with the density-peak forest, and read the clusters as the separate worlds the neuron fires through. Read it first; hits, elbows, density, the forest, probabilities and the map are all defined there and not repeated here. [glossary.md](../thesis/glossary.md) is a short lookup table.

This doc carries the same method into a language model, following Hariom's proposal: treat the dimensions of Gemma 3's embedding matrix as the units. It runs in two steps. First, cluster the dimensions themselves, to check that they are real units and to find the interesting ones. Then take single dimensions and run the full mixed4e:55 pipeline on them.

Earlier exploratory runs that cluster tokens rather than dimensions are in [gemma3_token_rows.md](gemma3_token_rows.md). They are a side branch and nothing here depends on them.

## Why a language model

mixed4e:55 shows that a vision neuron can fire through separate worlds, cats and cars, each through its own upstream route. The question is whether that is a quirk of vision models or a general property of neural networks. A language model is the natural second test.

## Language-model background

**Tokens.** A language model reads text cut into **tokens**, pieces from a fixed vocabulary: common words (`␣the`), word parts (`ize`), single characters, punctuation, and fragments of every script the model was trained on. `␣` marks a leading space, meaning the token starts a word. Gemma's vocabulary has 262,144 tokens.

**Embedding matrix W_E.** The model's first step looks each token up in a table. Row i of the table is a list of numbers, the token's **embedding**, and everything the model computes starts from it. This table is the **embedding matrix** `W_E`. For `google/gemma-3-270m`, a small 270-million-parameter Gemma 3, it has 262,144 rows and 640 columns.

**Tied unembedding.** At the end, the model turns its internal vector back into a score per token with the **unembedding matrix** `W_U`. Gemma **ties** them: `W_U` is the same matrix as `W_E`, so `W_E` is shaped both by how tokens are read and by how they are predicted.

**Rows and columns.** A **row** of `W_E` is one token's 640 numbers. A **column**, also called a **dimension**, is one of the 640 axes: one number per token. This doc is about columns.

**Kept tokens.** `<unused0>` to `<unusedN>` are reserved placeholders that never appear in training, and **special tokens** such as `<bos>` (beginning of sequence) control the model. Both are dropped, leaving **255,893 kept tokens**.

**Script.** The writing system a token is written in: Latin, CJK (Chinese, Japanese, Korean characters), Devanagari, Bengali and so on. Many of the results below are about script.

**Byte token.** A token of the form `<0xNN>`, standing for one raw byte. Used for text the vocabulary has no piece for.

## Mapping the method onto Gemma

Each part of Hariom's pipeline needs a Gemma counterpart.

| In mixed4e:55 | In Gemma |
|---|---|
| Neuron: channel 55 of `mixed4e_1x1` | One column (dimension) of `W_E` |
| Activation: the channel's value at an image position | The column's value for a token |
| Hits: positions past the elbow of sorted activations | Tokens past the elbow of the column's sorted values |
| Hit vector: activation times attribution over 528 upstream channels | The token's full 640-number embedding row |
| Distance: signed weighted Jaccard | Cosine distance |

Two of these need explaining.

**Why a dimension stands in for a neuron.** Gemma's real neurons are inside its MLP layers and need the model run on text. A dimension of `W_E` is the simplest unit-like thing available: a fixed axis every token has a value on. Whether these axes behave like units at all is exactly what step 1 checks.

**Why cosine and not weighted Jaccard.** In the vision run a hit vector says how much each upstream channel drove the hit. An embedding has nothing upstream of it, so there is no attribution to compute, and the hit vector is just the token's embedding row. Embedding rows are dense (every entry non-zero) and signed, so cosine distance is used:

1. **Centre**: subtract the mean row over the vocabulary. Every embedding shares a common offset, and removing it makes distances measure how a token differs from the typical token.
2. **L2-normalise**: divide each row by its length, so only direction matters.
3. **Compare**: Euclidean distance between unit vectors is `sqrt(2 - 2 cos θ)`, which orders pairs exactly like **cosine distance** (one minus the cosine of the angle). In code this is `distance="euclidean"` on the normalised rows.

A weighted-Jaccard rerun, for a like-for-like comparison with the vision run, has not been done.

Clustering is in-tfm's density-peak forest with elbow root cuts, and maps are in-tfm's bucket-graph layout, both exactly as in the method doc.

## Step 1: are the dimensions real units?

**Question.** A neural network's axes can be arbitrary: rotate the whole space and the model computes the same thing, just in a different basis. If Gemma's 640 axes are arbitrary, treating one as a neuron is meaningless. So first: do the axes carry structure that a random basis would not, and do they group into families?

**Setup.** Notebook `notebooks/gemma3_embedding_dims.ipynb`, command `dims`, report `outputs/reports/gemma3_embedding_dims_graph/`, shared at https://pub-1fcc7a923941489c9ded86b30ffa58ba.r2.dev/dims/index.html.

Each of the 640 dimensions is a point, described by its 255,893 values across the vocabulary. The values are centred and normalised, so cosine distance between two dimensions is **correlation distance**: they are close if their values rise and fall together across tokens. The forest uses k = 5.

Two measures and a control:

- **Correlation between dimensions.** If two axes are strongly correlated they partly encode the same thing.
- **Excess kurtosis.** How heavy a distribution's tails are compared with a Gaussian, where a Gaussian scores 0. A dimension with high kurtosis is near zero for most tokens and large for a few. That is how a feature detector behaves: off for most inputs, on for a specific few.
- **Rotated control.** The same pipeline on `W_E` multiplied by a random orthogonal (rotation) matrix. Rotation keeps every distance between tokens and changes only the axes. Anything the real axes show that the rotated ones do not belongs to the basis itself.

**Result.**

| | Raw | Rotated |
|---|---|---|
| Trees | 28 | 2 |
| Max correlation between dims | 0.74 | 0.43 |
| Median / max excess kurtosis | 0.21 / 2.1 | 0.05 / 0.2 |

302 of the 640 dimensions are more heavy-tailed than any rotated dimension. The axes are not arbitrary: the model uses its basis, and individual dimensions behave like feature detectors. That justifies step 2.

The families are weak. The 28 trees are loose (mean correlation inside a tree 0.03 to 0.10), and the tokens that score highest on them are multilingual fragments. With only 640 points the elbow root cut is fragile, so the tree count should not be trusted closely.

Single dimensions are much clearer, and they encode script and form: dim 421 separates Vietnamese capitals from CJK nouns, 163 separates byte tokens from Bengali, 239 marks digits, 218 marks code identifiers. Dims 421 and 218 are taken forward to step 2.

`outputs/reports/gemma3_embedding_dims/` predates a bug fix (the forest sorts its points internally and the report read them in input order) and is wrong. Use the `_graph` report.

## Step 2: one dimension as a neuron

**Question.** Take one dimension and run mixed4e:55's pipeline on it. Does it fire through several separate worlds?

**Setup.** Command `single-dim`, reports `outputs/reports/gemma3_dim{421,218}_{pos,neg}/`, run locally.

Hits are picked as for mixed4e:55, with one adjustment. A real neuron passes through a ReLU, so only its positive side matters, and its strong activations sit in one tail. An embedding dimension has no ReLU. Its values form a bell curve centred near zero, and an elbow taken over all tokens lands in the opposite tail. So the elbow is taken over one side of zero only, as a ReLU would leave it.

Both sides are run: `pos` (the highest values) and `neg` (the lowest). Flipping the sign of a column, and of every weight that reads it, gives an identical model, so neither side is privileged.

Each hit is its token's full 640-number embedding row, centred on the vocabulary mean and normalised, clustered with k = 10. Each tree is one candidate world.

**Result.**

| | Hits | Trees |
|---|---|---|
| dim 421 high | 5,629 | 298 |
| dim 421 low | 16,475 | 614 |
| dim 218 high | 8,822 | 589 |
| dim 218 low | 10,142 | 937 |

One dimension does fire through many unrelated worlds. Dim 421 low holds Indonesian verbs, Japanese particles, Portuguese nouns, rare CJK characters, and medical professions (physicians, psychiatrist, cardiologist). Most worlds are a language or a script. Next come word parts: -ize, -ist, Russian adjective endings, Slavic -ský. A minority are semantic, and some of those cross languages: "known" in several languages, Romance subjunctives.

Compared with mixed4e:55 (about 41 clusters for 14,240 hits), these hits break into hundreds of small islands. Part of the reason is **superposition**: with 640 dimensions shared by 262,144 tokens, each dimension carries parts of many unrelated features. Some of the fragmentation is therefore superposition rather than separate mechanisms. An embedding dimension is a weak stand-in for a neuron.

Each report has the elbow plot and a per-cluster view. Shared at:

- https://pub-1fcc7a923941489c9ded86b30ffa58ba.r2.dev/dim421_pos/index.html
- https://pub-1fcc7a923941489c9ded86b30ffa58ba.r2.dev/dim421_neg/index.html
- https://pub-1fcc7a923941489c9ded86b30ffa58ba.r2.dev/dim218_pos/index.html
- https://pub-1fcc7a923941489c9ded86b30ffa58ba.r2.dev/dim218_neg/index.html

## What it adds up to

Gemma's embedding axes are real units: heavy-tailed, unlike a rotated basis, and each one detects something specific, mostly script and form. Run as neurons through Hariom's pipeline, single dimensions fire through many separate worlds, organised mainly by language and script, so the worlds picture carries over to a language model. Two things stop this from being a fair test of the worlds hypothesis. A dimension has no nonlinearity and no upstream route to attribute through. And superposition makes its worlds far more fragmented than a neuron's.

## Next

The fair transformer equivalent of neuron 55 is an MLP neuron in Gemma layer 0 or 1: a real unit with a nonlinearity and upstream inputs. Its hits would come from real WikiText activations (WikiText-103 is a corpus of Wikipedia articles), its hit vectors from input times gradient attribution, and distances from weighted Jaccard. That is Hariom's pipeline with nothing substituted.

## Running

Local runs use this repo's env (`uv sync`), which installs in-tfm from git at `feat/bucket-report` and the `nint` command. `uv run nint --help` lists the commands. The code is in `src/neural_interp`: shared code in `lib/`, one module per experiment in `experiments/`, HTML templates in `templates/`. The Gemma licence must be accepted on Hugging Face, and the token goes in `HF_TOKEN` (kept in `.env`) or, on Colab, in `/content/.hf_token`.

Step 1, writing into `out/`:

```
uv run nint embed out
uv run nint dims out
uv run nint render dims out/dims.json out/dims_report.html
```

`embed` saves the kept tokens' embedding rows, their ids and display strings, and token features. `dims` clusters the columns and the rotated control and writes `dims.json`.

Step 2, for one dimension and side:

```
set -a; . ./.env; set +a
uv run nint single-dim 421 neg outputs/reports/gemma3_dim421_neg
uv run nint render single outputs/reports/gemma3_dim421_neg/single_dim.json outputs/reports/gemma3_dim421_neg/report.html
```

Step 2 runs fine on a laptop. Step 1 was run on Colab: create a session with `colab new -s interp --gpu T4`, write the token to `/content/.hf_token`, and upload `src/neural_interp` to `/content/gw/neural_interp`. Then `bash scripts/run_detached.sh <notebook> <cell>` runs a notebook cell detached on Colab and streams its log into the local notebook.
