# Attention heads in attn-only-2l

[Index](index.md)

Attention scores are bilinear units, decomposed exactly in the singular basis of `W_Q^T W_K` and into token and position parts ([method.md](method.md), step 9). attn-only-2l is a two-layer attention-only transformer, small enough that its induction circuit is known, which makes it a check that modes recover known structure.

## Previous-token head: L0H7

- The exact split shows 99% of the score margin comes from position·position.
- Position embeddings alone make i-1 the top key at all 126 query positions.
- In the `W_Q^T W_K` singular basis, about 8 directions carry 96% of the margin.

So the head's job is fully accounted for by a small, exact, position-only mechanism, which is the expected previous-token behaviour recovered without any learned dictionary.

## Induction heads: L1H10 and L1H4

Modes are monosemantic, and readable from the current token and from what is copied.

- **L1H10** has an "after a period" mode and an "after a comma" mode. Both copy the wrong token 88 to 94% of the time.
- **L1H4** has an "inside `@-@`" mode that copies '@' correctly every time.

Usage differs sharply by mode.

- Ablating L1H4's `@-@` mode raises loss by a median 5.6.
- L1H10's period mode is barely used, with a reader share of 0.05.

Mode identity predicts downstream use. That is what circuit tracing needs: a mode is a unit whose effect can be measured on its own, not averaged over the whole head.

## Sparse hits on these heads

Many hits fall outside every mode. On the induction heads these sparse hits are real name induction, used a little rather than not at all. This is the main evidence against dropping sparse hits as noise ([open_problems.md](open_problems.md)).

## Caveats to record

- These results were clustered with HDBSCAN, before the switch to the density-peak forest. They have not been rerun with the forest.
- Whether "after a period, copy the wrong token" is a meaning or a failure the model compensates for is open ([publication.md](publication.md)).
- The end-to-end trace L0H7 mode, then L1 key mode, then output, has not yet been done.
