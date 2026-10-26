# Towards publication

[Index](index.md)

## What would make the claim publishable

**A testable definition of monosemantic.** Candidate: one mode, one consistent cause in the inputs, and one consistent downstream effect.

**A head-to-head comparison with an SAE on the same layer.** Do modes match SAE features? Do modes explain behaviour that SAE features miss, or the reverse?

**A circuit traced end to end through modes.** For example induction: L0H7 mode, then L1 key mode, then output. The trace has to be validated by ablating modes, not whole neurons.

**Coverage numbers.** The fraction of hits assigned to a mode, and the fraction of the neuron's downstream effect those hits carry.

## Questions to settle

- Is "monosemantic" defined by the input side, the output side or both?
- Is a mode like "after a period, copy the wrong token" a meaning, or a failure the model compensates for?
- What is the claim for hits outside every mode?
- Which clustering result do we report: HDBSCAN, the density-peak forest, or both with the reason for the switch?
