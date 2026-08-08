# Protocol: Reflow CFM Gradient-Disagreement Audit

## Hypothesis

Conditional reflow's replicated 22% straightness improvement reduces disagreement among the Monte
Carlo CFM gradients used by FPO++, providing a mechanism by which an online reflow auxiliary could
stabilize policy updates even though offline reflow alone did not improve official-scale reward.

## Paired Audit

Use both existing 100-update matched control/reflow checkpoint pairs, trained with seeds 20260808
and 20260809. For each checkpoint, draw the same two batches of 16 Can dataset observation/action
pairs and the same eight fixed `(t, epsilon)` CFM samples. For every CFM sample, compute the gradient
of its batch-mean loss with respect to all non-vision actor parameters.

Report the cosine similarity of each sample gradient to the eight-sample mean gradient, gradient-norm
coefficient of variation, and CFM-loss coefficient of variation. Inputs and Monte Carlo samples must
be paired exactly between each seed's control and reflow checkpoint.

## Gate

Reflow must improve mean gradient-to-average cosine by at least 0.05 and reduce gradient-norm CV by
at least 10% relative in both training-seed pairs. All gradients must be finite and nonzero. If either
seed fails either threshold, refute H2 and do not implement or train an online reflow auxiliary.

This is a mechanism gate, not a reward result. Do not change the batch size, MC count, parameter
subset, or threshold after observing results.
