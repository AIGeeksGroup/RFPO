# Outer Loop Cycle 37: Surgical Adaptation of the Flow Actor

## Evidence Being Explained

The released Square actor has about 12.99 million non-visual MLP parameters, but its final linear
velocity head has only 114,800 parameters (about 0.88%). FPO++ updates every non-visual layer from a
small on-policy sparse-reward batch. Earlier audits show useful initial outcome gradients followed by
gradient rotation and endpoint drift, while lower learning rates and behavior anchors lose too much
surrogate progress. This is consistent with a parameter-subspace problem rather than only a step-size
problem: full fine-tuning may overwrite useful pretrained flow features while fitting scarce rewards.

## Diverged Candidates

1. Update only the final flow-velocity output head.
2. Linear-probe the first MLP layer that mixes image/state conditioning with noisy actions.
3. Update the last two MLP layers.
4. Add a low-rank adapter to every MLP layer.
5. Update biases only.
6. Freeze the actor and train a residual action head.
7. Alternate full and head-only FPO++ epochs.

## Convergence

Select H52, output-head-only FPO++. It is the smallest native parameter subspace that can directly
change every velocity component at every action time while keeping the pretrained observation,
time, and noisy-action representation fixed. Unlike H23/H39 it does not merely reduce update size;
unlike H28-H30 it adds no auxiliary behavior objective; unlike H48 it adds no new policy branch or
explicit ratio. Kumar et al. (2022) and Lee et al. (2022) provide independent evidence that full
fine-tuning can distort pretrained features and that selecting a small layer subset can improve
distribution-shift adaptation.

The other variants are deliberately excluded after observing H52. They add a layer-count, rank,
placement, or scheduling search that would turn a falsifiable subspace hypothesis into post-hoc
tuning. H52 first receives a paired held-out mechanism audit and reaches reward screening only if it
preserves useful FPO++ progress while reducing drift.
