# Outer Loop Cycle 35: Value Information Before Density Transport

## Evidence Being Explained

Square has reproducible non-saturated headroom, but inference geometry, source priors, FPO++
estimators, optimizer changes, endpoint anchors, residual steering, and complete-horizon collection
have not produced a stable reward gain. H19 additionally showed that a better-calibrated scalar
critic can still rank returns worse. H48-H49 showed that allowing stronger policy modulation is
harmful when the value signal is not independently validated.

## Diverged Candidates

1. Full RLDT actor, replay, trainable visual double-Q, and SVGD implementation.
2. Frozen-policy RLDT action transport at inference time.
3. MPPI selection over Gaussian flow sources using a learned chunk critic.
4. Double-Q best-of-eight source selection without actor updates.
5. Asymmetric full-state value critic with the unchanged sparse reward.
6. Potential-based Square reward shaping from nut and peg geometry.
7. A trainable critic-only visual encoder with online image augmentation.
8. A distributional terminal-success critic rather than scalar MSE value regression.
9. Auxiliary object-pose prediction for the existing frozen visual conditioning.
10. Stage-aware reach, grasp, lift, and hover value heads using simulator labels.
11. Flow-timestep reweighting based on held-out gradient alignment.
12. Layer-restricted flow finetuning after critic warmup.

## Convergence

- Full RLDT is parked because no official code is public, its Robomimic recipe costs about 30 A40
  GPU hours, and it changes the base policy, action horizon, critic, replay, and optimizer together.
- Frozen Q guidance, MPPI, and best-of-K are rejected before audit because they inherit the same
  unvalidated critic and overlap the closed latent-steering and candidate-selection routes.
- Reward shaping and stage labels are cheap but change the sparse-reward benchmark contract.
- A trainable visual critic and object-pose auxiliary loss add representation optimization without a
  direct low-cost test that the extra pixels become reward-relevant.
- Distributional targets repeat H19's target-side intervention without fixing missing state
  information.
- Flow-timestep and layer-restricted updates act downstream of the unresolved critic bottleneck.

## Selected Direction

Select H50, an asymmetric simulator-state critic audit. This is problem-first: the official critic's
return ranking is unreliable under sparse visual Square rollouts. It is the simplest candidate that
adds reward-relevant information without changing policy inputs, source exploration, flow loss, or
reward. A positive cross-episode audit authorizes integration into short FPO++ training; a negative
result closes privileged value information and keeps full RLDT parked until code is released.

