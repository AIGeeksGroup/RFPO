# Outer Loop 3: Distinct Training Signals

## Diagnosed Bottleneck

Fresh positive-GAE FPO++ gradients contain useful outcome information, but repeated optimizer epochs
quickly rotate the gradient and clip active positive ratios. Inference, source, CFM-estimator,
advantage-transform, learning-rate, batching, early-stop, replay, and post-hoc interpolation routes
have failed. A viable candidate must add information about which policy changes should be protected.

## Raw Candidates

1. PCGrad between FPO++ and a frozen BC velocity-field anchor.
2. Scalar BC velocity distillation added to FPO++.
3. Adaptive Lagrangian constraint on BC velocity error.
4. L2-SP distance to initial actor parameters.
5. Fisher-weighted initial-parameter anchoring.
6. EMA endpoint evaluation during online fine-tuning.
7. Positive-advantage-only FPO++ updates.
8. Per-layer gradient conflict masking with a BC anchor.
9. Orthogonal gradient descent relative to BC Jacobian directions.
10. Trust-region constraint on deterministic action-chunk displacement.
11. Trust-region constraint on full-noise action distribution samples.
12. Conservative Q-filtered behavior cloning on unsuccessful states.
13. Advantage-gated teacher distillation.
14. Unfreeze the vision encoder with a much smaller encoder learning rate.
15. Potential-based dense reward shaping from object-goal geometry.
16. A learned success classifier used only to select conservative updates.

## Shortlist

| Candidate | Real bottleneck | Distinct | Clear mechanism | Low tuning | Short audit |
|---|---:|---:|---:|---:|---:|
| BC-anchor PCGrad | high | high | high | high | high |
| Adaptive BC constraint | high | high | high | medium | high |
| Deterministic action trust region | medium | high | medium | medium | medium |
| Positive-advantage-only FPO++ | medium | high | medium | high | high |
| EMA endpoint evaluation | medium | medium | low | high | high |

## Selection

Select BC-anchor PCGrad. It intervenes exactly when the RL update conflicts with retention of the
initial BC velocity field, has no auxiliary-loss coefficient, and admits a fixed-rollout audit before
online reward training. Scalar distillation and adaptive constraints remain fallbacks only if the
projection mechanism passes geometrically but fails because it removes nearly all RL progress.

Do not pursue reward shaping because changing the benchmark reward weakens comparison fairness.
Do not pursue encoder unfreezing before a cheaper actor-only mechanism has evidence. Do not pursue
EMA evaluation because H25 showed that recovering base-like weights cannot create a reward gain when
the trained endpoint lacks one.

