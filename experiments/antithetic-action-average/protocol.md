# Protocol: Antithetic Action-Chunk Averaging

## Hypothesis

For a fixed observation, average the two action chunks obtained by integrating the released flow
policy from Gaussian sources `z` and `-z`. Symmetry cancels the endpoint component that is odd in
the source while retaining even nonlinear corrections. This inference-only estimator may therefore
reduce endpoint variance and improve Can success relative to the official zero-source action.

This differs from H14, which paired time/noise samples inside the training-time CFM gradient
estimator. H26 changes only checkpoint evaluation and does not alter training, weights, or the
FPO++ objective.

## Locked Intervention

- Draw exactly one `z ~ N(0, I)` action-chunk source at every policy replan.
- Run the unmodified 10-step Euler solver once from `z` and once from `-z`.
- Execute `0.5 * (action(z) + action(-z))`.
- Use scale 1, one antithetic pair, and arithmetic averaging. These choices will not be tuned after
  observing the screen.
- Reject non-finite candidate actions as a mechanism failure.
- Record rollout FPS because the candidate approximately doubles policy inference compute.

## Fixed-Seed Screen

- Checkpoint: official released Can step-1000 policy, EMA weights as in the reproduced baseline
- Environment: Can, 10 Euler sampling steps
- Seed: 20260824
- Conditions: official zero-source inference and antithetic action-chunk averaging
- Budget: 20 episodes per condition, 16 environments, OSMesa rendering
- Video and W&B logging: disabled
- Evaluation order: zero control, then antithetic candidate

H26 passes only if:

1. the candidate produces finite actions throughout evaluation; and
2. candidate success exceeds control by at least 2/20 episodes, or 10 percentage points.

If either gate fails, stop without changing the source scale, number of pairs, averaging rule,
sampling steps, checkpoint, or seed. If both gates pass, run an independent 50-episode confirmation
at seed 20260825 with the same two conditions. A confirmation requires at least a 5-point candidate
gain and no invalid actions before a 200-episode benchmark evaluation is considered.

OSMesa is acceptable for this screen and confirmation. A formal benchmark-improvement claim still
requires healthy EGL or an independently approved rendering GPU.

## Provenance

- Classical antithetic variates motivate symmetric paired samples as a variance-reduction device.
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*
  (arXiv:2209.03003), motivates examining deterministic transport geometry and source-to-endpoint
  sensitivity.
