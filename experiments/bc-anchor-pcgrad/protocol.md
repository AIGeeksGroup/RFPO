# Protocol: BC-Anchor Conflict Projection Audit

## Hypothesis H28

Projecting an FPO++ actor gradient only when it conflicts with a frozen behavior-cloning
velocity-field anchor preserves initial-policy behavior while retaining most of the useful RL
surrogate improvement.

## Candidate Rule

Let `g_rl` be the gradient of the released clipped FPO++ policy loss. Let `g_bc` be the gradient of
mean squared velocity error between the current actor and its exact pre-update actor on the same
fixed observations and flow-space query points. For minimization, conflict is
`dot(g_rl, g_bc) < 0`. In that case use

`g_candidate = g_rl - dot(g_rl, g_bc) / ||g_bc||^2 * g_bc`.

Otherwise use `g_candidate = g_rl`. No auxiliary coefficient, projection margin, layer selection, or
gradient rescaling is allowed in this audit.

## Locked Audit

- Initialization: Can `95j3noe4_step_6000`, EMA weights and released Huber CFM configuration.
- Rollout: official full Gaussian source, seed 20260827, 16 environments, 320 collection steps, two
  iterations. Iteration 1 retains the released critic-only warmup; iteration 2 supplies audit data.
- Audit examples: 64 fully valid positive-advantage chunks, selected once by the fixed seed and split
  into two batches of 32. Stop as infrastructure-inconclusive if fewer than 64 exist.
- RL gradient: independent fixed MC8 FPO++ variables per audit batch, using released clipping and GAE.
- BC anchor: exact actor snapshot immediately before iteration-2 actor updates. At each selected
  chunk-start observation, use fixed seeded standard-Gaussian action-space points and fixed uniform
  flow times. Store the anchor velocity target once and never update it.
- Comparison: apply one control virtual SGD step followed by a second step from the resulting state,
  using `g_rl` versus `g_candidate`. The step norm is matched to the released clipped actor-gradient
  norm times `1e-5`; optimizer moments are excluded so the audit isolates direction.
- Measurements per batch at the second step: RL/BC gradient cosine, conflict flag, retained candidate
  norm, pre/post RL surrogate, pre/post BC velocity MSE, and cosine from each post-step RL gradient to
  its pre-step gradient.
- Audit tensors do not enter online training and the actor is restored exactly after each branch.

## Gates

All gates must pass in both batches:

1. gradients and metrics are finite and both second-step gradient norms are nonzero;
2. measured RL/BC conflict exists after the common first step, so projection is non-vacuous;
3. candidate BC velocity-MSE increase is at most 50% of the control increase;
4. candidate RL surrogate gain is positive and at least 70% of the control gain;
5. candidate post-step RL-gradient cosine to the pre-step gradient is at least the control cosine.

If the control BC increase is non-positive, gate 3 passes only when candidate BC MSE is no larger
than control. Stop without integration or reward training if any batch fails. Do not tune query
count, gates, seed, projection rule, or virtual step after observing results.

## Conditional Next Step

Only if all gates pass, implement the same projection in the actor optimizer and run a matched short
Can screen from step 6000. Freeze that reward protocol before implementation. No official-scale or
multi-benchmark run is authorized by this audit.

