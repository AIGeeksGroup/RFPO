# H54 Protocol: ACT-Inspired Temporal Chunk Ensemble

## Hypothesis

At the official eight-step replanning boundary, averaging the previous 16-step prediction's unused
eight-step continuation with the new observation-conditioned eight-step prefix reduces action
discontinuity while retaining feedback, improving released Can step-1000 deterministic success.

## Locked Intervention

- Checkpoint: official released Can step-1000 EMA policy.
- Control: released 16-step prediction horizon, execute steps 0-7, discard steps 8-15.
- Candidate: execute the same first chunk. At every later replan, blend the saved old steps 8-15
  with the new steps 0-7 in unnormalized environment-action space.
- Fixed weights follow the released ACT evaluator for exactly two stored predictions:
  old continuation `0.50249998`, new prefix `0.49750002`, obtained by normalizing
  `[exp(0), exp(-0.01)]`.
- Save the new prediction's unmodified steps 8-15 for the following boundary. Do not recursively
  blend or retain a plan beyond its original 16-step horizon.
- Preserve zero source, 10 Euler integration steps, 16-step prediction horizon, eight executed
  actions, checkpoint weights, and one policy call per eight environment steps.
- The option is inference-only and disabled by default. It must not be used for FPO++ collection,
  because a weighted action does not have the returned single-flow path likelihood.
- No weight, recursive blend, action-space, cadence, solver, source, or checkpoint sweep is allowed.

## Stage A: Mechanism Audit

Run unit tests and a two-environment H200/OSMesa smoke with fixed seed `20261001` for at least three
replans per environment. The audit passes only if all conditions hold:

1. Candidate first-chunk actions are bitwise identical to the disabled candidate under identical
   observations and zero source.
2. Every candidate action and record is finite.
3. From the second replan onward, every executed candidate chunk reconstructs the locked weighted
   average within floating-point tolerance, and both old and new coefficients exceed `0.49`.
4. The median candidate boundary jump
   `||a_blend[0] - a_executed_previous[-1]||_2` is strictly below the median counterfactual raw-new
   jump measured from the same candidate trajectory.
5. The median normalized deviation from the old continuation lies in `[0.49, 0.51]` of the full
   old-to-new disagreement, proving that current visual feedback is not suppressed.
6. Resetting an environment clears its saved continuation, so its next chunk is again unmodified.

Any failed gate refutes H54 before reward evaluation. Do not adjust the weight, seed, task, tolerance,
or number of stored chunks after a failure.

## Stage B: Fixed-Seed Reward Screen

Only after Stage A passes, evaluate in this order:

- renderer: OSMesa paired method screen, not an official EGL benchmark or real-robot result
- seed: `20261001`, deterministic per-environment seeds `seed + env_id`
- environment: Can
- source: zero
- conditions: official control, then temporal-ensemble candidate
- budget: 20 completed episodes per condition using four environments and five episodes per
  environment
- video and W&B: disabled

The candidate passes only if all actions are finite and success improves by at least `+2/20`. A tie,
`+1/20`, or any degradation refutes H54 without another screen seed or neighboring variant.

## Independent Confirmation

If and only if Stage B passes, first commit its results and then lock an independent confirmation at
seed `20261002`: 50 completed episodes per condition, five environments, ten episodes per environment,
same control-first order. Require at least `+3/50` successes and finite actions. Only a confirmed gain
authorizes a healthy-EGL 200-episode official-scale evaluation; neither OSMesa stage supports an
official benchmark or real-robot claim.
