# Protocol: Tempered-Source Rollout Bridge for Can

## Hypothesis

The H8 zero-source subset supplied many successes but used a delta distribution with little overlap
with the official Gaussian source. Replacing that delta with a tempered Gaussian `N(0, 0.5^2 I)`
should retain more successful near-modal trajectories while continuously overlapping `N(0, I)`, so
shared FPO updates can improve the official full-noise policy rather than only its deterministic mode.

## Pilot

Run one five-update seed-0 candidate from the released Can step-1000 EMA checkpoint. Keep the completed
`can_maskreset_control_5iter_seed0` as the matched same-code control. Keep GAE lambda 0.99, the released
validity-mask behavior, and all official policy settings fixed. Assign a fixed 20% of the 30 collection
environments to source standard deviation 0.5 and the remaining 80% to the official standard deviation
1.0. Track successes and completed episodes separately by source group.

The primary metric is full-noise collection success pooled over actor-update iterations 2 through 5.
Continue only if it exceeds the control's 53/604 (8.77%) by at least two percentage points. The
tempered subset must collect at least twice the control's success rate as a mechanism check. Final
50-episode zero success may not be more than five points below the control's 84%, final random success
may not be below the control's 10%, and all losses and gradient norms must remain finite.

Overall mixed collection success and tempered-subset success are diagnostic only.

## Stop Conditions

- Stop if the full-noise collection or either evaluation gate fails.
- If the pilot passes, run a matched control and candidate with one new seed before longer training.
- Do not sweep source scale or mixture fraction, combine with reflow, reset the validity mask, or
  change the reward and advantage estimator in this pilot.
