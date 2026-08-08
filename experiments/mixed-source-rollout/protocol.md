# Protocol: Mixed-Source Rollout Distillation for Can

## Hypothesis

The released Can policy succeeds about 74% with a zero source but only about 10% with a Gaussian
source. Using a small fixed subset of zero-source environments during online collection should supply
more successful action chunks for FPO updates, while retaining a majority of Gaussian-source
environments for exploration. The shared update can distill those successful modal trajectories into
the stochastic policy and improve success on the held-out Gaussian-source portion.

## Pilot

Run one five-update seed-0 candidate from the released Can step-1000 EMA checkpoint. Keep the completed
`can_maskreset_control_5iter_seed0` as the matched same-code control. Keep all official settings fixed,
including GAE lambda 0.99 and the released cumulative validity-mask behavior. Assign a fixed 20% of
the 30 collection environments to zero-source action generation and the remaining 80% to Gaussian
sources. Track collection successes and completed episodes separately by source.

The primary metric is Gaussian-source collection success pooled over actor-update iterations 2 through
5. Continue only if it exceeds the control's 53/604 (8.77%) by at least two percentage points. As a
mechanism check, the mixed rollout must collect at least 1.5 times as many total successful episodes as
the control over the same iterations. Final 50-episode zero success may not be more than five points
below the control's 84%, final random success may not be below the control's 10%, and all losses and
gradient norms must remain finite.

Overall mixed-source collection success is diagnostic only and cannot satisfy the primary gate.

## Stop Conditions

- Stop if Gaussian-source collection or either evaluation gate fails.
- If the pilot passes, run a matched control and candidate with one new seed before longer training.
- Do not sweep the mixture fraction, combine with reflow, reset the validity mask, or change the reward
  and advantage estimator in this pilot.
