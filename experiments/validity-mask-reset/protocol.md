# Protocol: Reset CFM Validity Mask Per Rollout

## Hypothesis

The Can FPO++ rollout buffer reuses `cfm_value_invalid_stored` across iterations without clearing it.
Positions invalidated after an episode boundary therefore remain invalid in unrelated future
rollouts. Resetting only this mask at the start of every collection iteration will preserve more
valid on-policy chunks and improve early reward learning.

## Mechanism Check

Run a one-update smoke with the reset enabled. It must complete with finite losses and report a valid
CFM action fraction in `[0.95, 1.0]`. Unit tests must show that enabling the option clears stale mask
entries in place, while disabling it preserves the released behavior.

## Matched Pilot

Run seed-0 control and candidate jobs from the released Can step-1000 EMA checkpoint. Keep all
official FPO++ settings fixed: 30 environments, 1600 collection steps, 8 CFM samples, 10 Euler steps,
10 update epochs, and 240,000 environment steps (five collection iterations). The only difference is
`reset_cfm_invalid_mask_each_iteration=False` versus `True`. Evaluate 50 zero- and 50 random-sampling
episodes at iterations 1 and 5.

Primary metric: collection success pooled over actor-update iterations 2 through 5. Continue only if
the candidate exceeds control by at least two percentage points. The candidate's valid-CFM fraction
must remain at least 95% in every iteration, while the control must exhibit cumulative decline; this
is the mechanism gate. Final zero success may not be more than five points below control, final random
success may not be lower, and all losses and gradient norms must remain finite.

## Stop Conditions

- Stop this candidate if the mechanism gate or primary reward gate fails.
- If it passes, run one matched confirmation seed before any longer job.
- Do not combine it with GAE, minibatch, or reflow changes in this pilot.

