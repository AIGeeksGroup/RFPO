# Protocol: Full-Lambda GAE for Sparse Can Success

## Hypothesis

The official Can setting's `gamma=0.99, lambda=0.99` attenuates a terminal-success residual by
`(gamma * lambda)^d`. At the 300-step horizon this is about 0.0024. Increasing only GAE lambda to 1.0
raises the coefficient to about 0.049, so successful episodes provide materially stronger credit to
early manipulation actions and improve FPO++ reward learning despite higher estimator variance.

## Pilot

Compare one seed-0 candidate against the completed same-code control
`can_maskreset_control_5iter_seed0`. Start from the released Can step-1000 EMA checkpoint and keep the
official settings fixed: `gamma=0.99`, 30 environments, 1600 collection steps, 8 CFM samples, 10
Euler steps, 10 update epochs, no validity-mask reset, and 240,000 environment steps. Change only
`gae_lambda` from 0.99 to 1.0. Evaluate 50 zero- and 50 random-sampling episodes at iterations 1 and 5.

The primary metric is collection success pooled over actor-update iterations 2 through 5. Continue
only if lambda 1.0 exceeds the control's 53/604 (8.77%) by at least two percentage points. Final zero
success may not be more than five points below the control's 84%, final random success may not be
below the control's 10%, and losses and gradient norms must remain finite.

## Stop Conditions

- Stop if the primary reward or evaluation gate fails.
- If it passes, run matched control and candidate with one new seed before longer training.
- Do not change discount, mask reset, advantage normalization, or minibatch sampling in this pilot.

