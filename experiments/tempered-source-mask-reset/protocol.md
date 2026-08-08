# Protocol: Tempered-Source Rollouts with Validity-Mask Reset

## Hypothesis

The tempered-source pilot supplied many successful short episodes but its reused invalid-action mask
fell to 93.91% valid by iteration 5, versus 95.81% in the official control. Resetting the mask each
rollout should recover roughly four additional percentage points of training actions in this setting,
allowing the overlapping guided trajectories to improve the official full-noise policy.

## Pilot

Run one five-update seed-0 candidate from the released Can step-1000 EMA checkpoint. Use the same 20%
`N(0, 0.5^2 I)` and 80% `N(0, I)` source mixture as H9 and change only
`reset_cfm_invalid_mask_each_iteration` from false to true. Compare with both the official same-code
control `can_maskreset_control_5iter_seed0` and the completed no-reset interaction control
`can_tempered05_5iter_seed0`.

The primary metric is full-noise collection success pooled over actor-update iterations 2 through 5.
It must exceed the official control's 53/604 (8.77%) by at least two percentage points and exceed H9's
44/484 (9.09%). The valid-CFM fraction must remain at least 97.5% each iteration. Final 50-episode zero
success may not be more than five points below the official control's 84%, final random success may
not be below 10%, and all losses and gradient norms must remain finite.

## Stop Conditions

- Stop if the full-noise collection, validity, or either evaluation gate fails.
- If the pilot passes, run a matched control and candidate with one new seed before longer training.
- Do not change source scale/fraction, GAE, CFM samples, or any policy hyperparameter in this pilot.
