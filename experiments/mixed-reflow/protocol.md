# Protocol: Mixed Dataset/Reflow Endpoints

## Hypothesis

A 50/50 per-sample mixture of dataset-action endpoints and frozen-teacher reflow endpoints will retain
at least half of pure reflow's straightening benefit while restoring enough action-distribution
grounding to improve Can success over an equal-budget ordinary-CFM control.

## Pilot

Initialize from the released Can EMA checkpoint with a fresh optimizer. Use training seed 20260808,
batch size 64, 100 updates, a frozen 64-step EMA teacher, and reflow teacher probability 0.5. Compare
against the already completed seed-20260808 100-update CFM control and pure-reflow runs. Do not sweep
the mixture probability.

First evaluate 128 fixed observation/source pairs with seed 20260808. Continue only if normalized
64-step straightness error is at least 10% below the control, four-step endpoint MSE is below the
control, and 64-step action diversity does not fall by more than 5%.

If the geometry gate passes, run one 20-episode, 10-environment Can screen at 10 Euler steps with
evaluation seed 20260808 for zero and random sampling. Continue only if zero success exceeds the
control by at least two successes and random success is not zero when the matched control is nonzero.

## Stop Conditions

- Stop the mixed-endpoint direction if either geometry or rollout gate fails.
- Do not tune the mixture probability after observing this pilot.
- Do not launch online FPO++ or a large confirmation until the small rollout gate passes.

