# Protocol: Conditional Reflow for FPO++

## Hypothesis

Re-training on observation-conditioned `(initial_noise, sampled_action)` endpoint pairs from a trained policy will reduce path curvature and preserve or improve evaluation return at fewer Euler steps.

## Confirmatory Measurements

- Mean integrated straightness error along policy trajectories.
- Action endpoint error when reducing Euler steps from the official setting.
- Go2 evaluation return under 64, 16, 8, 4, 2, and 1 Euler steps.
- Random-noise action diversity and training return, to detect exploration collapse.

## Pilot

Use one reproduced Go2 checkpoint. Generate a bounded reflow dataset from fixed observations and policy noises, train a reflow copy for a short budget, and compare against an equal-budget ordinary CFM control. Do not launch full online RL until curvature decreases and low-step return is non-inferior.

## Stop Conditions

- Stop if curvature does not improve over the control.
- Stop if 8-step evaluation return drops by more than 5% without a compensating robustness benefit.
- Stop if action diversity collapses substantially under random-noise sampling.

