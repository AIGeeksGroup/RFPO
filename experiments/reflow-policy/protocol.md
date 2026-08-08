# Protocol: Conditional Reflow for FPO++

## Hypothesis

Re-training on observation-conditioned `(initial_noise, sampled_action)` endpoint pairs from a trained policy will reduce path curvature and preserve or improve evaluation return at fewer Euler steps.

## Stage-A Measurements

- Mean integrated straightness error along policy trajectories.
- Action endpoint error when reducing Euler steps from the official setting.
- Can endpoint error under 64, 10, 4, 2, and 1 Euler steps on fixed observation/source pairs.
- Can success at the official 10 steps and a reduced four-step setting after the geometry gate passes.
- Random-noise action diversity and training return, to detect exploration collapse.

## Pilot

Use the released and reproduced Can checkpoint as a frozen 64-step EMA teacher. Initialize both
students from the same EMA weights with fresh optimizers. Train an ordinary-CFM control on dataset
actions and a one-stage reflow student on teacher-generated `(Gaussian source, endpoint)` pairs for
100 updates each, batch size 64, seed 20260808. Generate reflow pairs online from the same dataset
observation stream; do not persist a large derived dataset.

Evaluate 128 fixed observation/source pairs. Continue to environment rollouts only if reflow reduces
64-step normalized straightness error by at least 20%, lowers path-length ratio, and does not have
higher four-step endpoint MSE than the equal-budget control. Then run 20-episode Can screens with
matched seeds. Reflow must preserve official-step zero success within 15 percentage points and beat
the control by at least one success at four steps before any longer run or online RL experiment.

## Stop Conditions

- Stop if normalized straightness error does not improve by at least 20% over the control.
- Stop if 8-step evaluation return drops by more than 5% without a compensating robustness benefit.
- Stop if action diversity collapses substantially under random-noise sampling.
