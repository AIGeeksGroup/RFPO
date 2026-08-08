# Protocol: Increased CFM Monte Carlo Samples for Can FPO++

## Hypothesis

The low random-sampling success of the Can base policy makes early FPO++ gradients noisy. Increasing
the per-action CFM samples from 8 to 16 will reduce Monte Carlo ratio/gradient variance and improve
early reward learning without reducing evaluation success.

## Pilot

Run matched official Can FPO++ jobs from the released step-1000 EMA checkpoint with seed 0. Keep 30
environments, 1600 collection steps per update, 10 Euler steps, 10 policy epochs, and all official Can
hyperparameters fixed. Compare `n_action_samples=8` against `16` for five RL iterations (240,000
environment steps). Evaluate 50 zero- and random-sampling episodes at iterations 1 and 5 only.

The primary metric is collection success pooled over actor-update iterations 2 through 5. Continue
only if the 16-sample run exceeds the 8-sample run by at least two percentage points. At iteration 5,
zero-sampling evaluation must be no more than five points below the control and random-sampling
evaluation must not be lower. Training losses and gradient norms must remain finite.

## Stop Conditions

- Stop if the primary collection-success gate fails.
- Stop if evaluation or numerical-stability gates fail.
- Do not try 32 samples or start a 5M-step run before this pilot passes.

