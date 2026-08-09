# Protocol: H60 Go2 Equal-NFE Midpoint Sampling

## Hypothesis

At the same 64 velocity-network evaluations as the official Go2 Euler-64 sampler, explicit
midpoint-32 more accurately integrates the learned flow and improves final-checkpoint locomotion
return under both zero and Gaussian sources.

## Locked Method

- Checkpoint: reproduced official seed-42 Go2 `model_1499.pt`; EMA actor weights baked into the
  checkpoint by the official runner.
- Control: 64 uniform Euler steps from `t=1` to `t=0` (64 NFE).
- Candidate: 32 uniform explicit-midpoint steps over the same interval (64 NFE).
- Reference for the fixed-state audit only: 256 uniform Euler steps (256 NFE).
- The policy weights, observation normalizer, action scale, environment, commands, source tensors,
  episode horizon, and action clipping remain unchanged.
- No Heun/RK4 variant, solver schedule sweep, checkpoint selection, or policy training is allowed.

## Stage A: Fixed-State Audit

- Collect 256 normalized observations from a frozen official-policy rollout, then evaluate the same
  observations and source tensors with all three solvers.
- Audit zero and standard-Gaussian sources separately with seed `20261060`.
- Require finite endpoints, exact measured NFE, and midpoint endpoint MSE no greater than 25% of
  Euler-64 MSE against Euler-256 in both source modes.
- For Gaussian sources, require midpoint/control mean element standard deviation and mean pairwise
  endpoint distance ratios in `[0.98, 1.02]`.

Failure stops H60 before environment reward evaluation.

## Stage B: Fixed-Checkpoint Reward Screen

- Environment seed: `20261061`, shared by control and candidate.
- 50 parallel environments, exactly one completed episode per environment and source mode.
- Evaluate zero and fresh standard-Gaussian sources separately.
- Record per-episode returns and lengths, mean, standard deviation, standard error, paired mean
  difference, and a paired bootstrap 95% interval.
- The candidate passes only if both source-mode mean differences are nonnegative, the mean of the two
  source-mode differences is at least `+0.5` return, and the pooled paired bootstrap interval excludes
  a degradation worse than `-0.25` return.

A pass authorizes an independent environment-seed confirmation before training or larger benchmark
claims. A failure closes higher-order Go2 integration without solver, step-count, seed, checkpoint,
or training variants.

