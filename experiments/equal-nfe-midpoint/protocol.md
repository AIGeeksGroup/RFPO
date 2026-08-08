# Protocol: H38 Equal-NFE Midpoint Flow Sampling

## Hypothesis H38

The released FPO Can policy is sampled with first-order Euler integration. At the same ten velocity-
network evaluations, five explicit-midpoint steps should approximate the high-resolution flow
endpoint more accurately than ten Euler steps, for both deterministic zero-source actions and
Gaussian-source exploration, without collapsing action diversity.

This is an inference-only numerical integration change. It does not alter the checkpoint, source
distribution, action horizon, observation conditioning, time interval, reward, or policy training.

## Locked Method

- Control: official Euler integration with 10 equal steps from `t=1` to `t=0` (10 NFE).
- Candidate: explicit midpoint with 5 equal steps (10 NFE). For each step, evaluate `v1=v(x,t)`, form
  `x_mid=x+dt*v1/2`, evaluate `v_mid=v(x_mid,t+dt/2)`, then set `x=x+dt*v_mid`.
- Reference: official Euler integration with 64 equal steps (64 NFE).
- Apply the checkpoint's existing velocity/output parameterization and transported clipping at both
  midpoint evaluations exactly as in Euler sampling.
- Midpoint is inference-only and incompatible with SDE sampling in this experiment.
- Do not test Heun, RK4, alternative schedules, 4/6 midpoint steps, unequal NFE, or solver ensembles.

## Fixed Geometry Audit

- Checkpoint: released Can `95j3noe4_step_1000`, EMA weights.
- Dataset observations: `ankile/robomimic-mh-can-image`.
- Seed: `20260909`.
- Audit set: 128 fixed observations in two 64-sample batches.
- Source conditions: one fixed standard-Gaussian source per observation and an all-zero source for the
  same observations.
- Compare unnormalized action endpoints against the source-matched Euler-64 endpoint.
- Diversity: on 64 copies of one fixed observation, compare mean element standard deviation and mean
  pairwise action distance under fixed Gaussian sources.

## Gates

All gates must pass before any environment rollout:

1. All control, candidate, and reference actions are finite.
2. Midpoint-5 endpoint MSE is at most 80% of Euler-10 endpoint MSE under Gaussian sources.
3. Midpoint-5 endpoint MSE is at most 80% of Euler-10 endpoint MSE under zero sources.
4. Midpoint-5 retains 90-110% of Euler-10 mean element standard deviation and mean pairwise distance
   under the fixed Gaussian diversity set.
5. Measured NFE is exactly 10 for both control and candidate.

Stop without rollout, step-count changes, solver variants, another audit seed, or policy retraining if
any gate fails. A full pass authorizes only a separately committed balanced 20-episode zero/random
Can screen; it is not itself reward-improvement evidence.

