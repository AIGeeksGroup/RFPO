# Protocol: H31 Marginal-Preserving Temporal Gaussian Source

## Hypothesis H31

Correlating consecutive Gaussian flow sources reduces action-chunk boundary jitter and improves
random-source Can success while preserving the exact standard-Gaussian marginal distribution seen
during behavior-cloning training.

For each environment and action-chunk replan, use

`z_k = rho * z_(k-1) + sqrt(1 - rho^2) * epsilon_k`,

where the first source and every post-reset source are independently standard Gaussian. Lock
`rho = 0.9`; do not sweep or tune it. The IID control is the existing `rho = 0` behavior.

## Mechanism Audit

- Check 16,384 scalar AR(1) transitions from a fixed seed.
- Require absolute marginal mean <= 0.03, marginal standard deviation in [0.97, 1.03], and measured
  lag-one correlation within 0.03 of 0.9.
- On 64 fixed Can observations from the released step-6000 EMA checkpoint, compare 32 consecutive
  IID and AR(1) source pairs using matched innovations and 10 Euler steps.
- Primary mechanism metric: mean L2 change between consecutive predicted action chunks. Require at
  least 10% reduction for AR(1).
- Require AR(1) action diversity, measured as mean per-coordinate standard deviation across chunks,
  to retain at least 70% of IID diversity. All sources and actions must be finite.
- Stop without rollout evaluation if any mechanism gate fails.

## Reward Screen

- Checkpoint: exact released Can `95j3noe4_step_6000` EMA actor; no training.
- Conditions: official IID random source versus AR(1) random source with `rho = 0.9`.
- Shared evaluation seed: 20260831.
- Sampling: 10 Euler steps, 16 executed actions per chunk, 20 episodes, 50 OSMesa environments.
- Primary gate: AR(1) must exceed IID by at least 2/20 successes. All actions must be finite.
- Zero-source evaluation is omitted because H31 does not alter zero sampling.
- Stop without tuning or confirmation on failure. On success only, run a 50-episode independent
  confirmation at seed 20260901 and require a nonnegative success difference before any larger or
  healthy-EGL benchmark claim.

OSMesa is accepted only for the paired mechanism and reward screens. Final benchmark claims still
require healthy EGL or an explicitly authorized rendering GPU.
