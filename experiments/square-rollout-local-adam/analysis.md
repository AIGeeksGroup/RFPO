# H43 Square Rollout-Local Actor Adam Analysis

## Validity

The two-environment smoke completed two actual iterations, saved finite checkpoints, and recorded the
expected vacuous iteration-2 reset (`0 -> 0` optimizer-state entries).

Both formal runs loaded the same released Square EMA checkpoint, used seed 20260916, and completed
five actual iterations with finite final checkpoints. Their first three collection records matched
exactly because the first actor update starts with empty Adam state and the first non-vacuous reset
occurs only after iteration-3 collection:

| Iteration | Control successes | Candidate successes | Control valid CFM | Candidate valid CFM |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 5097/5120 | 5097/5120 |
| 2 | 9 | 9 | 5036/5120 | 5036/5120 |
| 3 | 4 | 4 | 5024/5120 | 5024/5120 |
| 4 | 4 | 4 | 4998/5120 | 4994/5120 |
| 5 | 7 | 6 | 4946/5120 | 4951/5120 |

Candidate reset-state entries before clearing were `[0, 8, 8, 8]` in iterations 2-5 and were zero
after every clear. The learning rate remained `1e-5`. This passes the locked mechanism-activity and
pairing requirements.

## Reward Results

| Mode | Persistent-Adam control | Rollout-local candidate | Difference | Gate | Result |
|---|---:|---:|---:|---:|---|
| zero | 5/20 (25%) | 9/20 (45%) | +4 | no worse than -1 | pass |
| random | 8/20 (40%) | 5/20 (25%) | -3 | at least +2 | fail |
| pooled | 13/40 (32.5%) | 14/40 (35%) | +1 | at least +3 | fail |

Every evaluation cell completed 20 finite episodes with one episode per environment, seed 20260917,
Euler-10 integration, and 16 executed actions. Clearing cross-rollout Adam state substantially moved
the policy toward its deterministic mode but reduced Gaussian-source success.

## Decision

H43 is refuted under its preregistered screen. The optimizer state is active and behaviorally
consequential, but rollout-local Adam changes the zero/random tradeoff rather than producing a stable
benchmark improvement. Stop without resetting only first or second moments, changing betas, another
seed, longer training, checkpoint selection, or confirmation.
