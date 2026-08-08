# H38b Balanced Can Reward Screen

## Decision

H38b is refuted. Midpoint-5 improved zero-source success but degraded Gaussian-source success by the
same number of episodes, leaving pooled success unchanged and failing two locked reward gates. Stop
without another seed, solver, step count, schedule, ensemble, or training run.

## Results

All four conditions used the released Can step-1000 EMA checkpoint, seed `20260910`, isolated OSMesa,
and balanced accounting with 20 environments contributing one completed episode each.

| Source | Euler-10 | Midpoint-5 | Difference |
|---|---:|---:|---:|
| Zero | `14/20` (70%) | `16/20` (80%) | `+2/20` |
| Gaussian random | `4/20` (20%) | `2/20` (10%) | `-2/20` |
| Pooled | `18/40` (45%) | `18/40` (45%) | `0/40` |

The Euler control passed both preregistered sanity ranges. Every evaluation completed normally and
recorded balanced accounting. Midpoint failed the primary `+3/40` pooled-success gate and exceeded
the allowed per-mode degradation by losing two random-source successes.

## Interpretation

H38 established that midpoint-5 is much closer to the Euler-64 endpoint at unchanged NFE. H38b shows
that this numerical fidelity does not monotonically improve the closed-loop reward of the released
policy: moving the solver endpoint toward the high-NFE action helps the deterministic mode on this
seed but hurts stochastic exploration equally. Midpoint remains an inference-fidelity result, not a
benchmark reward improvement.
