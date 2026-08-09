# H44 Square Actor-Trajectory SWA Analysis

## Validity

The candidate is the uniform float32 arithmetic mean of exactly the four locked actor-update
checkpoints: `step_10240`, `step_15360`, `step_20480`, and `step_25600`. The generated manifest
records 35 floating tensors and no copied non-floating tensors. All averaged weights were finite,
all four policy configs were byte-identical, and the final checkpoint was retained as the control.

Every evaluation cell completed 20 finite episodes with one episode per environment, shared seed
20260919, Euler-10 integration, 16 executed actions, and OSMesa rendering. OSMesa makes this a paired
screen rather than an official benchmark claim.

## Reward Results

| Mode | Final checkpoint | Uniform SWA | Difference | Gate | Result |
|---|---:|---:|---:|---:|---|
| zero | 7/20 (35%) | 8/20 (40%) | +1 | no worse than -1 | pass |
| random | 5/20 (25%) | 3/20 (15%) | -2 | at least +2 | fail |
| pooled | 12/40 (30%) | 11/40 (27.5%) | -1 | at least +3 | fail |

Uniform averaging slightly improved the deterministic source mode but reduced independent
standard-Gaussian success. The direction matches the broader pattern in H25 and H43: smoothing or
regularizing the short online trajectory can move behavior toward its modal policy while sacrificing
the stochastic-source behavior needed by FPO++ exploration.

## Decision

H44 is refuted under its preregistered screen. It passes only the zero-source non-degradation gate
and fails both primary improvement gates. Stop without changing the averaging window, adding the
critic-only checkpoint, using unequal weights, evaluating another seed, EMA retraining, or a
50-episode-per-mode confirmation.
