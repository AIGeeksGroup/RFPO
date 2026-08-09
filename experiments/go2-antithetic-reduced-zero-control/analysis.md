# H72 Result: Reduced-Step Zero-Source Control

## Outcome

H72 is refuted. The new `zero32` cell completed 256 finite episodes and matched
H70's checkpoint, task, evaluation seed, and initial-observation hash exactly.

| Method | NFE/action | Mean return |
| --- | ---: | ---: |
| zero64 | 64 | 41.5765 |
| zero32 | 32 | 41.5882 |
| antithetic32 | 64 | 41.5720 |

`Antithetic32 - zero32` was -0.0162 return (paired SEM 0.0117; bootstrap
95% CI [-0.0368, +0.0090]), failing both reward gates. `Zero32 - zero64` was
+0.0117 with 95% CI [-0.0005, +0.0265], so halving deterministic integration
steps preserved return at this resolution.

## Decision

Do not claim antithetic projection as the best latency-oriented deployment.
H71 remains a valid implementation benchmark showing that a doubled batch is
faster than sequential pairing and random64, but H72 shows that reduced
integration depth, rather than symmetric projection, supplies the simpler
deployment-efficiency route. Retain H70 only as an improvement over the
official stochastic-source baseline at equal total NFE.
