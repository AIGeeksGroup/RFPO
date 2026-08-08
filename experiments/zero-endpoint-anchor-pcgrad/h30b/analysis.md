# H30b Analysis: Balanced Zero-Endpoint PCGrad Screen

## Outcome

The corrected balanced screen is valid but does not authorize confirmation. The candidate gained one
success in each sampling mode and two successes pooled, but the preregistered primary random-source
gate required at least two additional random successes.

| Mode | Control | Candidate | Difference | Gate |
|---|---:|---:|---:|---:|
| Zero source | 19/20 (95%) | 20/20 (100%) | +1 | no worse than -1: pass |
| Random source | 9/20 (45%) | 10/20 (50%) | +1 | at least +2: fail |
| Pooled | 28/40 (70%) | 30/40 (75%) | +2 | at least +2: pass |

The previously recorded projection-active fraction was 36/320 (11.25%), so the 10-90% mechanism
gate also passed. Each evaluation used 20 environments with exactly one completed episode from every
environment, eliminating the first-completion censoring that invalidated H30.

## Decision

H30b is refuted under its locked small-screen gate. The same-direction +5-point differences in both
modes are suggestive but only one episode per mode and are not sufficient evidence of a benchmark
gain. Per protocol, do not run the 50-episode confirmation, retrain, select a checkpoint, or tune the
anchor/projection from this result.
