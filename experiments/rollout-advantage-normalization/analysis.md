# Analysis: Rollout-Level Advantage Normalization

## Result

H13 is refuted. Both conditions completed five 48k-step iterations from the step-6000 checkpoint
with seed 20260812. Losses and gradients remained finite, and the candidate logged 3,000 valid
samples for each rollout-level moment calculation.

| Iteration | Minibatch control | Rollout candidate |
|---:|---:|---:|
| 1 (critic-only) | 111/182 (60.99%) | 100/182 (54.95%) |
| 2 | 106/180 (58.89%) | 98/179 (54.75%) |
| 3 | 97/179 (54.19%) | 114/185 (61.62%) |
| 4 | 97/179 (54.19%) | 107/186 (57.53%) |
| 5 | 125/187 (66.84%) | 107/184 (58.15%) |

The preregistered iterations 2-5 pool was 426/734 (58.04%) for rollout normalization versus
425/725 (58.62%) for control, a difference of -0.59 percentage points (two-sided Fisher
`p=0.832`). This misses the required +3-point gain.

Final control evaluation was 50/50 zero and 36/50 random. Candidate evaluation was 48/50 zero and
32/50 random. The candidate retained high zero-source success but trailed random evaluation by
8 points, also failing the non-degradation gate.

## Interpretation

Reusing fixed advantage moments changes individual update weights but does not improve their
reward signal. Minibatch rescaling drift is therefore not a material bottleneck in this
higher-signal Can regime. Stop without another seed or normalization, epsilon, learning-rate, or
clipping sweep. Keep the opt-in implementation for reproducibility; retain minibatch normalization
as the default.
