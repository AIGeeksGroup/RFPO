# H60 Go2 Equal-NFE Midpoint Analysis

## Result

H60 is refuted under its locked reward gates. Explicit midpoint integration produced a much more
accurate endpoint at exactly the same 64 velocity-network evaluations as the official Euler sampler,
but this numerical gain did not improve paired closed-loop return.

## Stage A: Fixed-State Geometry

The formal H200 audit used 256 rollout observations, seed `20261060`, and an Euler-256 reference.
Every output was finite and all measured NFE counts were exact.

| Source | Euler-64 MSE | Midpoint-32 MSE | MSE ratio | Gate |
|---|---:|---:|---:|:---:|
| zero | 1.83438e-6 | 2.29631e-7 | 0.12518 | pass |
| random | 1.25623e-5 | 1.71880e-6 | 0.13682 | pass |

For random sources, midpoint/control mean element standard deviation was `1.01327` and mean
pairwise distance was `1.01307`. Both lie inside the locked `[0.98, 1.02]` diversity interval. All
nine geometry, finite-value, NFE, and diversity gates passed.

## Stage B: Paired Reward Screen

The control and candidate used the same final reproduced Go2 checkpoint, environment seed
`20261061`, source seed `20261060`, 50 environments, and exactly one episode per environment in
each source mode. Both produced 50 finite episodes per mode.

| Source | Euler-64 mean | Midpoint-32 mean | Paired difference | Paired bootstrap 95% |
|---|---:|---:|---:|---:|
| zero | 41.11486 | 41.11980 | +0.00493 | [-0.00316, +0.01354] |
| random | 39.53676 | 39.47403 | -0.06274 | [-0.07855, -0.04937] |

The average of the two mode differences was `-0.02890`; the pooled paired bootstrap interval was
`[-0.04004, -0.01866]`. The candidate failed random non-degradation and the required `+0.5` average
gain. The small random-source loss is statistically stable under the paired screen, although it is
small relative to the approximately 40-return scale.

## Decision

Stop H60 without a solver, step-count, seed, checkpoint, or training variant. Higher integration
fidelity is a useful sampling property, but neither Can nor dense-reward Go2 supports it as a reward
objective for the released policies. Reflow and second-order integration remain valid efficiency or
endpoint-fidelity tools only; they are not benchmark-improvement methods under the completed tests.

Raw JSON and complete logs are stored in `results/` and `logs/`. The downloaded files match the
remote SHA-256 checksums recorded at completion.
