# H66 Go2 Antithetic Source Ensemble: Screen Analysis

## Decision

H66 passes its locked mechanism audit and 256-environment paired random-mode screen. This authorizes
only the separately preregistered 4,096-environment independent confirmation; it is not yet an
official benchmark-improvement claim.

## Mechanism

Across eight consecutive 256-environment states, all 24,576 sampled source values and all endpoints
were finite. Negative sources were bitwise exact negations, candidate outputs were bitwise exact
float32 endpoint means, zero dispatch was bitwise unchanged, and actor parameters remained bitwise
unchanged. Candidate-to-zero normalized RMS was `0.03462`, versus `0.28726` for the positive random
endpoint, giving a cancellation ratio of `0.12050`. The candidate is active but removes about 88% of
the source-dependent displacement measured by this audit.

## Reward Screen

Both methods completed exactly 256 episodes with finite actions and returns. Their initial-
observation hashes and complete pre-generated Gaussian source-stream hashes matched exactly.

| Method | Mean return | Difference from control |
|---|---:|---:|
| Official random source | 40.32237 | - |
| One-pair antithetic average | 41.68368 | +1.36131 |

The paired SEM was `0.20704`, and the deterministic 20,000-resample bootstrap 95% interval was
`[1.02934, 1.81896]`. This passes the locked `+0.30` mean-gain and strictly-positive lower-bound
gates by wide margins. Gaussian inference uses two Euler-64 endpoint solves; zero inference remains
the unmodified single solve.

Raw geometry, evaluator, analysis, and log artifacts are archived under `results/screen/`.
