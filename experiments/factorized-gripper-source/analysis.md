# H57 Factorized Gaussian Arm / Zero Gripper Source Analysis

## Outcome

H57 is refuted under its locked gate. The paired random-source screen was valid and the candidate
improved from 1/20 to 3/20, rescuing two seeds without losing the control success. However, the
predeclared requirement was at least +3/20, so the observed +2/20 does not authorize independent
confirmation, a variance sweep, mixture, state-conditioned source, or training.

## Stage A: Paired Smoke

The two-environment smoke completed both conditions with finite episodes. Initial normalized
observations matched exactly, all 76 common `(environment, replan)` arm-source hashes matched, the
candidate gripper latent was exactly zero, and the control gripper source was active. The maximum
first-action difference was 0.2243. All implementation gates passed; the 0/2 reward tie was not used
as method evidence.

## Stage B: Paired Reward Screen

Both conditions used the released Can step-1000 EMA checkpoint, 10 Euler steps, 16/8 horizons,
environment seeds `20261031..20261050`, and stateless source seed `20261101`.

| Condition | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| Full Gaussian source | 1 | 20 | 5% |
| Gaussian arm / zero gripper source | 3 | 20 | 15% |

The control lies inside the locked 1-8 success regime. Across 734 common active plan keys, every arm
source hash was exact. The observed arm-source mean and standard deviation were approximately
0.0076 and 0.9995, confirming that the intervention did not temper arm exploration. Candidate seeds
`20261036` and `20261037` changed from failure to success; no control success was lost. The +2/20 gain
nevertheless falls one success below the locked +3/20 gate.

## Interpretation

Gripper latent noise can influence random-source outcomes, and coordinate factorization is more
promising than H56's reward-neutral output binarization. But three candidate successes are too few to
distinguish stable improvement from sparse-screen variation under the preregistered rule. This result
is retained as a directional clue, not a benchmark gain. H57 and nearby gripper-source variance or
mixture variants are closed.

## Artifacts

- `raw/smoke/audit_results.json` and `audit.log`: two-seed paired implementation smoke
- `raw/audit/audit_results.json` and `audit.log`: complete 20+20 paired reward screen

These are OSMesa method screens, not official EGL benchmark or real-robot results.
