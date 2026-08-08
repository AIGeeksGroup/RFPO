# H3a Analysis: Exploratory WP-Past Warm Start on Can

Date: 2026-08-08

Checkpoint: `95j3noe4_step_1000`, EMA weights, 10 Euler steps

Seed: 20260808, 10 parallel environments, 20 episodes per condition

| Source | Sampling | Successes | Episodes | Success rate | FPS |
|---|---|---:|---:|---:|---:|
| Gaussian | Zero | 17 | 20 | 85% | 203.7 |
| Gaussian | Random | 2 | 20 | 10% | 296.6 |
| Previous action, sigma 0.5 | Zero | 0 | 20 | 0% | 397.2 |
| Previous action, sigma 0.5 | Random | 0 | 20 | 0% | 404.3 |

## Decision

Stop the inference-only warm-start branch. It fails both predeclared gates: random success does not
improve and deterministic success falls below 50%. The result does not refute WarmPrior as published,
because the released policy was trained against a Gaussian source. It does refute the cheaper claim
that a WarmPrior can be grafted onto this checkpoint only at inference time.

The next valid H3 test must adapt the behavior-cloning objective to the previous-action source. A
short, matched Gaussian-versus-WP-Past continuation from the same checkpoint is the next screening
experiment before considering from-scratch BC.
