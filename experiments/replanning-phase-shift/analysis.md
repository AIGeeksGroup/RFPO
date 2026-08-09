# H55 Replanning Phase-Shift Prerequisite Analysis

## Outcome

H55 is refuted. The common-seed audit was valid, but first-chunk phase shifts changed the outcome of
only 3/20 seeds and the hindsight phase oracle improved over the official `phi=8` control by only
2/20 successes. These miss the locked 4/20 sensitivity and +3/20 oracle gates. A BCP continuation
head is therefore not authorized, and no extra seeds, phase variants, or heuristic continuation
triggers will be tested.

## Validity

All eight locked phases completed for all 20 environment seeds, for 160 finite episodes in total.
For each seed, the SHA-256 of the initial normalized observation was exactly identical across phases.
The official `phi=8` control scored 15/20, above the 12/20 validity floor. The archived outcome
matrix and episode records independently reproduce every aggregate statistic below.

## Results

| First executed prefix `phi` | Successes | Episodes | Success rate |
|---:|---:|---:|---:|
| 8 (official control) | 15 | 20 | 75% |
| 1 | 17 | 20 | 85% |
| 2 | 16 | 20 | 80% |
| 3 | 16 | 20 | 80% |
| 4 | 16 | 20 | 80% |
| 5 | 15 | 20 | 75% |
| 6 | 14 | 20 | 70% |
| 7 | 16 | 20 | 80% |

Only seed indices 3, 8, and 14 were phase-sensitive. The hindsight oracle succeeded on 17/20 seeds,
compared with 15/20 for the official phase, for a +2/20 upper-bound gain. The apparent 17/20 score
at `phi=1` is not a method result: the protocol explicitly forbids selecting a fixed noncontrol phase,
and the full matrix shows that the phase effect is confined to three seeds.

## Interpretation

Changing a single initial execution prefix has some behavioral effect, but the observed dependence is
too sparse to justify BCP's learned adaptive-horizon infrastructure and heavy trajectory-level GRPO
training. Together with H27 and H54, this closes global shorter execution, fixed overlap averaging,
and phase-triggered adaptive continuation as near-term reward-improvement routes on the released Can
checkpoint.

## Artifacts

- `raw/audit_results.json`: metadata, hashes, 160 episode outcomes, phase-by-seed matrix, and analysis
- `raw/audit.log`: complete OSMesa audit log

This is an OSMesa mechanism audit, not an official EGL benchmark or real-robot result.
