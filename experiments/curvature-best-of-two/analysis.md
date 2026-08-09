# Analysis: H46 Curvature-Selected Gaussian Best-of-Two Screen

## Outcome

H46 passed its pipeline audit but failed the corrected reward screen. Its first balanced reward
comparison appeared to pass, but a later seed-propagation audit invalidated that decision before
confirmation. Repeating the unchanged screen with deterministic per-environment seeds reversed the
apparent gain.

The seed-20260921 smoke recorded 202 replanning decisions. All actions, paths, and scores were finite;
every selection matched the lower of the two normalized straightness errors; both exchangeable
branches were selected (107 and 95 times); and every record used exactly two candidates with ten
Euler steps. The lower-curvature choice reduced score by 11.38% on average relative to the two-score
mean. Smoke success was 2/8 and is diagnostic only.

## Invalid Initial Reward Screen

Both conditions used the frozen H43 official-control Square checkpoint, seed 20260922, 20 environments
contributing one episode each, Euler-10, 16 executed actions, no EMA, and OSMesa.

| Random-source mode | Success | Difference |
|---|---:|---:|
| Ordinary one-sample control | 5/20 (25%) | - |
| Lower-curvature best-of-two | 8/20 (40%) | +3/20 |

The candidate numerically meets the locked +3/20 gate. However, `eval_checkpoint.py` did not pass
`cfg.seed` to the spawned environment constructors, so robosuite received `seed=None` in both cells.
These remain balanced independent samples but do not implement the intended evaluation seed and
cannot satisfy the protocol gate.

The formal candidate recorded 508 replanning decisions. Both branches were selected exactly 254
times, all audit invariants passed, and selected curvature was 11.58% lower on average than the two-
candidate mean. The reward gain therefore accompanies an active, non-degenerate selection mechanism.

## Corrected Seeded Reward Screen

Both corrected conditions used deterministic environment seeds `20260922 + env_id`, with every
other locked setting unchanged.

| Random-source mode | Success | Difference |
|---|---:|---:|
| Ordinary one-sample control | 7/20 (35%) | - |
| Lower-curvature best-of-two | 4/20 (20%) | -3/20 |

The corrected candidate recorded 504 replanning decisions. Both branches were selected (266 and
238 times), all values were finite, every selection was the exact score minimum, and every record
used two candidates with ten sampling steps. Selection lowered normalized curvature by 12.62% on
average relative to the candidate-pair mean. The mechanism was therefore active and valid, but it
degraded rather than improved reward.

## Interpretation and Decision

Within-observation curvature is not a useful relative confidence score for this frozen Square policy.
The intervention reliably chose straighter ODE paths while success fell by three episodes, reinforcing
the earlier finding that flow geometry and closed-loop action quality are different objectives. H46
also doubles policy network evaluations at replanning time and does not improve training sample
efficiency.

Refute H46. Do not run confirmation, change candidate count or score, try thresholds, change the seed,
or test nearby curvature-ranking variants.
