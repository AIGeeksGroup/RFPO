# Analysis: H46 Curvature-Selected Gaussian Best-of-Two Screen

## Outcome

H46 passed its pipeline audit. Its first balanced reward comparison appeared to pass, but a later
seed-propagation audit invalidated that reward decision before confirmation.

The seed-20260921 smoke recorded 202 replanning decisions. All actions, paths, and scores were finite;
every selection matched the lower of the two normalized straightness errors; both exchangeable
branches were selected (107 and 95 times); and every record used exactly two candidates with ten
Euler steps. The lower-curvature choice reduced score by 11.38% on average relative to the two-score
mean. Smoke success was 2/8 and is diagnostic only.

## Balanced Reward Screen

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

## Interpretation and Decision

Global flow straightening did not produce a stable Can reward gain, but within-observation curvature
may still provide useful relative confidence between stochastic action candidates. H46 changes only
Gaussian inference and doubles policy network evaluations at replanning time; it does not improve
training sample efficiency.

Keep H46 active only for an unchanged corrected screen after deterministic per-environment seeding is
wired into the evaluation entry point. Do not run confirmation, change candidate count, score,
checkpoint, action horizon, solver, renderer, seed, or reward gate based on the invalid comparison.
