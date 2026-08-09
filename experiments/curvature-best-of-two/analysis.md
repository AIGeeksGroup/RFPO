# Analysis: H46 Curvature-Selected Gaussian Best-of-Two Screen

## Outcome

H46 passed its pipeline audit and its first balanced reward gate. This is a positive screening result
that authorizes an independently seeded confirmation; it is not yet a stable improvement or an
official benchmark claim.

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

The candidate exactly meets the locked +3/20 gate. Zero-source inference does not enter the new code
path, so its paired value is identical by construction and the random gain is also a +3 pooled gain
against a shared zero cell.

The formal candidate recorded 508 replanning decisions. Both branches were selected exactly 254
times, all audit invariants passed, and selected curvature was 11.58% lower on average than the two-
candidate mean. The reward gain therefore accompanies an active, non-degenerate selection mechanism.

## Interpretation and Decision

Global flow straightening did not produce a stable Can reward gain, but within-observation curvature
may still provide useful relative confidence between stochastic action candidates. H46 changes only
Gaussian inference and doubles policy network evaluations at replanning time; it does not improve
training sample efficiency.

Keep H46 active and lock one independent 50-episode balanced confirmation before another rollout.
Do not change candidate count, score, checkpoint, action horizon, solver, or renderer. Failure of the
confirmation gate closes the method without another seed or variant.
