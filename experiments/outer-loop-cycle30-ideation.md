# Outer-Loop Cycle 30: Average the Online Policy Trajectory

## Reflection

H42 and H43 moved zero- and Gaussian-source success in opposing directions. The next candidate
should smooth a learned policy trajectory without selecting a source-specific inference rule or
changing the online objective. H25's BC-to-final midpoint failed on Can, but that experiment mixed a
pretraining anchor with one finetuned endpoint; it did not average multiple policies from the same
online optimization basin.

## Candidates

| Rank | Candidate | Rationale | Decision |
|---:|---|---|---|
| 1 | Uniform average of all actor-update checkpoints | SWA; coefficient-free, no checkpoint selection, inference-only screen | Select H44 |
| 2 | Exponential moving average during online training | trajectory smoothing, but introduces a decay and requires retraining | Park behind H44 |
| 3 | Greedy model soup selected on reward | may improve reward, but directly overfits checkpoint selection | Reject |
| 4 | Average H43 control and reset candidate | combines source-mode tradeoffs post hoc and reuses a failed method | Reject |
| 5 | BC-to-final interpolation on Square | direct repeat of H25 with a new task | Reject |
| 6 | Per-layer checkpoint averaging | adds arbitrary layer choices before global averaging is supported | Reject |

## Selection

H44 averages the four checkpoints after actor updates in the fixed H43 official-control trajectory.
It excludes the iteration-1 critic-only checkpoint, uses equal weights, and compares against the
already-frozen final checkpoint. This is a direct low-cost test of trajectory smoothing before any
EMA retraining or larger benchmark run.
