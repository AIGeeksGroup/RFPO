# H42 Square Checkpoint-Native Action Horizon Analysis

## Validity

All four cells loaded the same frozen final non-EMA H41 control checkpoint, used seed 20260915,
Euler-10 integration, 20 vectorized Square environments, and balanced one-episode-per-environment
accounting. Every cell completed 20 finite episodes. The saved summaries report the requested 16 or
8 executed action steps. OSMesa makes this a paired screening result, not an official benchmark
claim.

## Reward Results

| Mode | 16-step control | 8-step candidate | Difference | Gate | Result |
|---|---:|---:|---:|---:|---|
| zero | 10/20 (50%) | 7/20 (35%) | -3 | no worse than -1 | fail |
| random | 6/20 (30%) | 8/20 (40%) | +2 | at least +2 | pass |
| pooled | 16/40 (40%) | 15/40 (37.5%) | -1 | at least +3 | fail |

More frequent replanning moved Gaussian-source performance in the predicted direction, but the
gain was exactly the minimum random-mode threshold and came with a larger deterministic loss. The
pooled result decreased by one success rather than increasing by three.

## Decision

H42 is refuted under its preregistered screen. Restoring the base checkpoint's 8-action execution
horizon is not a stable inference improvement for the short-finetuned Square policy. Stop without
trying 4 or 12 actions, another seed, another checkpoint, 50-episode confirmation, or retraining at
8 actions.
