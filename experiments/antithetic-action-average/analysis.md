# Antithetic Action-Chunk Averaging: Result

## Outcome

H26 is refuted. The locked 20-episode screen passed, but its gain reversed on the independent
50-episode confirmation. No 200-episode evaluation or hyperparameter variation is justified.

| Stage | Seed | Episodes | Zero | Antithetic average | Difference |
|---|---:|---:|---:|---:|---:|
| Screen | 20260824 | 20 each | 15/20 (75%) | 18/20 (90%) | +15 points |
| Confirmation | 20260825 | 50 each | 37/50 (74%) | 34/50 (68%) | -6 points |
| Pooled | two seeds | 70 each | 52/70 (74.29%) | 52/70 (74.29%) | 0 points |

The candidate produced finite actions in the one-episode smoke and both formal runs. Measured FPS
was 105.2 versus 100.2 in the screen and 93.8 versus 96.3 in confirmation. Environment rendering
dominates these OSMesa measurements, so they do not establish a policy-only latency advantage; they
do show that the paired inference path remained operational at benchmark parallelism.

## Interpretation

The exact pooled tie and sign reversal are consistent with ordinary small-sample rollout variation,
not a stable benefit from canceling source-odd endpoint components. The result does not show that
the flow endpoint is linear in its source, only that averaging one `z,-z` pair does not improve this
checkpoint's Can success reliably enough to pursue.

Stop H26 without changing source scale, averaging rule, number of pairs, sampling steps, checkpoint,
or evaluation seed. Retain the evaluation mode as reproducible infrastructure, but do not present it
as a benchmark improvement.
