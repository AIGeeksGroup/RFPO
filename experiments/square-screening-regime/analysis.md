# H40 Square Screening-Regime Analysis

## Validity

Both evaluations used the released `trc7rbt0_step_110000` EMA checkpoint, seed 20260912,
official Euler-10 inference, and 20 environments contributing exactly one completed episode each.
The summaries report balanced per-environment accounting, 20 finite episodes, and the expected
eight executed action steps. No model or environment setting differed between source modes.

This is an OSMesa screening-regime check, not a full official benchmark reproduction. The small
sample and renderer substitution are acceptable for deciding whether to start a paired method
screen, but not for a final reward claim.

## Results

| Source mode | Successes | Success rate | Average episode length | Gate |
|---|---:|---:|---:|---|
| zero | 9/20 | 45% | 319.8 | pass |
| random | 7/20 | 35% | 345.3 | pass |

Both modes lie within the locked non-saturation interval of 3-17 successes, and both completed all
20 episodes with finite returns. The random-source result is also near the paper's roughly 28.6%
base-success regime, though 20 episodes are too few to interpret the difference quantitatively.

## Decision

H40 is supported as a screening-regime result. Square has enough successes and enough remaining
headroom for inexpensive paired FPO++ continuation screens. This result does not demonstrate a
method improvement. Any candidate training run requires a new Square-specific protocol committed
before execution, followed by a tiny pipeline smoke before the paired screen.
