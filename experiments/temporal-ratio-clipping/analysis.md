# H33 Analysis: Temporal Per-Sample Ratio Clipping

## Outcome

H33 is refuted at the fixed mechanism audit. The candidate exactly preserved the behavior-policy
gradient but did not change clipping activity or gradient direction after the locked actor-size
virtual step. No online integration or temporal-group variant is authorized.

## Audit Health

- Iteration-1/2 Gaussian collection success: 10/17 and 10/16.
- Iteration-2 valid CFM actions: 4,922/5,120 (96.13%).
- Fully valid chunks with observed terminal labels: 225; audited chunks: 64.
- All ratios, losses, gradients, and norm ratios were finite and nonzero.

## Gate Results

| Metric | Batch 0 | Batch 1 | Gate |
|---|---:|---:|---:|
| Pre candidate/control gradient cosine | 1.000000 | 1.000000 | >= 0.999 |
| Pre candidate/control norm ratio | 1.0000003 | 1.0000002 | 0.99 to 1.01 |
| Control post/pre gradient cosine | 0.9999986 | 0.9999985 | diagnostic |
| Candidate post/pre gradient cosine | 0.9999996 | 0.9999998 | >= 0.75 |
| Candidate cosine gain | 0.0000010 | 0.0000013 | >= 0.10 |
| Control positive active fraction | 1.000 | 1.000 | diagnostic |
| Candidate positive active fraction | 1.000 | 1.000 | control +0.10 |
| Candidate/control post norm ratio | 1.00179 | 0.99855 | 0.70 to 1.30 |

The equivalence and norm gates passed. Both direction-gain gates and both active-ratio-gain gates
failed by large margins.

## Interpretation

The official virtual step is too small to move any audited positive ratios beyond the clipping
boundary. Consequently, chunk-level and per-timestep clipping implement effectively the same update
at the point where H18 already found first-step instability. Increasing the virtual step until the
candidate becomes active would change the frozen question and revisit later-stage drift rather than
the first-step bottleneck. H33 is therefore stopped without tuning.
