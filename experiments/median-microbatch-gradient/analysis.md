# H34 Analysis: Coordinate-Wise Median Microbatch Gradient

## Outcome

H34 is refuted at the fixed two-replica mechanism audit. Coordinate-wise median aggregation retained
the broad control direction and gradient scale, but reduced alignment to the observed terminal-return
gradient in both replicas. No optimizer integration, microbatch-count variant, trimming rule, or sign
vote is authorized.

## Audit Health

- Iteration-1/2 Gaussian collection success: 10/16 and 12/16.
- Iteration-2 valid CFM actions: 4,934/5,120 (96.37%).
- Fully valid chunks with observed terminal labels: 207; audited chunks: 192.
- Each replica contained 96 chunks split into four ordered 24-chunk microbatches.
- All microbatch, control, candidate, and reference gradients were finite and nonzero.

## Gate Results

| Metric | Replica 0 | Replica 1 | Gate |
|---|---:|---:|---:|
| Control/outcome cosine | 0.83334 | 0.68845 | diagnostic |
| Candidate/outcome cosine | 0.69561 | 0.61857 | >= 0.75 |
| Candidate cosine gain | -0.13773 | -0.06988 | >= +0.10 |
| Candidate/control cosine | 0.88532 | 0.91249 | >= 0.75 |
| Candidate/control norm ratio | 0.98231 | 1.05762 | 0.50 to 1.20 |
| Median constituent/outcome cosine | 0.41582 | 0.34318 | candidate not below |

The finite-gradient, control-direction, norm-retention, and typical-constituent gates passed. The
absolute outcome-alignment gate failed twice, and the required improvement gate failed with negative
gains in both replicas.

## Interpretation

The mean gradient's microbatch-specific coordinates are not merely harmful outliers. Replacing them
coordinate by coordinate destroys task-relevant joint structure: the candidate remains close to the
mean in aggregate, yet is consistently less aligned with terminal outcomes. This is direct negative
evidence against robust coordinate aggregation at this scale, so H34 stops before online training.
