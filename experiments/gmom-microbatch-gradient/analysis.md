# H37 Analysis: Geometric Median-of-Means Gradient

## Outcome

H37 is refuted at the locked two-replica mechanism audit. The full-vector geometric median converged
and was nearly identical to the arithmetic mean in direction and norm, but outcome-reference
alignment decreased in both replicas. No optimizer integration, online reward training, block-count
change, gradient normalization, solver variant, or additional seed is authorized.

## Audit Health

- Seed: `20260908`; aggregation method recorded as `geometric`.
- Iteration-1/2 Gaussian rollout success: 6/16 and 5/16.
- Iteration-2 valid CFM actions: 5,018/5,120 (98.01%).
- Fully valid terminal-labeled chunks: 250; audited chunks: 192.
- Each replica contained 96 chunks split into four ordered 24-chunk blocks.
- GMOM converged in 12 and 11 Weiszfeld iterations; all gradients were finite and nonzero.

## Gate Results

| Metric | Replica 0 | Replica 1 | Gate |
|---|---:|---:|---:|
| Control/outcome cosine | 0.74076 | 0.68823 | diagnostic |
| GMOM/outcome cosine | 0.73632 | 0.67840 | >= 0.75 |
| GMOM cosine gain | -0.00444 | -0.00983 | >= +0.10 |
| GMOM/control cosine | approximately 1.0 | approximately 1.0 | >= 0.75 |
| GMOM/control norm ratio | 1.00313 | 1.00254 | 0.50 to 1.20 |
| Median block/outcome cosine | 0.32512 | 0.40492 | candidate not below |

Finite convergence, control-direction, norm, and typical-block gates passed. Absolute outcome
alignment and the required gain failed in both replicas.

## Numerical Note

The raw JSON reports GMOM/control cosine as 1.00037 and 1.00110. Investigation reproduced this as
float32 reduction error on a 13.37-million-element near-collinear vector: a synthetic vector of the
same scale gave 1.00072 in float32 and 0.999999995 in float64. This does not affect the H37 decision,
whose primary cosine gains are negative and roughly two orders of magnitude below the +0.10 gate.
Future audit cosine reductions now use float64; H37 is not rerun or reselected.

## Interpretation

Unlike H34's coordinate median, GMOM preserves complete-vector structure, but the four block
gradients do not contain a robust central direction superior to their mean. At this sample scale the
geometric median changes the mean by too little to improve its relationship with observed outcomes.
Robust microbatch aggregation is therefore closed rather than tuned further.

