# H38 Equal-NFE Midpoint Audit

## Decision

H38 passes every locked geometry gate. Five explicit-midpoint steps give a substantially more
accurate approximation to the source-matched Euler-64 action endpoint than ten Euler steps at the
same measured ten velocity-network evaluations. This supports the numerical-integration mechanism
and authorizes only the separately preregistered reward screen; it is not yet benchmark-improvement
evidence.

## Results

The audit used the released Can step-1000 EMA checkpoint, 128 fixed observations, seed `20260909`,
and matched Gaussian and all-zero sources.

| Gate | Control Euler-10 | Candidate midpoint-5 | Decision |
|---|---:|---:|---|
| Gaussian endpoint MSE | `9.92959e-5` | `1.44625e-6` (`1.4565%`) | pass |
| Zero-source endpoint MSE | `6.49063e-5` | `7.29372e-7` (`1.1237%`) | pass |
| Mean element std | `0.261348` | `0.262258` (`100.348%`) | pass |
| Mean pairwise distance | `5.347481` | `5.360935` (`100.252%`) | pass |
| Measured velocity NFE | `10` | `10` | pass |
| Finite actions | yes | yes | pass |

Relative to Euler-10, midpoint-5 reduced endpoint MSE by `98.54%` for Gaussian sources and `98.88%`
for zero sources. The magnitude is consistent across both source regimes and is not explained by
diversity collapse: both fixed-observation diversity statistics increased by less than `0.4%`.

## Scope

The reference is Euler-64 rather than a ground-truth ODE solution, so this audit establishes much
better agreement with the repository's high-NFE policy endpoint, not absolute solver error. The
candidate changes intermediate states even at equal NFE and may therefore change closed-loop reward
in either direction. No solver, step-count, schedule, or seed variant was tested.
