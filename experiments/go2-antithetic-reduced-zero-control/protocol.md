# H72: Reduced-Step Zero-Source Control

## Question

Does equal-total-NFE antithetic projection provide reward value beyond merely
reducing Euler integration from 64 to 32 steps on the deterministic zero-source
policy?

## Fixed audit

- Reuse the immutable H70 seed-42 final checkpoint artifacts for `zero64`,
  `random64`, `antithetic32`, and `antithetic64`.
- Add exactly one method: `zero32`, Euler-32, 32 NFE/action.
- Use H70's 256 environments, evaluation seed `20261600`, and one episode per
  environment, requiring an exact initial-observation hash match.
- Bootstrap seed `20261612`, 20,000 paired resamples.
- No other source, solver, or step-count variant is screened.

## Decisions

H72 passes only if `antithetic32 - zero32` has a positive point estimate and a
strictly positive paired-bootstrap 95% lower bound. A pass supports reward value
from symmetric source projection beyond reduced integration depth. A failure
forbids claiming antithetic projection as the best latency-oriented deployment
on seed 42; H70's equal-NFE improvement over the official stochastic-source
baseline remains valid and separately reported.
