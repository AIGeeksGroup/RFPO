# H36 Paired Audit Analysis

## Validity

The seeded control and rollback runs used the locked seed `20260907` and produced the same two
rollouts: iteration-1 success was 9/16, iteration-2 success was 5/17, and iteration-2 had 85
positive chunks. Every scalar in the two pre-update held-out batch records is exactly equal in the
saved JSON files, so the pair satisfies the `1e-6` matching requirement.

## Epoch-10 Gates

| Batch | Metric | Control | Rollback | Required | Result |
|---:|---|---:|---:|---:|---|
| 0 | gradient cosine | 0.453628 | 0.430540 | candidate >= 0.553628 and >= 0.25 | fail |
| 1 | gradient cosine | 0.463311 | 0.570661 | candidate >= 0.563311 and >= 0.25 | pass |
| 0 | active positive ratio fraction | 0.257812 | 0.441406 | candidate >= 0.307812 | pass |
| 1 | active positive ratio fraction | 0.246094 | 0.398438 | candidate >= 0.296094 | pass |
| 0 | unclipped surrogate gain | 0.020787 | 0.007479 | positive and >= 0.010394 | fail |
| 1 | unclipped surrogate gain | 0.038224 | 0.011117 | positive and >= 0.019112 | fail |

Rollback was active on 17,004/25,600 actor-loss elements (`66.421875%`), within the locked
5-80% interval. Ratios and gradients were finite and both epoch-10 gradient norms were nonzero.

## Decision

H36 is refuted. PPO-RB retained 18.36 and 15.23 percentage points more active positive ratios, but
its gradient-direction effect was inconsistent and its two surrogate gains retained only 35.98% and
29.08% of control. Because all gates were conjunctive, no reward training is authorized. Stop this
hypothesis without changing `alpha`, clipping, seed, audit batches, or update budget.

The unseeded first pair remains exploratory and invalid and is not included in this decision.
