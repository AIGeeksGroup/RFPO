# H73: Independent-Policy Equal-Total-NFE Confirmation

## Question

Does H70's equal-total-NFE stochastic-source recovery persist across the three
independently trained H69 Go2 policies?

## Fixed confirmation

- Fixed H69 final checkpoints from training seeds 43, 44, and 45.
- Reuse each seed's immutable H69 `random64` and `zero64` artifacts.
- Add two 512-environment cells per training seed with the exact H69 evaluation
  and source seeds:
  - `antithetic32`: sequential `z/-z` Euler-32 endpoints, 64 total NFE/action
  - `zero32`: zero-source Euler-32, 32 NFE/action
- One episode per environment; require exact initial-observation hashes and an
  exact primary-source-stream match between `antithetic32` and archived
  `random64` within each training seed.
- Stratified bootstrap seed `20261620`, 20,000 resamples.
- No batched execution, additional step count, or source scale is screened.

## Decisions

The primary hypothesis passes if `antithetic32 - random64` is positive in all
three training seeds and its stratified bootstrap 95% lower bound is positive.
This supports an independent-policy equal-total-NFE stochastic-source claim.

`Antithetic32 - zero32` is a separate method-specific deployment gate and
`zero32 - zero64` checks reduced-step deterministic retention. Failure of the
primary gate stops equal-total-NFE cross-task expansion. Passing the primary
gate authorizes a medium official Spot task screen while retaining all zero
comparisons regardless of outcome.
