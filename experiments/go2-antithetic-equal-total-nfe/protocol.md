# H70: Antithetic Pairing at Equal Total NFE

## Question

Can antithetic source projection retain its Go2 return gain without H66's 2x
inference cost by halving the integration steps per endpoint?

## Fixed screen

- Checkpoint: official Go2 seed-42 `model_1499.pt`
- 256 environments per method, one episode per environment
- Evaluation seed: `20261600`
- Primary source seed: `20261601`
- Bootstrap seed: `20261602`, 20,000 paired resamples
- Methods:
  - `zero64`: zero source, Euler-64, 64 NFE/action
  - `random64`: one Gaussian source, Euler-64, 64 NFE/action
  - `antithetic32`: two symmetric endpoints, Euler-32 each, 64 total NFE/action
  - `antithetic64`: two symmetric endpoints, Euler-64 each, 128 total NFE/action

All four methods must share exact initial-observation hashes. The two stochastic
controls/candidates must share the complete primary source stream. No source
scale, solver, interpolation weight, or additional step count is screened.

## Decisions

The efficiency hypothesis passes if paired `antithetic32 - random64` has a
positive point estimate and bootstrap 95% lower bound. `Antithetic32 - zero64`
is a separate deployment comparison: a positive lower bound supports a same-NFE
best-deployment claim; otherwise only the stochastic-source efficiency claim is
retained.

`Antithetic32 - antithetic64` quantifies return retained after halving steps and
is descriptive, not a gate. A failed primary gate closes reduced-step
antithetic inference without trying 16/24/40/48 steps, midpoint integration, or
source scaling. A passed primary gate authorizes independent-seed confirmation
after H69 completes.

