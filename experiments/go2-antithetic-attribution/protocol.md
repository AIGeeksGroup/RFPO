# H67: Equal-Compute Attribution of Antithetic Source Ensembling

## Question

Does H66 improve Go2 return because the two flow sources are exact negatives, or
would any two-endpoint average obtain the same gain at 128 network evaluations
per action? Does H66 beat the matched deterministic zero-source policy, or only
recover toward it from the weaker random-source policy?

## Fixed setting

- Task: `Isaac-Velocity-Flat-Unitree-Go2-v0`
- Checkpoint: official seed-42 `model_1499.pt`
- Integrator: Euler, 64 steps per endpoint
- Evaluation seed: `20261110`
- Primary source seed: `20261111`
- Independent secondary source seed: `20261112`
- Screen size: 256 environments per method, exactly one episode per environment
- Methods:
  - `zero`: `F(o, 0)`, 64 NFE/action
  - `random`: `F(o, z1)`, 64 NFE/action
  - `iid_pair`: `0.5 * (F(o, z1) + F(o, z2))`, 128 NFE/action
  - `antithetic`: `0.5 * (F(o, z1) + F(o, -z1))`, 128 NFE/action

The primary source stream must be identical for `random`, `iid_pair`, and
`antithetic`. The secondary IID stream must be generated from its separately
fixed seed. Initial-observation hashes must match across all four methods.

## Predictions and decisions

The mechanistic prediction is that exact source symmetry cancels substantially
more source-dependent action displacement than IID averaging. On fixed states,
the antithetic endpoint mean should be closer to `F(o, 0)` than the IID endpoint
mean.

The primary reward comparison is paired `antithetic - iid_pair`, since both use
128 NFE/action. H67 passes the screen only if this point estimate is positive and
its paired bootstrap 95% lower bound is above zero. A positive estimate whose
interval crosses zero is suggestive and may receive one larger confirmation,
but it is not evidence of an antithetic advantage. A non-positive estimate
closes H67 without extra pair counts, source scales, or seed retries.

The matched `antithetic - zero` comparison is reported regardless of outcome.
If antithetic does not beat zero, the method is described only as improving the
official stochastic-source mode; it is not claimed as a compute-efficient best
deployment policy.

Only after the equal-compute attribution is positive may the study proceed to
multiple checkpoints, independent training seeds, or additional locomotion
tasks. No official-scale 4096-environment run is authorized by this protocol.

