# H68: Cross-Checkpoint Antithetic Attribution

## Question

Is H67's equal-compute antithetic advantage present at earlier points of the
same official Go2 training trajectory, or is it isolated to the final model?
This is a low-cost stage-generalization screen, not evidence across independent
training seeds.

## Fixed setting

- Source run: official Go2 seed-42 trajectory
- New checkpoints: iterations 500 and 1000
- Archived checkpoint: iteration 1499 from H67
- Methods per new checkpoint: `zero`, `iid_pair`, and `antithetic`
- Integrator: Euler-64 per endpoint
- Environments: 128 per checkpoint and method, one episode per environment
- Checkpoint 500 seeds: evaluation `20261120`, primary source `20261130`,
  secondary source `20261140`
- Checkpoint 1000 seeds: evaluation `20261121`, primary source `20261131`,
  secondary source `20261141`
- Pooled bootstrap seed: `20261150`, 20,000 paired resamples

For each checkpoint, all methods must have identical initial-observation hashes,
and IID/antithetic must have identical complete primary source-stream hashes and
exactly 128 NFE/action. The IID secondary stream must be independently seeded.

## Decision rule

For both checkpoints, the paired `antithetic - iid_pair` point estimate must be
positive. Pooling the 256 paired differences across checkpoints, the bootstrap
95% lower bound must also be above zero. Failure stops cross-checkpoint expansion
without adding checkpoints or retrying seeds. Passing authorizes official Go2
training for independent seeds.

`Antithetic - zero` is reported separately at every checkpoint but is not a gate
for this attribution hypothesis. The iteration-1499 H67 result is included only
in the descriptive cross-stage table and is not reused in the locked pooled
test.

