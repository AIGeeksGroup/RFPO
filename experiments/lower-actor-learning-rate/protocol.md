# Protocol: Lower Actor Learning Rate Drift Audit

## Hypothesis

The official GAE gradient is aligned with observed outcomes before optimization, but the actor moves
too far under the released `1e-5` learning rate. A fixed fourfold lower actor learning rate will
retain useful surrogate improvement while preserving held-out gradient direction through all ten
optimizer epochs.

## Locked Paired Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260822; official scale-1 Gaussian source
- Budget per condition: 16 environments, 320 collection steps, and two iterations. The original
  8-environment audit was rejected before candidate-gradient computation because it produced only
  54 positive valid chunks against the locked requirement of 64.
- Control actor learning rate: `1e-5`
- Candidate actor learning rate: `2.5e-6`
- Unchanged settings: ten update epochs, critic learning rate and updates, GAE, optimizer, clipping,
  minibatches, MC8 CFM sampling, and all rollout settings
- Audit data: 64 seeded positive-advantage fully valid iteration-2 chunks in each condition, split
  into two batches of 32. Separate asynchronous MuJoCo processes are not assumed to have identical
  episode boundaries even with the same seed.
- Held-out anchor: one independent MC8 set drawn before the actor update and fixed across epochs
- Measurements after every epoch: active positive-ratio fraction, gradient cosine to the pre-update
  held-out gradient, and held-out surrogate gain
- The audit tensors never enter an optimizer step

## Gates

Require at least 64 eligible chunks and exactly 64 audited chunks in each condition, plus finite
nonzero gradients. At epoch 10, all gates must pass:

1. candidate pooled active positive-ratio fraction is at least 0.80;
2. in each batch, candidate gradient cosine is at least 0.60 and at least 0.20 higher than control;
3. in each batch, candidate held-out surrogate gain is positive and at least 50% of the control
   epoch-10 gain when that control gain is positive.

Stop without reward training if any gate fails. If all pass, expose the candidate learning rate as
an opt-in configuration and run one matched five-update step-6000 reward screen. Do not tune the
learning rate, epochs, seed, threshold, audit set, or gates after observing the pair.
