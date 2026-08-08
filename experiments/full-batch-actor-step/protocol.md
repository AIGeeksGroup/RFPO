# Protocol: Full-Batch Actor Step Audit

## Hypothesis

FPO++ gradient rotation is driven partly by applying Adam after each of eight shuffled minibatches.
Accumulating all eight minibatch gradients before one optimizer step per epoch will preserve the
held-out pre-update gradient direction while retaining useful surrogate improvement.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260822; official scale-1 Gaussian source
- Budget: 16 environments, 320 collection steps, and two iterations
- Control: `gradient_accumulation_steps=1`, using the completed H23 control artifact
- Candidate: `gradient_accumulation_steps=8`
- Unchanged settings: actor LR `1e-5`, ten update epochs, eight minibatches, critic objective and LR,
  GAE, clipping, MC8 CFM sampling, and all rollout settings
- Audit data: 64 seeded positive-advantage fully valid iteration-2 chunks, split into two batches of
  32; require at least 64 eligible chunks
- Held-out anchor: one independent MC8 set drawn before the actor update and fixed across epochs
- Measurements after every epoch: active positive-ratio fraction, gradient cosine to the pre-update
  held-out gradient, and held-out surrogate gain

Gradient accumulation changes both actor and critic optimizer frequency. The iteration-2 advantages
are computed before either optimizer is updated, so the candidate's iteration-2 held-out actor audit
isolates the actor update path. A later reward screen, if reached, would test the complete training
configuration rather than attribute its effect to the actor alone.

## Gates

At epoch 10, all gates must pass:

1. candidate pooled active positive-ratio fraction is at least 0.80;
2. in each batch, candidate gradient cosine is at least 0.60 and at least 0.20 higher than the H23
   control (`0.29089` and `0.27988`);
3. in each batch, candidate held-out surrogate gain is positive and at least 50% of the H23 control
   gain (`0.01569` and `0.02288`).

Stop without reward training if any gate fails. If all pass, run one matched five-update step-6000
reward screen before considering any larger experiment. Do not tune the accumulation count, seed,
epochs, audit set, or gates after observing the result.
