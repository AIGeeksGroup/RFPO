# Protocol: FPO++ KL-Adaptive Actor Learning Rate Audit

## Hypothesis

The released FPO++ actor moves into a poorly aligned held-out gradient during its ten-epoch update,
while a fourfold fixed LR reduction loses surrogate progress without preserving direction. Adapting
LR to the behavior-policy `x1_pred` divergence will retain the initial LR near-policy and reduce it
only after measured divergence, preserving more held-out gradient direction without discarding most
official surrogate progress.

## Locked Method

- Cache behavior-policy `x1_pred = x_t + (1-t) * velocity_pred` for every stored
  `(observation, action, t, epsilon)` CFM sample before the actor update.
- Before each actor optimizer step, recompute `x1_pred` and set `KL_proxy` to the elementwise mean
  squared difference from the cached prediction, matching the released motion-tracking code.
- Fix `desired_kl=1e-4`, the released FPO++ locomotion default.
- If `KL_proxy > 2e-4`, divide the current actor LR by `1.5`; if
  `0 < KL_proxy < 5e-5`, multiply it by `1.5`; otherwise leave it unchanged.
- Preserve the released locomotion bounds relative to its base LR: clamp manipulation LR to
  `[0.1 * 1e-5, 100 * 1e-5] = [1e-6, 1e-3]`.
- Initialize the candidate at exactly the control actor LR produced by the released warmup scheduler
  before iteration 2. Once adaptive control is active, do not overwrite it with the actor scheduler;
  critic scheduling remains unchanged.
- The feature is disabled by default. Do not add entropy regularization in H39.

## Locked Paired Audit

- Initialization: Can `95j3noe4_step_6000`, EMA weights, Huber CFM loss
- Conditions: official fixed-scheduler FPO++ control and KL-adaptive candidate
- Shared seeded rollout: `20260911`, official scale-1 Gaussian source
- Budget: 16 environments and two iterations: one critic-only warmup and one ten-epoch actor update
- Unchanged settings: MC8, ten epochs, eight minibatches, clipping `0.02`, GAE, AdamW, gradient
  clipping, critic updates, environment seeds, rollout settings, and CFM samples
- Audit set: 64 positive-advantage fully valid iteration-2 chunks split into two seeded disjoint
  batches of 32, with one independent fixed held-out MC8 set that never enters optimization
- Measurements after every epoch: held-out official-objective gradient cosine to pre-update,
  positive active-ratio fraction, unclipped surrogate gain, KL mean/max, LR min/max/final, and counts
  of increase/decrease/hold decisions
- Control and candidate pre-update held-out records and initial actor LR must match within `1e-6`;
  otherwise the pair is invalid.

## Gates

Require at least 64 eligible chunks, exactly 64 audited chunks, finite nonzero gradients, finite KL,
and LR within the locked bounds. At epoch 10, both held-out batches must satisfy:

1. candidate gradient cosine is at least `0.10` above control and at least `0.25`;
2. candidate positive active-ratio fraction is at least `0.05` above control;
3. candidate surrogate gain is positive and at least 50% of a positive control gain;
4. the controller records at least one increase and one divergence-triggered decrease, and every
   decrease occurs only for `KL_proxy > 2e-4`.

Stop without reward training if either batch fails any outcome gate or if the controller fails its
activity gate. Do not change target, factor, bounds, seed, epochs, audit set, or gates after observing
the pair. If all gates pass, separately register a matched five-update step-6000 balanced reward
screen before running it; no larger benchmark confirmation is authorized by this protocol.
