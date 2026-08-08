# Protocol: Held-Out Ratio Early Stopping

## Hypothesis

The official ten-epoch FPO++ actor update overshoots its CFM trust region. Rolling back the first
epoch that leaves fewer than 80% of independently redrawn positive-advantage ratios below the PPO
upper clip preserves most of the surrogate improvement while retaining a materially more stable
policy-gradient direction.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260817; official scale-1 Gaussian source
- Data: iteration-2 fresh on-policy chunks after the critic-only first iteration
- Audit set: 64 positive-advantage, fully valid chunks selected once and split into two seeded,
  disjoint batches of 32
- Behavior anchor: one independent MC8 set of CFM times and noises per audited chunk, drawn before
  the actor update and fixed across epoch checks
- Update: the unchanged official ten actor epochs, minibatches, optimizer, clipping, and learning rate
- Per-epoch measurements: positive active-ratio fraction, advantage-weighted unclipped surrogate,
  and cosine of the current held-out gradient to its pre-update value
- Selector: retain the latest completed epoch for which the pooled held-out positive active-ratio
  fraction is at least 0.80; if an epoch violates the threshold, select the preceding snapshot
- No audit draw enters an actor optimizer step

The full ten-epoch trajectory is diagnostic only. An online implementation is allowed only after the
mechanism gates pass; it must snapshot actor and optimizer state before each epoch, roll back a
violating epoch, and stop the actor update while allowing the critic update to remain unchanged.

## Gates

The selected epoch must be between epochs 1 and 9. In each 32-chunk batch require:

1. selected-epoch held-out pre/post gradient cosine at least 0.85;
2. selected-epoch gradient cosine at least 0.10 higher than epoch 10;
3. selected-epoch held-out surrogate gain has the same positive sign as the epoch-10 gain and is at
   least 50% as large.

Stop without online integration if any gate fails. If all gates pass, implement the locked rollback
rule and run one matched five-update step-6000 screen. Do not tune the 0.80 threshold, audit size,
seed, or gates after observing the audit.
