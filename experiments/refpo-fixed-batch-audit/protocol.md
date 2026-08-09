# Protocol: ReFPO Fixed-Batch Regularization Audit

## Hypothesis

On the first genuine step-6000 Can FPO++ actor update, adding ReFPO's unweighted current CFM loss
reduces independently redrawn fixed-batch proxy-ratio drift while retaining useful reward-weighted
surrogate progress and not worsening outcome-gradient preservation.

This does not reopen H2's refuted claim that offline reflow aligns gradients across independent CFM
draws. It tests the subsequently published ReFPO objective's direct fixed-batch regularization claim.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss.
- Can rollout seed: `20261061`; official unit-Gaussian source.
- Budget: 16 environments, 320 collection steps, and two iterations. Iteration 1 is the unchanged
  critic-only warmup; iteration 2 contains the first actor update.
- Control: official FPO++ (`lambda=0`).
- Candidate: `L_FPO + 0.04 * mean(current valid CFM loss)`, using exactly the already-computed
  current losses and stored time-noise pairs. No extra actor forward is allowed.
- Unchanged settings: actor and critic learning rates, ten epochs, eight minibatches, MC8 sampling,
  clipping, optimizer, GAE, source distribution, and rollout order.
- Audit data: the same 64 seeded positive-advantage, fully valid chunks in both conditions, split into
  two disjoint batches of 32. One independent MC8 draw is fixed before either actor update.
- Record at epoch 10: held-out median absolute log ratio, ratio standard deviation, clip fraction,
  positive advantage-weighted unclipped surrogate gain, and pre/post outcome-gradient cosine.
- Pairing gate: pre-update observations, actions, advantages, behavior losses, held-out CFM draws,
  ratios, gradients, and surrogate values must match exactly between conditions.
- Also record the pre-update norms and cosine of the FPO and unweighted-CFM gradients to expose the
  effective regularization scale; this is diagnostic and cannot be used to retune `lambda`.

## Gates

All quantities must be finite and all audited gradients nonzero. In each held-out batch at epoch 10:

1. candidate median absolute log ratio and ratio standard deviation are each at most 80% of control;
2. candidate clip fraction is at least 10 percentage points below control;
3. candidate surrogate gain is positive and at least 50% of a positive control gain;
4. candidate outcome-gradient cosine is no lower than control minus `0.02`.

Every gate must pass in both batches. If any gate fails, refute H61 and do not tune `lambda`, seed,
epochs, or audit selection. If all pass, run only a matched 2-to-5-update reward screen before any
larger Go2 or manipulation training. Lower ratio movement without retained surrogate progress is
regularization-induced inertia, not support for ReFPO.
