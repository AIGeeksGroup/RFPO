# Protocol: AdamW Extragradient Actor Audit

## Hypothesis

An exact one-step AdamW extragradient correction improves the alignment of the first FPO++ actor
update with observed terminal outcomes while retaining useful held-out surrogate progress.

## Locked Audit

- Initialization: exact released Can `95j3noe4_step_6000` EMA actor.
- Renderer and seed: isolated OSMesa, seed `20261050`.
- Collection: 16 environments and two 320-step iterations. Iteration 1 remains critic-only;
  iteration 2 supplies the fresh Gaussian rollout for the audit.
- Select 64 seeded, fully valid iteration-2 chunks with finite normalized GAE and observed terminal
  returns; split them into two fixed 32-chunk batches.
- Training estimator: reuse each chunk's exact stored MC8 CFM times, noises, behavior losses, and
  normalized GAE weights.
- Outcome reference: use an independent fixed MC8 draw on the same chunks, weighted by centered
  observed terminal returns. It never enters an optimizer step.
- Control: one exact actor AdamW step from the common base state using the official stored-draw
  FPO++ gradient.
- Candidate: take that control step virtually, recompute the official gradient at the lookahead on
  the same stored draws, restore actor parameters and optimizer state bitwise, then take one exact
  AdamW step from the common base state using the lookahead gradient.
- Keep actor LR `1e-5`, clipping, gradient clipping, weight decay, GAE, MC8, and every policy setting
  unchanged. Exclude critic parameters and updates from the virtual comparison.

## Validity

Both batches must have nonzero finite official, lookahead, and outcome-reference gradients. The
control and candidate must start from bitwise-identical parameters and optimizer states. Restoration
after the lookahead must be bitwise exact, the control path must match an ordinary AdamW step, and
the candidate must produce a nonzero parameter displacement.

## Gates

For each 32-chunk batch, all gates must pass:

1. candidate update-direction cosine to the independent outcome-reference gradient is at least
   `0.05` higher than control;
2. candidate post-update outcome-gradient cosine to its pre-update reference is at least `0.05`
   higher than control;
3. candidate held-out GAE surrogate gain is positive and at least 80% of the positive control gain;
4. candidate/control parameter-displacement norm ratio lies in `[0.5, 1.5]`.

Failure stops H59 without online integration, reward training, lookahead-step scaling, extra
lookaheads, seed changes, or optimizer variants. Passing all gates authorizes implementation behind
an opt-in flag and one paired five-iteration Square reward screen, followed by an independent
confirmation only if the preregistered reward gate passes. OSMesa can screen the method but cannot
support an official benchmark claim.
