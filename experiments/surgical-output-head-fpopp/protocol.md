# Protocol: Surgical Output-Head FPO++ Audit

## Hypothesis

Updating only the final linear velocity head of the pretrained Square flow actor will retain useful
FPO++ surrogate improvement while reducing full-model feature distortion, held-out gradient rotation,
and deterministic endpoint drift.

## Locked Method

- Add `actor_trainable_scope` with values `all` (released behavior) and `output_head` (candidate).
- For `output_head`, require the MLP flow architecture and select exactly the final `nn.Linear` in
  `actor.model.mlp`; freeze every other actor parameter before optimizer construction.
- Keep the vision encoder frozen in both conditions. The optimizer receives only parameters with
  `requires_grad=True`; learning rate, AdamW settings, FPO++ objective, MC draws, minibatch order,
  gradient clipping, critic, and all rollout settings remain unchanged.
- Record names/counts of trainable and frozen parameters, their fractions, optimizer membership,
  and maximum absolute displacement. Every frozen actor parameter must remain bitwise unchanged.
- Do not test bias-only, last-two-layer, LoRA, another learning rate, another checkpoint, or another
  seed after observing this audit.

## Locked Paired Audit

- Initialization: released Square checkpoint `trc7rbt0_step_110000`, EMA weights and official FPO++
  settings.
- Conditions: official all-nonvisual-layer control and output-head-only candidate.
- Shared rollout/training seed: `20260981`.
- Budget: 16 environments, 320 collection steps, two iterations, with the first iteration retaining
  the official critic-only warmup and the second performing ten actor epochs.
- Audit data: 64 positive-advantage fully valid iteration-2 chunks, selected deterministically and
  split into two disjoint batches of 32. A single independent fixed MC8 draw is held out from every
  optimizer step.
- For each held-out batch, record the epoch-10 official-objective gradient cosine to its own
  pre-update gradient and unclipped advantage-weighted surrogate gain.
- On the same held-out observations, fix a zero source and integrate the released Euler sampler
  before and after training. Endpoint drift is mean squared normalized-action displacement.
- The paired runs must have matching pre-update audit records and identical collection fingerprints
  through the controlled rollout; otherwise the comparison is invalid.

## Gates

Require exactly 64 audited chunks, finite nonzero losses and gradients, the exact expected output-head
selection, no frozen-parameter displacement, and optimizer membership equal to the trainable set.
At epoch 10, both held-out batches must satisfy all of the following:

1. candidate surrogate gain is positive and at least 50% of a positive control gain;
2. candidate gradient cosine is at least `0.10` above control;
3. candidate zero-source endpoint drift is at most 50% of control drift.

If any validity or outcome gate fails, refute H52 without variants or reward training. If all gates
pass, separately commit a five-iteration matched Square OSMesa reward-screen protocol. A positive
screen requires an independent confirmation before any healthy-EGL official benchmark; OSMesa is
only a paired method screen and is never reported as the official benchmark.
