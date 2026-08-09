# Protocol: H43 Square Rollout-Local Actor Adam

## Hypothesis

Resetting actor AdamW moments at each new on-policy rollout prevents stale cross-rollout gradient
statistics from distorting FPO++ updates and improves Square random-source success without degrading
zero-source behavior.

## Locked Training

- Base policy: released Square checkpoint `trc7rbt0_step_110000`, loading EMA weights exactly as in
  the official finetuning command.
- Conditions: official persistent actor AdamW control versus rollout-local actor AdamW candidate.
- Candidate intervention: immediately before actor epochs in iterations 2-5, clear only
  `optimizer_actor.state`. Iteration 2 must observe an empty state; iterations 3-5 must each observe a
  nonempty state before clearing it and an empty state immediately afterward.
- The actor LR scheduler, parameter groups, configured LR, critic optimizer and scheduler, model
  weights, GAE, CFM draws, rollout source, and all losses remain unchanged.
- Seed: 20260916 shared across conditions.
- Budget: 16 environments, 320 collection steps, exactly five actual iterations (25,600 total
  timesteps): one critic-only iteration followed by four actor-update iterations.
- Official Square settings: discount 0.995, GAE 0.99, Huber CFM delta 1, CFM and PPO clip 0.01,
  max grad norm 25, no log-ratio or old-loss clamp, PPO trust region, actor LR `1e-5`, MC8,
  Euler-10 sampling, and 16 executed actions.
- Validity: both runs must finish finite. Iteration-1 and iteration-2 collection records must match
  exactly because the first candidate clear is vacuous.

## Locked Evaluation

- Frozen final non-EMA checkpoints from the paired runs; no checkpoint selection.
- Seed 20260917 shared across conditions and modes.
- Zero and independent standard-Gaussian source evaluated separately.
- 20 environments, exactly one completed episode each, for 20 episodes per cell.
- OSMesa is accepted for this paired screen only; it is not an official benchmark claim.

## Gates

Require finite training and evaluation, the locked reset-state observations, and all of:

1. candidate random-source success exceeds control by at least 2/20;
2. candidate zero-source success is no worse than control by more than 1/20;
3. candidate pooled success exceeds control by at least 3/40.

Failure closes rollout-local Adam without resetting only one moment, changing betas, another seed,
longer training, checkpoint selection, or related optimizer-reset variants. Success requires a
separately registered independent 50-episode-per-mode confirmation before larger training.
