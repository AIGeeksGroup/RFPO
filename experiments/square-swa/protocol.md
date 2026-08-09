# Protocol: H44 Square Actor-Trajectory SWA

## Hypothesis

Uniformly averaging all actor-update checkpoints from one short official Square FPO++ trajectory
reduces update-path variance and improves Gaussian-source success without degrading zero-source
behavior relative to the final checkpoint.

## Locked Checkpoints

- Source run: frozen H43 official persistent-Adam control
  `square_h43_control_osmesa_seed20260916`.
- Control: final non-EMA `checkpoints/step_25600`.
- Candidate: tensorwise float32 arithmetic mean of exactly
  `step_10240`, `step_15360`, `step_20480`, and `step_25600`, cast back to original dtypes.
- All floating policy tensors must be averaged. Non-floating tensors must be identical and copied.
- Policy configs must be byte-identical. No critic, optimizer, or scheduler state is averaged.
- No checkpoint selection, unequal weights, BC/iteration-1 checkpoint, EMA decay, or retraining is
  allowed after evaluation.

## Locked Evaluation

- Environment: Square, Euler-10 integration, 16 executed actions.
- Conditions: frozen final control versus SWA candidate.
- Modes: zero source and independent standard-Gaussian source, evaluated separately.
- Seed: 20260919 shared across all cells.
- Accounting: 20 environments, exactly one completed episode each, for 20 episodes per condition per
  mode.
- OSMesa is accepted for this paired screen only; it is not an official benchmark claim.

## Gates

Require finite checkpoint generation and 20 completed episodes in every cell, plus all of:

1. SWA random-source success exceeds final control by at least 2/20;
2. SWA zero-source success is no worse than final control by more than 1/20;
3. SWA pooled success exceeds final control by at least 3/40.

Failure closes actor-trajectory averaging without changing the window, adding the critic-only
checkpoint, unequal weighting, another seed, or EMA retraining. Success requires a separately
registered independent 50-episode-per-mode confirmation before any broader claim.
