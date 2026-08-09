# Protocol: H42 Square Checkpoint-Native Action Horizon

## Hypothesis

Restoring the short-finetuned Square policy from the official 16 executed actions to its pretrained
8-action execution horizon improves Gaussian-source success by incorporating visual feedback twice
as often, without materially degrading zero-source behavior.

## Locked Evaluation

- Checkpoint: frozen final non-EMA H41 control checkpoint
  `square_h41_control_osmesa_seed20260913/checkpoints/latest`.
- Conditions: official 16 executed actions versus an 8-action override. Policy weights, prediction
  horizon 16, Euler-10 integrator, and source distribution remain identical.
- Modes: zero source and independent standard-Gaussian source, evaluated separately.
- Seed: 20260915 shared across conditions and modes.
- Accounting: 20 environments, exactly one completed episode per environment, for 20 episodes per
  condition per mode.
- OSMesa is accepted for this paired screen only. No result is an official benchmark claim.

## Gates

Require finite 20-episode completion in every cell and all of:

1. 8-step random-source success exceeds 16-step control by at least 2/20;
2. 8-step zero-source success is no worse than control by more than 1/20;
3. 8-step pooled success exceeds control by at least 3/40.

If any gate fails, stop without trying 4 or 12 steps, changing seed, selecting another checkpoint,
or retraining at 8 steps. If all pass, register an independent 50-episode-per-mode seed before
confirmation; confirmation must precede any 8-step training protocol.
