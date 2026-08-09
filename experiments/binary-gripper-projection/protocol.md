# H56 Protocol: Binary Gripper-Support Projection

## Hypothesis

Projecting the released Can policy's continuous gripper output onto the demonstrated two-state
controller support improves deterministic success without altering its six-dimensional arm action.

## Locked Intervention

- Checkpoint: official released Can step-1000 EMA policy.
- Source and solver: zero source, 10 Euler integration steps.
- Prediction and execution horizons: 16 and 8 actions.
- Candidate: after selecting each action, replace only final coordinate `g` with `+1` when `g >= 0`
  and `-1` otherwise.
- Control: unchanged official action. Reuse H55 phase `phi=8`, which used the identical checkpoint,
  source, solver, cadence, renderer, and environment seeds and scored 15/20.
- No clipping, smoothing, hysteresis, confidence threshold, extra prediction, source change, or
  checkpoint update.

## Stage A: Implementation Smoke

Run seeds `20261010` and `20261011`. Require finite actions, projected gripper values exactly in
`{-1, +1}`, bitwise-exact preservation of coordinates 0-5, at least one materially changed gripper
command, two completed finite episodes, and exact initial normalized-observation hashes matching the
archived H55 control for those seeds.

## Stage B: Common-Seed Reward Screen

- renderer: OSMesa method screen, not official EGL or real robot
- environment seeds: `20261010` through `20261029`
- one completed episode per environment, 20 episodes total
- video and W&B: disabled

Require every Stage A invariant across the full candidate run and exact initial normalized-observation
hashes matching H55. H56 passes screening only if the candidate scores at least 17/20, a gain of
`+2/20` over the locked 15/20 control. If it passes, freeze a separate independent-seed confirmation;
otherwise refute H56 without threshold, hysteresis, hold-time, source, seed, or training variants.
