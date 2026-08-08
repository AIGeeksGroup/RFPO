# Protocol: Four-Step Receding-Horizon Execution

## Hypothesis

The released Can policy predicts a 16-action horizon and executes eight actions before observing and
replanning. Executing only the first four actions should reduce open-loop grasp and placement error
by incorporating new visual feedback twice as often. This changes neither checkpoint weights nor
the flow source, solver, or training objective.

## Locked Intervention

- Control: official `n_action_steps=8`.
- Candidate: override `n_action_steps=4` after loading the checkpoint.
- Both policies retain the 16-action prediction horizon, EMA weights, zero source, and 10 Euler
  integration steps.
- No action-step sweep is allowed after observing the result. In particular, 1, 2, and 6 executed
  steps are outside H27.
- Record rollout FPS and reject non-finite actions.

## Fixed-Seed Screen

- Checkpoint: official released Can step-1000 EMA policy
- Environment: Can
- Seed: 20260826
- Conditions: 8 and 4 executed actions per predicted chunk
- Budget: 20 episodes per condition, 16 environments, OSMesa rendering
- Evaluation order: 8-step control, then 4-step candidate
- Video and W&B logging: disabled

H27 passes only if the four-step candidate produces finite actions and exceeds control by at least
2/20 successes, or 10 percentage points. If either gate fails, stop without testing another action
count or seed. If both gates pass, run the same two conditions for 50 episodes at seed 20260827 and
require at least a 5-point candidate gain before considering a 200-episode evaluation.

OSMesa is acceptable for screening. A formal benchmark claim still requires healthy EGL or an
independently approved rendering GPU.
