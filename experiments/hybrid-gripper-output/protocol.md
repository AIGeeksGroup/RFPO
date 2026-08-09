# H58 Protocol: Random Arm / Deterministic Gripper Output

## Hypothesis

Using the zero-source gripper trajectory while preserving the standard Gaussian-source arm output
improves released Can step-1000 random-source success.

## Locked Intervention

- Checkpoint: official released Can step-1000 EMA policy.
- Solver: two 10-step Euler solves per candidate replan, one standard Gaussian and one zero source.
- Prediction and execution horizons: 16 and 8 actions.
- Candidate output: coordinates 0-5 exactly from the Gaussian-source prediction; coordinate 6
  exactly from the zero-source prediction at the same observation.
- Control: archived H57 full-Gaussian condition, using source seed `20261101`, environment seeds
  `20261031..20261050`, and scoring 1/20.
- No averaging, binarization, source modification, confidence rule, changed cadence, checkpoint
  update, or training.

## Stage A: Paired Smoke

Run candidate seeds `20261031`, `20261032` and compare with the archived H57 smoke control. Require
exact initial observation hashes, exact common Gaussian arm-source hashes, bitwise equality between
hybrid and Gaussian arm outputs at every candidate replan, exact equality between hybrid and
zero-source gripper outputs, finite actions/episodes, and a nonzero first-action change.

## Stage B: Reward Screen

Run one candidate episode for each seed `20261031..20261050` using the same stateless Gaussian source
keys as H57. The archived 1/20 control must reproduce all comparison metadata and all Stage A
invariants must hold. H58 passes only at 4/20 or better, at least +3/20 over control. Passing
authorizes a separately frozen 50-episode independent confirmation; failure closes deterministic
gripper substitution, averaging, agreement selection, and a separate gripper head without retries.

This is an OSMesa method screen, not an official EGL or real-robot benchmark.
