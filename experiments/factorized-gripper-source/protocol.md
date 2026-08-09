# H57 Protocol: Factorized Gaussian Arm / Zero Gripper Source

## Hypothesis

Removing Gaussian noise only from the gripper latent improves released Can step-1000 random-source
success while preserving the standard Gaussian exploration of all six arm coordinates.

## Locked Intervention

- Checkpoint: official released Can step-1000 EMA policy.
- Solver: 10 Euler integration steps.
- Prediction and execution horizons: 16 and 8 actions.
- Control: standard independent `N(0, 1)` source in all seven action coordinates.
- Candidate: the identical source tensor with coordinate 6 set to zero at all 16 horizon positions.
- Common source: stateless CPU float32 Gaussian generation keyed by fixed source seed, environment
  index, and episode-local replan index; control and candidate arm coordinates must hash identically.
- No output projection, source scaling, temporal correlation, extra samples, changed cadence,
  checkpoint update, or training.

## Stage A: Paired Smoke

- environment seeds: `20261031`, `20261032`
- source base seed: `20261101`
- one completed episode per environment and condition

Require exact initial normalized-observation hashes, exact common arm-source hashes for every shared
`(environment, replan)` key, candidate gripper sources exactly zero, finite episodes/actions, and at
least one action difference between conditions. This is implementation validation only.

## Stage B: Paired Reward Screen

- renderer: OSMesa method screen, not official EGL or real robot
- environment seeds: `20261031` through `20261050`
- source base seed: `20261101`
- condition order: control, candidate
- one completed episode per environment, 20 episodes per condition
- video and W&B: disabled

The control must score between 1/20 and 8/20 to validate the expected non-saturated random-source
regime. All Stage A source and state invariants must hold. H57 passes only if candidate success is at
least control `+3/20`. Passing authorizes a separately frozen 50-episode independent-seed
confirmation; failure closes coordinate-zeroing, gripper variance, mixture, and state-conditioned
source neighbors without additional seeds or training.
