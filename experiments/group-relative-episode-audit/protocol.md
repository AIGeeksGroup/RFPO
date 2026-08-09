# Protocol: Group-Relative Complete-Episode Signal Audit

## Hypothesis

Four complete Gaussian-source Square trajectories from the same simulator initialization produce a
dense group-relative terminal-success signal suitable for critic-free FPO++ weighting. Common-scene
leave-one-out baselines should remove scene difficulty while retaining within-scene policy-outcome
variation that H62's local binary vines lacked.

## Locked Setup

- Hypothesis ID: `H63`.
- Policy: released Square checkpoint `trc7rbt0_step_110000`, EMA weights.
- Root environment seeds: `20261080` through `20261087`.
- Replicas: four independent policy-source streams per root seed, for 32 complete episodes.
- Environment construction: every replica in a group receives the same root seed; initial processed
  observation and 26-dimensional privileged-state hashes must match within each group.
- Source: stateless unit Gaussian keyed by `(root_seed, replica, replan_index)`.
- Sampler: official Euler integration with ten velocity evaluations.
- Prediction horizon: 16 actions; execution horizon: 16 actions, matching official Square FPO++
  collection rather than the checkpoint-native eight-action evaluator.
- Episode horizon: 400 environment steps; stop accounting after each lane's first terminal.
- Renderer: rootless OSMesa. This is a mechanism audit, not an official EGL benchmark claim.

For group outcomes `R_i in {0,1}`, define the leave-one-out episodic advantage

`A_i = R_i - sum_{j != i} R_j / 3`.

The same scalar would later weight every valid action chunk from replica `i`; this audit does not
update the actor or critic.

## Locked Validity Gates

1. all 32 episodes terminate within 400 steps with finite binary outcomes;
2. processed initial-observation and privileged-state hashes match across all four replicas of every
   root seed;
3. aggregate source absolute mean is at most `0.05` and source standard deviation lies in
   `[0.95, 1.05]`;
4. median within-group pairwise normalized RMS difference of first action chunks is at least `0.05`;
5. leave-one-out advantages are finite and sum to zero within every group to absolute tolerance
   `1e-7`;
6. all policy parameters remain bitwise unchanged.

## Locked Signal Gates

All gates must pass:

1. at least six successes and at least six failures occur among the 32 episodes;
2. at least five of eight root groups contain both a success and a failure;
3. at least 20 episodes have nonzero leave-one-out advantage;
4. effective sample size of the absolute nonzero advantages,
   `(sum |A|)^2 / sum A^2`, is at least 15.

The smoke uses only seeds `20261080,20261081` and two replicas per seed. It checks infrastructure and
validity invariants only; formal signal gates do not apply to smoke results.

## Decision Rule

Any formal validity or signal failure refutes H63. Do not change group size, seeds, execution horizon,
source distribution, success threshold, or rerun another block. If every gate passes, separately
register and commit a paired two-actor-update Square reward-screen protocol before changing the
training loop. That later screen must compare official zero- and Gaussian-source metrics and include
a matched official-FPO++ control; this audit alone cannot establish a benchmark improvement.

