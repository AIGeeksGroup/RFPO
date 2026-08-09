# Protocol: Deterministic-Reference Episodic Advantage Audit

## Hypothesis

A frozen zero-source Square rollout from the same initial scene provides a more data-efficient
control variate for Gaussian-source FPO++ than H63's four-sample group mean. The reference-relative
advantage

`A(scene, gaussian trajectory) = R_gaussian - R_zero(scene)`

reinforces stochastic rescues of deterministic failures and penalizes stochastic regressions on
deterministically successful scenes, while assigning no update to behavior with the same terminal
outcome as the reference.

## Locked Setup

- Hypothesis ID: `H64`.
- Policy: released Square checkpoint `trc7rbt0_step_110000`, EMA weights.
- Gaussian trajectories: the 32 immutable H63 formal records in
  `experiments/group-relative-episode-audit/results/audit/records.json`, SHA-256
  `d6e6568f2117f88ebb15239a589853882b1653a48471591abac20689b249003c`.
- Root environment seeds: `20261080` through `20261087`, exactly matching H63.
- Reference: one complete zero-source policy trajectory per root seed.
- Sampler: official Euler integration with ten velocity evaluations.
- Prediction and execution horizon: 16 actions, matching H63 and official Square FPO++ collection.
- Episode horizon: 400 environment steps.
- Renderer: rootless OSMesa. This is a control-variate audit, not an official benchmark result.
- No actor or critic gradients, updates, fitting, or source selection occur.

The smoke uses seeds `20261080,20261081` and the corresponding eight archived Gaussian records. It
checks only infrastructure and validity; formal signal gates do not apply.

## Locked Validity Gates

1. the archived H63 file checksum and per-seed four-replica manifest are exact;
2. every zero-reference episode completes within 400 steps with a finite binary outcome;
3. each zero-reference initial processed-observation and privileged-state hash exactly matches all
   four archived H63 replicas for that seed;
4. every supplied source tensor is bitwise zero and every first action chunk is finite;
5. all reference-relative advantages are finite and lie in `{-1, 0, +1}`;
6. all policy parameters remain bitwise unchanged.

## Locked Signal Gates

All gates must pass:

1. zero references contain at least two successes and two failures;
2. at least four Gaussian trajectories have positive reference-relative advantage;
3. at least four Gaussian trajectories have negative reference-relative advantage;
4. at least 20 of 32 Gaussian trajectories have nonzero advantage, strictly improving H63's 16;
5. absolute-weight effective sample size is at least 15. For unit nonzero weights this equals the
   nonzero count, but it is retained for compatibility with later weighted variants.

## Decision Rule

Any formal validity or signal failure refutes H64. Do not add reference smoothing, partial pooling,
extra zero references, source mixtures, new seeds, or a training run. If every gate passes, separately
register and commit a fixed-batch actor-gradient audit comparing official GAE with deterministic-
reference weights before integrating them into a reward screen. This audit alone cannot establish a
benchmark improvement.

