# Protocol: H66 Go2 Antithetic Source Ensemble

## Hypothesis

For the reproduced final Go2 FPO++ actor, averaging actions obtained from matched Gaussian sources
`z` and `-z` reduces the official random-source return deficit while leaving zero-source inference
exactly unchanged. Symmetry cancels the source-odd component of the learned flow endpoint; unlike
reflow or a higher-order solver, this mechanism directly reduces action variability associated with
the measured reward gap.

## Locked Intervention

- Checkpoint: `model_1499.pt` from reproduced run
  `2026-08-08_16-24-22_go2_official_seed42_20260808`.
- Candidate Gaussian action: `0.5 * (F(o, z) + F(o, -z))` for one freshly sampled
  `z ~ N(0, I)` per environment and policy step.
- Candidate zero action: the unmodified `F(o, 0)` path, with no duplicate solve.
- Flow solver: official Euler-64 for every endpoint.
- One antithetic pair, unit source scale, arithmetic averaging, and no checkpoint changes.
- Report the doubled Gaussian-mode velocity-network evaluations; latency is not claimed to improve.

## Stage A: Fixed-State Mechanism Audit

Use 256 environments, seed `20261100`, source seed `20261101`, and eight consecutive control-policy
states. At every state evaluate `F(o,z)`, `F(o,-z)`, `F(o,0)`, and the candidate without updating the
policy. All gates must pass:

1. all sources and endpoints are finite, and negative sources are bitwise exact negations;
2. candidate actions equal the float32 arithmetic mean of the two endpoint tensors bitwise;
3. candidate-to-zero normalized RMS is at least `0.01`, proving the intervention does not merely
   duplicate zero inference;
4. candidate-to-zero normalized RMS is strictly below positive-source-to-zero normalized RMS,
   confirming cancellation of source-dependent variation;
5. actor parameters remain bitwise unchanged.

Any failure refutes H66 without reward rollout, source scale, pair count, checkpoint, solver, or seed
changes.

## Stage B: Paired Random-Mode Screen

If Stage A passes, evaluate the official random control and antithetic candidate in separate fresh
processes with exactly 256 environments and one completed episode per environment. Both use
evaluation seed `20261100`, source seed `20261101`, and identical first-member Gaussian source
streams. Require exact initial-observation and complete source-stream hashes across methods.

All gates must pass:

1. all 512 episodes complete with finite actions and returns;
2. candidate-minus-control paired mean return is at least `+0.30`;
3. a deterministic 20,000-resample paired bootstrap using seed `20261102` has a strictly positive
   95% lower bound.

Zero-mode return is not rerun because the candidate dispatches the exact unmodified zero path and
Stage A requires action identity. A screen pass authorizes only a separately committed independent
4,096-environment random-mode confirmation using evaluation seed `20261103` and source seed
`20261104`. The confirmation must retain at least `+0.20` paired mean return with a strictly positive
bootstrap lower bound before H66 is called an official benchmark improvement.

## Stop Rules

On any failed gate, do not tune source scale, number of pairs, averaging weights, solver, checkpoint,
seed, or evaluation mode. Do not replace the antithetic pair with independent samples. This protocol
tests one fixed inference method, not a sampling sweep.

## Provenance

- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*
  (arXiv:2209.03003), motivates analyzing the deterministic source-to-endpoint transport map.
- Classical antithetic variates motivate exact symmetric pairs for cancellation of source-odd
  components.
- H26 supplies the cross-domain boundary: the same construction tied rather than beat zero-source
  Can behavior, so H66 targets only the measured Gaussian-mode deficit on dense Go2.
