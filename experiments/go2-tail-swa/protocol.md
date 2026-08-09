# Protocol: H65 Fixed-Tail Go2 Actor Averaging

## Hypothesis

Uniformly averaging the EMA actor tensors from the final five saved checkpoints of the reproduced
official Go2 FPO++ run improves paired zero/random return relative to the final checkpoint by
reducing late optimization-path variance. Critic parameters, observation normalizers, optimizer
state, flow sampler, and environment configuration remain those of the final checkpoint.

## Locked Candidate

- Source run: `2026-08-08_16-24-22_go2_official_seed42_20260808`.
- Actor checkpoints: iterations `1300, 1350, 1400, 1450, 1499`, exactly.
- Candidate actor: elementwise arithmetic mean with weight `0.2` for each checkpoint.
- Candidate container: a copy of `model_1499.pt` with only `model_state_dict` keys prefixed by
  `actor.` replaced by the average. No critic, normalizer, optimizer, EMA metadata, or iteration field
  is averaged or selected.
- Control: unchanged `model_1499.pt`.
- Sampler: official Euler integration with 64 velocity evaluations.
- This is an inference checkpoint audit, not additional FPO++ training.

## Stage A: Construction Validity

The creator must refuse overwrite and verify identical checkpoint keys, tensor shapes, and floating
dtypes. Every averaged actor tensor must be finite and equal the float64-computed five-checkpoint mean
after casting to its stored dtype. Every non-actor tensor and non-model payload must match the final
checkpoint exactly. The candidate actor's relative L2 displacement from the final actor must lie in
`[0.01, 0.06]`, proving activity without leaving the observed late-trajectory neighborhood.

Failure stops H65 without changing the window, weights, payload policy, or source run.

## Stage B: Paired Reward Screen

- Environments and episodes per method per mode: 256, exactly one completed episode per environment.
- Evaluation seed: `20261090`; Gaussian source seed: `20261091`.
- Modes: zero and standard-Gaussian random, both required.
- Run control and candidate with identical task, environment count, seeds, Euler-64 sampler, and
  episode accounting.
- Analyze per-environment paired return differences and a deterministic seed-20261092 bootstrap with
  20,000 resamples over all 512 mode-balanced pairs.

All gates must pass:

1. all 1,024 method-mode episodes complete with finite actions and returns;
2. candidate minus control mean return is at least `-0.05` in each mode;
3. the equally weighted mean of the zero and random paired gains is at least `+0.10`;
4. the pooled paired-bootstrap 95% lower bound is strictly positive.

Any failure refutes H65. Do not change the averaging window, use reward-selected checkpoints, tune
weights, add a seed, continue training, or test an ensemble. A full pass authorizes only a separately
committed independent 4,096-environment confirmation using seed `20261093` and source seed
`20261094`; the screen alone is not a benchmark-improvement claim.
