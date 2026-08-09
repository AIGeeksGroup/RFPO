# H71: Batched Antithetic Inference Throughput

## Question

Can the two H70 Euler-32 endpoint solves be fused along the environment batch
dimension so that equal-NFE antithetic projection also has practical latency
close to one Euler-64 random solve?

## Fixed audit

- Checkpoint: official Go2 seed-42 `model_1499.pt`
- Device: one otherwise idle H200 GPU
- Frozen normalized observations from 4,096 Go2 environments
- Fixed Gaussian source generated with seed `20261610`
- Methods:
  - `random64`: one Euler-64 endpoint
  - `antithetic32_sequential`: two sequential Euler-32 endpoints
  - `antithetic32_batched`: concatenate observations and `z/-z`, solve one
    doubled Euler-32 batch, split endpoints, and average
- Ten untimed warm-up calls per method, followed by five timing blocks of 20
  calls per method. Synchronize CUDA before and after every block.
- Rotate method order across blocks. Report median block latency, interquartile
  range, actions/second, and peak allocated CUDA memory.

No rollout return is measured: the frozen-policy H70 paired evaluation remains
the reward evidence. This audit tests only numerical equivalence and deployment
throughput.

## Decisions

H71 passes if:

1. batched and sequential antithetic actions are finite and have maximum
   absolute difference at most `1e-5`;
2. batched median latency is at least 15% lower than sequential latency; and
3. batched median latency is no more than 1.25 times random64 latency.

A pass authorizes using the batched implementation in later independent-seed
and cross-task evaluations after a paired rollout equivalence check. A failure
retains H70's equal-NFE result but forbids a wall-clock parity claim; do not tune
batch splitting, compilation modes, CUDA graphs, or precision in this screen.
