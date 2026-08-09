# H66 Go2 Antithetic Source Ensemble: Final Analysis

## Result

H66 is supported. The coefficient-free one-pair antithetic source ensemble passed its mechanism
audit, paired 256-environment screen, and independently seeded 4,096-environment official-scale
confirmation. It improves the final reproduced Go2 actor's Gaussian-source return while leaving its
zero-source actions bitwise unchanged.

| Stage | Environments per method | Random control | Antithetic candidate | Paired gain | Bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| Screen | 256 | 40.32237 | 41.68368 | +1.36131 | [1.02934, 1.81896] |
| Independent confirmation | 4,096 | 40.52085 | 41.43657 | +0.91571 | [0.81598, 1.01501] |

The official-scale control independently reproduces the earlier final random-source result of
`40.5234` to within `0.0026`. The confirmation gain exceeds the locked `+0.20` gate by more than four
times and its confidence interval lies entirely above zero. Since candidate and control zero actions
are identical, an equally weighted zero/random aggregate improves by exactly half the random-mode
gain, or `+0.45786`, without requiring another stochastic zero rollout.

## Method

For every Gaussian inference action, draw one source `z`, integrate the unchanged flow actor from
`z` and `-z` with the official Euler-64 solver, and execute

```text
0.5 * (F(observation, z) + F(observation, -z)).
```

Zero inference calls the original `F(observation, 0)` path exactly once. No model tensor,
normalizer, optimizer state, environment configuration, or training procedure changes.

## Validity

The fixed-state audit used eight consecutive batches of 256 observations. Candidate-to-zero
normalized RMS was `0.03462`, compared with `0.28726` for the positive random endpoint, so symmetric
averaging retained a nonzero even component while canceling `87.95%` of the measured random-source
displacement. All sources/endpoints were finite, negative sources and endpoint means were bitwise
exact, zero dispatch was bitwise exact, and actor parameters were unchanged.

The confirmation completed all 8,192 episodes with finite actions and returns. Both conditions had
initial-observation SHA-256
`dddfc254ad84d866ca2bc7b66a0c6d7ce40e6460fd34d4e798e08c0a3b9c8b71` and complete source-stream
SHA-256 `87c831c638f45285d9b0dc3abcbcb6cabb1e3a3cefb26a13d082ad816c7c734b` over 98,304,000 source
values. The paired SEM was `0.05117`; the bootstrap used the preregistered seed and 20,000 resamples.

## Cost And Scope

The Gaussian candidate uses two Euler-64 endpoints, or 128 velocity-network evaluations per action,
versus 64 for control. This is a reward/compute tradeoff, not a faster sampler. The supported claim is
for the reproduced final Go2 checkpoint and Gaussian-source evaluation; Can H26 remains negative for
the different claim that antithetic averaging beats zero-source sparse-task behavior.

All screen and confirmation evaluator JSON, analysis JSON, and complete logs are archived under
`results/`. Reproduction entry points are `research_scripts/remote/run_go2_h66.sh` and
`research_scripts/remote/run_go2_h66_confirmation.sh`.
