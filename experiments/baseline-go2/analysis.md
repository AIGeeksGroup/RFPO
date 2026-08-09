# Official Go2 Baseline Analysis

Date: 2026-08-08

Run: `2026-08-08_16-24-22_go2_official_seed42_20260808`

Configuration: official 4096 environments, 1500 iterations, seed 42. The runner evaluated all 31
saved checkpoints under both zero and random sampling. Although `eval_episodes=10`, the released
runner computes `target_episodes_per_env=max(1, eval_episodes // num_envs)`. With 4096 evaluation
environments it therefore collects one episode per environment, or 4096 episodes per source mode
and checkpoint, rather than 10. The recorded means below are unchanged; their effective evaluation
sample size is larger than initially documented.

| Metric | Final | Best | Best iteration |
|---|---:|---:|---:|
| Training mean reward | 40.6197 | 40.8130 | 1485 |
| Zero-sampling evaluation reward | 41.5294 | 41.5572 | 1450 |
| Random-sampling evaluation reward | 40.5234 | 40.5234 | 1499 |

At iteration 1499, zero-sampling reward standard deviation was 1.9566 and random-sampling reward
standard deviation was 2.2273. Mean episode lengths were 998.57 and 997.56 respectively. Final
value loss was 0.001401, surrogate loss was -0.000509, and throughput was 88,680 FPS.

## Decision

H0 is supported. The training and evaluation returns both reach and slightly exceed the paper's
approximately 40-return reference regime. The complete 31-checkpoint post-evaluation sequence rules
out an incomplete-run interpretation. A multi-seed uncertainty estimate remains useful for a final
paper table, but is not required before screening improvement hypotheses.
