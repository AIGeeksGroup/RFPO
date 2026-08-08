# Can Base Checkpoint: 200-Episode Reproduction

Date: 2026-08-08

Server: `haoyu@36.33.157.152:30061`, NVIDIA H200, physical GPU 1

Checkpoint: `95j3noe4_step_1000`, EMA weights, 10 Euler sampling steps

| Run | Parallel envs | Sampling source | Successes | Episodes | Pooled success | Wilson 95% CI |
|---|---:|---|---:|---:|---:|---:|
| Validation A | 30 | Zero | 141 | 200 | 70.5% | 63.8%-76.4% |
| Validation A | 30 | Gaussian random | 19 | 200 | 9.5% | 6.2%-14.4% |
| Official-parallelism confirmation | 50 | Zero | 142 | 200 | 71.0% | 64.4%-76.9% |
| Official-parallelism confirmation | 50 | Gaussian random | 28 | 200 | 14.0% | 9.9%-19.5% |

The paper reports 73.76% zero-sampling base success over 1,000 episodes and highlights an approximately 10% random-sampling Can base policy. Both zero-sampling intervals cover 73.76%. Across the two random-sampling runs, pooled success is 47/400 (11.75%), compatible with the reported low-success regime.

The evaluation script originally returned an unweighted average of per-environment rates, which differs from the pooled rate when the episode count is not divisible by the number of environments. The reproduction records raw successes divided by raw completed episodes; the default runner now uses the paper's 50 environments.
