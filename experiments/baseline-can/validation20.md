# Can Base Checkpoint: 20-Episode Validation

Date: 2026-08-08

Server: `haoyu@36.33.157.152:30061`, NVIDIA H200, physical GPU 1

Checkpoint: `95j3noe4_step_1000`, EMA weights, 10 Euler sampling steps

Command:

```bash
FPO_GPU=1 EVAL_EPISODES=20 EVAL_ENVS=10 \
  bash research_scripts/remote/eval_can_base.sh
```

| Sampling source | Successes | Episodes | Success rate | Throughput |
|---|---:|---:|---:|---:|
| Zero | 16 | 20 | 80% | 187.6 FPS |
| Gaussian random | 2 | 20 | 10% | 314.6 FPS |

Interpretation: this short validation matches the reported behavior regime but is not used as the final reproduction estimate. The pre-registered 200-episode evaluation follows.
