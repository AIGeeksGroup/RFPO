# Four-Step Receding-Horizon Execution: Result

## Outcome

H27 is refuted at the locked screen. The official eight-action execution achieved 14/20 (70%) Can
success, while executing four actions before replanning achieved 12/20 (60%). The candidate missed
the required 10-point gain and instead lost 10 points.

| Executed actions | Success | Average episode length | FPS |
|---:|---:|---:|---:|
| 8 (control) | 14/20 (70%) | 209.0 | 99.4 |
| 4 (candidate) | 12/20 (60%) | 237.7 | 110.3 |

Both conditions used the released step-1000 EMA policy, zero-source inference, 10 Euler steps, seed
20260826, 16 environments, and OSMesa. No non-finite action occurred.

## Interpretation

More frequent observation feedback is not sufficient to improve this checkpoint. One plausible
explanation is that behavior cloning couples adjacent actions across the trained eight-step execution
cadence, so replanning halfway through a chunk disrupts temporal consistency. The experiment does
not isolate that mechanism, and the negative reward gate does not justify a follow-up mechanism
study.

Stop without a confirmation seed or testing 1, 2, or 6 executed actions. Retain `--action-steps` as
reproducible evaluation infrastructure, not as a benchmark improvement.
