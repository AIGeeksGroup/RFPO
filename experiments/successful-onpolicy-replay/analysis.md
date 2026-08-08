# Analysis: Successful On-Policy Replay Staleness

## Result

H15 is refuted. The locked two-iteration step-6000 audit completed with finite losses and gradients.
Iteration 2 collected 127/193 successful Gaussian-source episodes and exposed 1,608 valid chunks
linked to successful episodes, well above the required 32. The seeded audit used 64 chunks.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Active positive replay ratios | at least 70% | 40.82% | fail |
| Ratio ESS | at least 80% | 99.56% | pass |
| Pre/post replay-gradient cosine | at least 0.8 | 0.61384 | fail |
| Success/fresh-positive gradient cosine | at least 0.5 | 0.04877 | fail |

Before the actor update, replay ratios were numerically one (mean 1.000000, standard deviation below
`1e-6`). After the standard update, their mean was 1.0361 with a range of 0.8635 to 1.3020. The
importance weights remained well balanced, but most positive replay terms were already above the
PPO upper clipping boundary. Replay-gradient norm changed from 10.012 to 6.590, while the fresh
positive-advantage gradient norm was 13.291.

## Interpretation

The obstacle is not importance-weight degeneracy. A single official FPO++ update changes the useful
positive replay direction enough that most successful chunks are clipped and the remaining replay
gradient is only moderately aligned with its pre-update direction. More importantly, selecting
successful episodes produces a gradient nearly orthogonal to the same-rollout positive-advantage
gradient, indicating strong selection bias rather than extra unbiased on-policy information.

Stop without implementing replay training. Do not tune replay age, capacity, ratio clipping, replay
weight, success threshold, learning rate, source distribution, or mask behavior. The next candidate
is a critic-free terminal-outcome advantage audit, which should test label/gradient agreement before
changing the online objective.

Remote evidence:

- `~/workspace/outputs/fpo-control/results/can_step6000_successreplay_audit_seed20260814/success_replay_audit.json`
- `~/workspace/outputs/fpo-control/logs/can_step6000_successreplay_audit_seed20260814.log`
