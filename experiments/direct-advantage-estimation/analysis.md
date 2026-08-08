# Analysis: Continuous-Action Direct Advantage Audit

## Result

H21 is refuted. The locked step-6000 audit completed with 129 fully valid fresh-rollout chunks,
finite nonzero gradients, and nonzero candidate-weight variance. The candidate failed every
performance gate.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Spearman gain over control GAE | at least +0.05 | -0.85797 | fail |
| Batch-0 gradient cosine gain | at least +0.05 and positive candidate | -0.82380; candidate -0.13419 | fail |
| Batch-1 gradient cosine gain | at least +0.05 and positive candidate | -1.20361; candidate -0.48609 | fail |
| DAE loss reduction | at least 20% | 2.96% | fail |

Control GAE had a fresh-rollout Spearman correlation of 0.81660 with uncensored discounted Monte
Carlo returns. The direct-advantage candidate had correlation -0.04137. On the two seeded 32-chunk
batches, control gradient cosine to the Monte Carlo reference was 0.68961 and 0.71752, whereas the
candidate cosine was -0.13419 and -0.48609. Reference gradient norms were 3.19827 and 3.68489.

Iteration-1 side training used 102 complete DAE windows. Its loss decreased monotonically from
0.005726 to 0.005557, but the 2.96% reduction was far below the locked gate. Iteration-2 collection
success was 6/10 under the official scale-1 Gaussian source, so the negative result is not caused by
an all-zero outcome batch.

## Interpretation

Monte Carlo centering over four implicit continuous actions does not recover the discrete-action
DAE mechanism in this setting. More importantly, the released GAE signal already ranks outcomes
well; replacing it with a learned action-effect head destroys both ordering and actor-gradient
alignment. Stop without online integration, additional seeds, or tuning the head, horizon,
centering count, optimizer, or loss.

The next direction should preserve the strong ordering in official GAE and test whether a simple
rank-preserving nonlinear weight transform can reduce harmful magnitude distortion. It should be
audited on frozen actor gradients before any reward training.

## Artifacts

- `results/direct_advantage_audit.json`
- `~/workspace/outputs/fpo-control/results/can_step6000_dae_audit_osmesa_formal_seed20260819/direct_advantage_audit.json`
- `~/workspace/outputs/fpo-control/logs/can_step6000_dae_audit_osmesa_formal_seed20260819.log`
