# Outer Loop Cycle 46: Move the Geometry Test to Dense Locomotion Reward

## Corrected Boundary

The official manipulation benchmark reports both zero and Gaussian-source success, so historical
methods that improved only zero-source behavior remain tradeoffs rather than benchmark gains. The
MuJoCo EGL failure also prevents official manipulation confirmation, while the reproduced Go2
benchmark remains healthy on the same server and supplies dense reward.

## Diverged Candidates

| Rank | Candidate | Distinct mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | Equal-NFE midpoint on final Go2 | improves flow integration accuracy without retraining | fixed-state endpoint audit, then one checkpoint evaluation | Select H60 |
| 2 | Post-RL Go2 reflow distillation | directly straightens the trained conditional policy | offline rollout dataset plus matched distillation | Park; higher infrastructure cost |
| 3 | Learned deterministic latent | optimizes the inference source rather than the flow | held-out latent-search evaluation | Park; changes the zero-sampling definition and risks validation overfit |
| 4 | Go2 clip-parameter sweep | changes FPO++ update size | short multi-seed training | Reject as immediate step; paper already sweeps 0.04-0.06 |
| 5 | Reopen zero-only manipulation candidates | optimizes the selected-checkpoint convenience metric | existing checkpoint evaluation | Reject; Figure 4 treats both source modes as main results |

## Selection

H60 transfers the numerically supported part of Rectified Flow to a domain where small action
improvements have a dense return signal. It compares official Euler-64 with explicit midpoint-32,
holding velocity-network evaluations fixed at 64. The existing final Go2 checkpoint is sufficient,
so a negative result ends the test without any new training.

