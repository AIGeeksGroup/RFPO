# Outer Loop Cycle 5: Gradient-Budget Candidates

H16 shows that independent CFM draws are only modestly less shifted than stored draws and do not
retain a stable enough gradient after the update. Resampling fixed draws is closed. The next
candidate must improve the accuracy of the reward-weighted gradient at fixed model-evaluation cost,
not add reward selection, source mixtures, or another scalar transform.

| Rank | Candidate | Mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | Advantage-stratified MC budget | Spend more CFM samples on chunks that dominate the weighted policy gradient | Fixed-budget gradient MSE audit | Test H17 |
| 2 | Bernoulli success critic | Match terminal binary return likelihood rather than scalar MSE | Calibration and advantage-rank audit | Park behind H17 |
| 3 | Leave-environment-out baseline | Reduce common-mode rollout baseline error | Offline variance audit | Park; weak state conditioning |
| 4 | Actor trust-region line search | Reject updates whose measured CFM ratio shift is too large | Per-epoch gradient-path audit | Park; higher implementation cost |
| 5 | Fewer PPO epochs | Reduce actor drift | Schedule comparison | Reject without an adaptive mechanism |
| 6 | More uniform MC samples | Reduce estimator noise globally | Existing MC16 online result | Reject; H5 already failed |

Selected pitch: the official estimator spends MC8 on every chunk even though the surrogate gradient
is multiplied by advantage. If gradient-noise scale is not strongly anti-correlated with advantage,
low-advantage chunks consume samples that reduce little of the final weighted-gradient variance. A
fixed 12/4 allocation between the upper and lower halves of absolute advantage preserves the exact
average MC8 budget and can be falsified before changing an online update.
