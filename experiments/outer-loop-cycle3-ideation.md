# Outer Loop Cycle 3: Candidate Methods

The step-6000 audit supplies dense on-policy rewards, while reward weighting, source guidance,
reflow geometry, and normalization scope have not improved FPO++. The next intervention should
improve information efficiency rather than rescale the same estimate.

| Rank | Candidate | Mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | Joint antithetic CFM sampling | Balance both uniform time and Gaussian noise in each MC8 ratio estimate | Fixed-data repeated gradient-estimator audit | Test H14 |
| 2 | Successful on-policy replay | Reuse rare standard-Gaussian successes across updates | Staleness/ratio audit then short training | Park; higher bias and implementation cost |
| 3 | Critic-free outcome advantage | Replace noisy early critic targets with terminal success labels | Offline advantage/gradient audit | Park; overlaps failed lambda-1 direction |
| 4 | Successful-chunk CFM auxiliary | Add direct self-imitation signal on successful actions | One-update gradient alignment audit | Park; auxiliary weight introduces tuning |
| 5 | Pretrained-policy anchor | Preserve the BC distribution during RL updates | Short paired degradation test | Reject for now; preservation is not a gain mechanism |
| 6 | kNN entropy bonus | Preserve action diversity without likelihoods | Sampling/entropy audit | Reject for Can; Appendix D.5 reports entropy-preserving ASPO hurts manipulation |
| 7 | Quasi-Monte Carlo CFM samples | Reduce integration-variable discrepancy | Fixed-data estimator audit | Park behind simpler antithetic sampling |
| 8 | Categorical success critic | Model binary terminal outcomes better than scalar MSE | Critic calibration audit | Park; larger code change |
| 9 | Pairwise success preference loss | Contrast successful and failed action chunks | Matched-state availability audit | Reject; trajectories do not share states |
| 10 | Longer online reflow auxiliary | Straighten the updated policy during RL | Existing gradient-disagreement audit | Reject; H2 refuted the stabilization mechanism |

Selected pitch: FPO++ spends eight CFM evaluations on independent time/noise draws whose Monte
Carlo error directly perturbs its policy ratio. Pairing each draw with its distribution-symmetric
counterpart may reduce gradient-estimator noise at unchanged compute and without altering the
objective or policy source distribution.
