# Outer Loop Cycle 4: Update-Generalization Candidates

H14-H15 rule out symmetric sampling and successful replay under their locked gates. H15 also shows
that unit terminal-success gradients are nearly orthogonal to fresh positive-advantage gradients, so
critic-free outcome labels and successful-chunk auxiliaries are not promoted. The next direction
must change estimator reuse rather than reward selection or scalar weighting.

| Rank | Candidate | Mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | Epoch-resampled behavior-anchored CFM | Prevent ten PPO epochs from overfitting the same stored MC8 draws | Stored-versus-held-out ratio audit | Test H16 |
| 2 | Advantage-proportional MC budget | Spend fixed CFM compute on chunks with the largest policy contribution | Per-chunk variance/contribution audit | Park behind H16 |
| 3 | Bernoulli success critic | Match the binary terminal outcome distribution rather than scalar MSE | Calibration and advantage-rank audit | Park; larger change |
| 4 | Leave-environment-out baseline | Reduce shared rollout baseline bias without a learned critic change | Offline variance audit | Park; weak state conditioning |
| 5 | Temporal-difference residual weighting | Focus updates where critic Bellman errors indicate new information | Gradient-alignment audit | Reject for now; another scalar weight |
| 6 | Reward redistribution | Move terminal success credit to causal action chunks | Requires learned return decomposition | Reject; insufficient trajectories for a clean pilot |
| 7 | Fewer PPO epochs | Reduce fixed-sample overfitting directly | Short hyperparameter comparison | Reject as a pure schedule change before mechanism audit |
| 8 | BC anchor | Limit actor drift from the strong initialization | Degradation audit | Reject; preservation is not a gain mechanism |
| 9 | Critic-free terminal outcome | Remove critic noise with binary success-to-go labels | Existing H15 gradient comparison | Reject; cosine 0.049 |
| 10 | Successful-chunk CFM auxiliary | Self-imitate successful chunks | Existing H15 gradient comparison | Reject; same conflicting selection gradient |

Selected pitch: FPO++ stores eight CFM draws with each rollout action and reuses them through ten
optimization epochs. If the actor fits those integration variables rather than the underlying CFM
expectation, stored ratios will move farther than independent held-out ratios. Recomputing behavior
losses from a frozen rollout policy on fresh draws each epoch would then improve ratio information
without changing rewards, actions, source distribution, or total MC samples per forward pass.
