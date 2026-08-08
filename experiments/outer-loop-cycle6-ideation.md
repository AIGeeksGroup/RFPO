# Outer Loop Cycle 6: Trust-Region and Learning-Signal Candidates

H14 and H17 both reduced finite-MC gradient error but failed to preserve the realized update
direction. H16 separately showed that a full ten-epoch official update leaves only 54.7% of fresh
positive ratios active and rotates the held-out gradient to cosine 0.743. The next experiment should
address update drift or target quality rather than rearrange CFM samples.

| Rank | Candidate | Mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | Held-out ratio early stop | Roll back an actor epoch once fresh positive ratios leave the PPO trust region | One-update epoch-path audit | Test H18 |
| 2 | Discounted-success critic | Fit bounded sparse-return targets with a proper Bernoulli loss | Cross-iteration calibration and gradient audit | Park behind H18 |
| 3 | Actor line search | Backtrack the complete actor step using a held-out surrogate | One-update displacement audit | Park; more optimizer-state complexity |
| 4 | FPO ratio group averaging | Reduce Jensen amplification by averaging CFM losses before exponentiation | Ratio-bias audit | Park; overlaps the released vanilla-FPO ablation |
| 5 | Fixed fewer epochs | Reduce actor drift without measuring it | Five-update schedule comparison | Reject without an adaptive mechanism |

Selected pitch: FPO++ repeatedly optimizes a noisy pseudo-ratio on the same rollout until much of the
fresh positive signal is already clipped. A small fixed held-out CFM redraw can act as a direct trust-
region sensor, retaining early useful movement and rejecting the first epoch that crosses a locked
active-ratio boundary.
