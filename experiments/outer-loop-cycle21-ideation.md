# Outer Loop Cycle 21: Reward-Target Candidates

## Constraint From H34

The next method must change which observed credit enters the first actor step. Coordinate-wise robust
aggregation preserved scale but removed task-relevant joint structure, while earlier gradient
sampling and optimizer-only variants have not improved reward.

## Shortlist

| Rank | Candidate | Distinct evidence source | Fast audit | Decision |
|---:|---|---:|---:|---|
| 1 | Terminal-consistency GAE filter | observed terminal sign | outcome-gradient cosine | Select H35 |
| 2 | Complete-trajectory Monte Carlo FPO | observed terminal return | reward screen only | Park; overlaps H7 |
| 3 | Leave-environment-out terminal baseline | observed terminal return | variance audit | Park; higher variance |
| 4 | Frozen-initial-gradient projection | initial GAE gradient | virtual update | Park; no new reward evidence |
| 5 | Extragradient actor step | future surrogate gradient | virtual update | Park; doubles compute |
| 6 | Coordinate sign consensus | microbatch gradients | outcome-gradient cosine | Reject by H34 stop rule |

## Selection

H35 keeps the magnitude information supported by H22 and does not select success-only data as in
H15. It uses terminal labels only to remove sign-contradictory terms, retains both successful and
failed outcomes, and leaves censored chunks unchanged in any later online implementation. A fixed
two-replica audit can reject it before training if filtering merely induces selection bias.
