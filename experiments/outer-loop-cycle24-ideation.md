# Outer-Loop Cycle 24: Preserve Joint Gradient Structure

## Problem-First Diagnosis

H34 showed that coordinate-wise robust aggregation preserves overall scale but deletes useful joint
gradient structure. H36 showed that strong ratio-boundary pressure retains active terms but sacrifices
about two thirds of surrogate progress. H37 must therefore operate on complete gradient vectors,
preserve the official reward and objective, and be rejectable before optimizer integration.

## Candidate Set

| Rank | Candidate | Distinct mechanism | Decision |
|---:|---|---|---|
| 1 | Geometric median-of-means (GMOM) | robust center of complete microbatch gradients | Select H37 |
| 2 | Outer-PPO update application | decouple inner update estimate from outer optimizer | Park; larger optimizer and tuning surface |
| 3 | Extragradient actor step | anticipate local gradient rotation | Park; doubles actor passes and needs a step rule |
| 4 | Rollout-gradient SVRG | control minibatch variance with a full-rollout reference | Park; full gradient storage and online integration first |
| 5 | Empirical-Fisher natural gradient | precondition by local policy geometry | Park; expensive implicit-flow Fisher solve |
| 6 | Frozen-initial-gradient PCGrad | reject later components conflicting with the initial update | Reject; audited later cosines are positive, so conflict-only projection is mostly inactive |
| 7 | Held-out Armijo line search | accept steps only when a held-out surrogate improves | Reject; risks training on the audit sensor and repeats H18 stopping logic |
| 8 | Epochwise behavior-reference refresh | make every epoch locally on-policy | Reject; changes the importance-ratio target and may hide cumulative drift |
| 9 | Per-step EMA interpolation | damp actor displacement after every update | Reject; close to failed fixed interpolation H25 without a new mechanism |
| 10 | Per-sample gradient norm clipping | suppress individual large gradients | Reject; adds a clip threshold and discards joint sample structure |
| 11 | Coordinate trimming/sign voting | robustify gradient coordinates | Reject explicitly by H34 |
| 12 | Adaptive ratio-rollback strength | trade active ratios against progress | Reject explicitly by H36's no-tuning rule |

## Selection

GMOM directly follows the heavy-tailed PPO diagnosis of Garg et al. (2021) while addressing the
specific structural failure of coordinate-wise median aggregation. It has a deterministic,
training-free audit: use the same four-block geometry as H34, compute the geometric median in the full
parameter-gradient space, and compare it with centered observed terminal outcomes in two replicas.

Two-sentence pitch: FPO++ actor updates vary across microbatches, but coordinate-wise robustification
destroys task-relevant parameter relationships. We test whether a robust center of whole gradient
vectors improves outcome alignment while retaining the official mean update's direction and scale.

