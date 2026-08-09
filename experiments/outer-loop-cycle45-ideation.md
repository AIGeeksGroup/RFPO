# Outer Loop Cycle 45: Correct the First Actor Step

## Failure Boundary

The apparent step-1000 one-iteration gain is invalid because iteration 1 is critic-only and its actor
is bitwise identical to the released EMA actor. H18 and H23-H24 nevertheless establish a real
optimization problem on iteration 2: the official GAE direction is informative before optimization,
but held-out gradients rotate strongly after updates, and simply lowering the learning rate or
accumulating a full batch does not fix the rotation.

## Diverged Candidates

| Rank | Candidate | Distinct mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | AdamW extragradient | evaluates the update direction at a one-step lookahead | exact one-step two-batch audit | Select H59 |
| 2 | Rollout-gradient SVRG | reduces minibatch variance around a full-rollout reference | multi-epoch audit | Park; H24 weakens the premise and cost is higher |
| 3 | Empirical-Fisher natural gradient | changes geometry rather than scalar step size | implicit-solve audit | Park; too expensive before simpler curvature correction |
| 4 | Leave-environment-out baseline | removes shared rollout baseline noise | variance audit | Park; H13 found normalization scope inactive |
| 5 | Same-observation best-of-two ranking | changes inference candidates | reward screen | Reject; H26/H46 and cycle 39 close this family |
| 6 | More gripper source/output variants | narrows contact noise | reward screen | Reject; H57-H58 explicitly close the family |

## Selection

H59 uses one extra actor forward/backward pass on the same minibatch and the same stored CFM random
variables. It first makes an exact virtual AdamW step, recomputes the FPO++ gradient at that
lookahead, restores both parameters and optimizer state exactly, and applies the lookahead gradient
from the original state. Rewards, GAE, source distribution, CFM estimator, clipping, critic, and
learning rate are unchanged.

Two-sentence pitch: FPO++ has an informative on-policy gradient, but repeated nonlinear actor updates
quickly rotate it away from independently measured outcome directions. Extragradient asks where the
current update is about to go before committing, potentially correcting curvature-induced drift
without shrinking all policy movement.
