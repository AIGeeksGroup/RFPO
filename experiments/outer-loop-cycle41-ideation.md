# Outer Loop Cycle 41: Contact-Relevant Action Semantics

## Failure Boundary

H27, H54, and H55 close global replanning frequency, fixed overlap averaging, and adaptive-horizon
headroom. H31 also shows that smoothing the complete seven-dimensional action chunk can destroy
Gaussian-source success. None of these tests whether the continuous flow output violates the
discrete semantic support of the gripper command while the six arm-control dimensions remain useful.

The released Can checkpoint stores gripper normalization statistics with mean `-0.16610` and standard
deviation `0.98611`, consistent with demonstrations concentrated near the two controller commands
`-1` and `+1`. At inference, however, the policy unnormalizes and executes the continuous flow
endpoint directly.

## Diverged Candidates

1. Project only the gripper coordinate to `{-1, +1}` at zero threshold.
2. Add gripper hysteresis around zero.
3. Hold the gripper state until a high-confidence sign change.
4. Project all seven action coordinates to empirical support.
5. Train a categorical gripper head beside the continuous arm flow.
6. Penalize intermediate gripper values during behavior cloning.
7. Replan immediately before a predicted gripper sign change.
8. Use simulator contact to gate gripper closure.

## Convergence

Select H56, zero-threshold binary projection of only the final gripper coordinate. It is the minimal
test of support mismatch, requires no training or privileged state, preserves the complete arm
trajectory bitwise, and can reuse H55's valid 15/20 common-seed control. A binary head or auxiliary
loss would add training cost before establishing that discrete gripper execution helps at all.
Hysteresis, confidence thresholds, sign-change replanning, and contact gates introduce additional
parameters or timing mechanisms and are not authorized neighbors if the fixed projection fails.

Two-sentence pitch: Can demonstrations use a two-state gripper controller, but a continuous flow can
emit weak intermediate commands that are neither semantic state. Snap only that coordinate back to
the demonstrated support and leave the learned continuous arm control untouched.
