# FPO++ KL-Adaptive Learning Rate

## Evidence

Yi et al. (2026), *Flow Policy Optimization*, Appendix D.2, attributes part of the motion-tracking
gap to the absence of entropy regularization and KL-adaptive learning rates. The authors report
"some improvement" after adding both components, while noting that FPO++ still remained slightly
below tuned Gaussian PPO in return. This is supporting evidence for a small mechanism test, not a
claim that adaptive LR improves the released manipulation benchmark.

The appendix defines its KL proxy as the L2 distance between the current policy's predicted noise
and the corresponding behavior-policy prediction. Predicted noise is obtained algebraically from
the velocity prediction and sampled action. It does not publish the target or update factors.

The released repository supplies those missing details in
`isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/algorithms/fpo.py`: cache `x1_pred` under the behavior
policy, compute mean squared current-minus-old `x1_pred`, divide LR by 1.5 above twice the desired
KL, multiply it by 1.5 below half the desired KL, and use no change inside the band. The G1 config
uses adaptive scheduling; the shared locomotion default is `desired_kl=1e-4`, base LR `1e-4`, and
absolute bounds `[1e-5, 1e-2]`.

## Transfer Decision

H39 keeps the paper/repository target, thresholds, and factor. Because manipulation's base actor LR
is already `1e-5`, copying the locomotion absolute lower bound would prohibit the hypothesized
downward response. Preserve the locomotion bounds relative to its base LR instead: candidate bounds
are `[0.1 * manipulation_base_lr, 100 * manipulation_base_lr]`. This scale transfer is frozen before
results and is reported as a manipulation adaptation, not an exact reproduction of the unpublished
manipulation setting.

Source provenance: local paper PDF `FPO for robot.pdf` and released repository, inspected
2026-08-09. Paper metadata is recorded in `literature/yi2026_fpo_control.md`.

