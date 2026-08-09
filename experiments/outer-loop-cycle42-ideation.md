# Outer Loop Cycle 42: Factorized Source for Hybrid Actions

## Failure Boundary

H56 shows that snapping the final executed gripper command to binary support changes most gripper
actions but leaves every deterministic Can outcome unchanged. This rejects output post-processing,
not the effect of Gaussian gripper latent noise inside the coupled flow network. The released
step-1000 checkpoint has a large zero/random gap: approximately 71% versus 11.75% success. Existing
H8-H10 source interventions operate on complete environment trajectories and all seven latent
coordinates; none preserves a standard Gaussian arm source while removing only gripper-source noise.

## Diverged Candidates

1. Standard Gaussian source for six arm coordinates and zero source for the gripper coordinate.
2. Independent learned categorical source for the gripper latent.
3. Temper only the gripper latent variance.
4. Use a two-component gripper latent mixture.
5. Condition gripper source on the current gripper state.
6. Block gripper-latent influence on arm outputs architecturally and retrain.
7. Train separate arm and gripper flow networks.

## Convergence

Select H57, the coefficient-free endpoint of candidate 3: retain unit Gaussian noise for every arm
latent and set only the final gripper latent coordinate to zero. It tests whether irrelevant
stochasticity in the hybrid-action coordinate contaminates the coupled flow transport, without
shrinking arm exploration toward the high-success deterministic policy. Learned mixtures, state
conditioning, and architectural factorization are too costly before this prerequisite demonstrates a
reward effect. A variance sweep is forbidden if the zero-coordinate endpoint fails.

The paired evaluation uses stateless Gaussian sources keyed by environment and replan index. Thus
control and candidate have exactly equal arm-source tensors even when their trajectories terminate at
different times, preventing global RNG consumption from masquerading as a method effect.
