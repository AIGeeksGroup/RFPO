# Outer Loop Cycle 43: Decouple Arm Exploration from Gripper Decision

## Failure Boundary

H57 preserved exact unit-Gaussian arm sources and rescued two random-source failures by removing the
gripper latent, but missed its reward gate. Because the flow network couples all input and output
coordinates, zeroing the gripper latent also changes the predicted arm trajectory. The result cannot
identify whether the clue came from a more stable gripper decision or from incidental arm changes.
H56's binary post-processing is different: it quantizes one zero-source prediction rather than
substituting the observation-conditioned deterministic gripper trajectory into a random arm sample.

## Diverged Candidates

1. Random-source arm output with zero-source gripper output.
2. Zero-source arm output with random-source gripper output as a diagnostic inversion.
3. Average random and zero gripper outputs.
4. Select gripper output by agreement across source samples.
5. Train a separate deterministic gripper head.
6. Split the flow architecture into arm and gripper branches.

## Convergence

Select H58, candidate 1. It retains the complete random-source arm action bitwise at every replan and
replaces only the final output coordinate with the same-observation zero-source prediction. This is a
coefficient-free causal test of gripper decision stochasticity. It costs a second 10-step flow solve,
so it must first clear the same +3/20 reward gate before efficiency work or training is justified.
The H57 stateless control, environment seeds, and source keys are reused exactly.
