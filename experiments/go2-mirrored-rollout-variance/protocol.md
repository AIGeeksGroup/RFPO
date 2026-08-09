# H77: Mirrored Closed-Loop Rollout Variance

## Question

Does the approximate odd symmetry of the final Go2 flow policy survive feedback
dynamics strongly enough that separately executed `z/-z` trajectories form a
lower-variance return pair than two IID-source trajectories?

## Motivation

H75 establishes nearly opposite immediate action displacements, but H74 shows
that averaging actions is not a cross-task improvement over IID averaging or
zero inference. A distinct use of the symmetry is to preserve both exploratory
trajectories and use their paired returns as a training-time control variate.
That idea is viable only if closed-loop return deviations retain useful
negative covariance after the trajectories diverge.

## Fixed evaluation

- Task: `Isaac-Velocity-Flat-Unitree-Go2-v0`
- Fixed official seed-42 final checkpoint used by H66/H67/H70
- Euler-32 policy integration
- 256 environments and exactly one completed episode per environment per mode
- Evaluation seed `20261650`
- Primary source seed `20261651`
- Secondary IID source seed `20261652`
- Bootstrap seed `20261653`, 20,000 paired resamples
- Four independently executed modes from identical initial states:
  - `zero32`: zero source
  - `positive32`: primary source sequence `z`
  - `negative32`: the bitwise sign reversal `-z`
  - `iid32`: independent source sequence `w`

No action averaging occurs in this experiment. Each source drives its own
closed-loop trajectory. All episode returns, lengths, initial-observation
hashes, base-source hashes, signed-source hashes, and finiteness flags are
archived.

## Estimands

For each paired initial environment, define source-induced deviations from the
zero trajectory:

`d+ = R(z) - R(0)`, `d- = R(-z) - R(0)`, and `dw = R(w) - R(0)`.

Compare:

- covariance and correlation of `(d+, d-)` versus `(d+, dw)`;
- centered sample variance of the mirrored pair residual `(d+ + d-) / 2`
  versus the IID pair residual `(d+ + dw) / 2`;
- the paired bootstrap distribution of their variance ratio.

The training-seed policy remains the unit of later generalization; this screen
tests only whether the prerequisite covariance mechanism exists on one fixed
policy.

## Gates

H77 passes only if all of the following hold:

1. all 1,024 episodes complete with finite actions and returns, initial hashes
   match, primary base-source hashes match, and the negative source is an exact
   sign reversal;
2. mirrored deviation covariance is strictly lower than IID deviation
   covariance;
3. mirrored pair-residual variance is at most 80% of IID pair-residual
   variance; and
4. the paired-bootstrap 95% upper bound of the variance ratio is below 1.0.

A pass authorizes only a separately registered fixed-batch FPO++ gradient-
variance audit. A failure stops mirrored-rollout control variates without
training, source scaling, seed, checkpoint, step-count, task, or gate tuning.
This experiment does not reopen Spot/H1/G1 expansion.
