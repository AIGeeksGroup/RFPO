# Analysis: Reset CFM Validity Mask Per Rollout

## Mechanism

The released behavior accumulates stale invalid positions across independent rollout iterations.
In the matched control, the valid-CFM action fraction declined monotonically from 98.11% to 95.81%
over five iterations. Resetting the mask kept it between 97.99% and 98.09% in every iteration. The
one-update smoke also completed with finite losses and 98.07% valid actions, so the mechanism and
implementation gates passed.

| Iteration | Control valid CFM | Reset valid CFM | Control success | Reset success |
|---:|---:|---:|---:|---:|
| 1 (critic only) | 98.11% | 98.08% | 12/151 (7.95%) | 8/151 (5.30%) |
| 2 | 97.35% | 98.07% | 15/151 (9.93%) | 15/151 (9.93%) |
| 3 | 96.70% | 97.99% | 13/151 (8.61%) | 12/152 (7.89%) |
| 4 | 96.14% | 98.09% | 12/151 (7.95%) | 13/150 (8.67%) |
| 5 | 95.81% | 98.07% | 13/151 (8.61%) | 16/151 (10.60%) |
| Iterations 2-5 pooled | - | - | 53/604 (8.77%) | 56/604 (9.27%) |

## Reward Gate

The reset improved pooled actor-update collection success by only 0.50 percentage points, below the
preregistered +2-point gate and statistically indistinguishable (two-sided Fisher `p=0.841`). Final
zero-sampling evaluation was 41/50 (82%) for reset versus 42/50 (84%) for control. Random-sampling
evaluation was tied at 5/50 (10%). Training remained numerically stable.

## Decision

H6 is refuted as an early reward-improvement claim. Retain the optional correctness fix and its
diagnostic metric, but do not run a confirmation seed or longer job based on this pilot. The result
shows that losing roughly 2.3 additional points of valid samples over five iterations is measurable
but not the dominant short-budget Can bottleneck.

## Artifacts

- `~/workspace/outputs/fpo-control/results/can_maskreset_smoke_seed0`
- `~/workspace/outputs/fpo-control/results/can_maskreset_control_5iter_seed0`
- `~/workspace/outputs/fpo-control/results/can_maskreset_candidate_5iter_seed0`
- `~/workspace/outputs/fpo-control/logs/can_maskreset_smoke_seed0.log`
- `~/workspace/outputs/fpo-control/logs/can_maskreset_control_5iter_seed0.log`
- `~/workspace/outputs/fpo-control/logs/can_maskreset_candidate_5iter_seed0.log`

