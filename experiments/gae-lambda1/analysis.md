# Analysis: Full-Lambda GAE for Sparse Can Success

## Result

Changing only GAE lambda from 0.99 to 1.0 did not improve early Can reward learning. The candidate's
collection success over actor-update iterations 2 through 5 was 40/602 (6.64%), compared with the
matched control's 53/604 (8.77%). This is a decrease of 2.13 percentage points and is not
statistically distinguishable in a two-sided Fisher exact test (`p=0.195`).

| Iteration | Control success | Lambda 1.0 success |
|---:|---:|---:|
| 1 (critic only) | 12/151 (7.95%) | 13/150 (8.67%) |
| 2 | 15/151 (9.93%) | 9/150 (6.00%) |
| 3 | 13/151 (8.61%) | 5/150 (3.33%) |
| 4 | 12/151 (7.95%) | 12/150 (8.00%) |
| 5 | 13/151 (8.61%) | 14/152 (9.21%) |
| Iterations 2-5 pooled | 53/604 (8.77%) | 40/602 (6.64%) |

Final 50-episode evaluation was 40/50 (80%) with zero sampling and 6/50 (12%) with random sampling,
versus 42/50 (84%) and 5/50 (10%) for the control. Policy and value losses remained finite. Thus the
stability and evaluation non-degradation checks passed, but the preregistered primary +2-point reward
gate failed decisively.

## Interpretation

At lambda 1.0, GAE becomes a Monte Carlo return-minus-value estimator. This removes lambda-induced
decay but increases variance and does not create additional successful trajectories. The lower pooled
success indicates that longer-horizon propagation alone is not the short-budget bottleneck in Can.
Do not run another seed, a longer job, or a lambda sweep for H7.

## Artifacts

- `~/workspace/outputs/fpo-control/results/can_gae1_5iter_seed0`
- `~/workspace/outputs/fpo-control/logs/can_gae1_5iter_seed0.log`
