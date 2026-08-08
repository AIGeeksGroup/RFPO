# Analysis: Tempered-Source Rollouts with Validity-Mask Reset

## Result

H10 is refuted. The reset behaved as intended: valid CFM actions stayed between 97.88% and
98.08% over all five iterations, instead of falling to 93.91% by iteration 5 in the no-reset H9
run. This recovered the samples lost to the persistent invalid-action mask, but did not improve the
predeclared reward metric.

| Iteration | Guided success | Full-noise success | Valid CFM |
|---:|---:|---:|---:|
| 1 | 18/36 (50.00%) | 6/120 (5.00%) | 97.88% |
| 2 | 16/34 (47.06%) | 8/120 (6.67%) | 98.08% |
| 3 | 17/35 (48.57%) | 15/121 (12.40%) | 97.97% |
| 4 | 16/34 (47.06%) | 4/120 (3.33%) | 98.01% |
| 5 | 15/35 (42.86%) | 14/120 (11.67%) | 97.96% |

The confirmatory full-noise metric over iterations 2-5 was 41/481 (8.52%). This is below both the
official same-code control's 53/604 (8.77%; two-sided Fisher `p=0.914`) and H9 without reset's
44/484 (9.09%; `p=0.821`), and far below the required 10.77% gate. The guided subset was 64/138
(46.38%) over the same iterations. Final 50-episode evaluation was 43/50 (86%) for zero sampling and
3/50 (6%) for random sampling; random evaluation also failed its non-degradation gate.

## Interpretation

The cumulative validity-mask defect is real and the reset fully corrects its sample-retention
effect. However, invalid-mask data loss is not why successful tempered-source trajectories fail to
transfer into the official full-noise behavior. Increasing retained samples from this mixture did
not raise full-noise collection success and did not prevent the final random-policy degradation.

Stop this interaction without another seed, longer training, or a sweep over source scale, source
fraction, GAE lambda, or mask settings. The next step is an outer-loop literature and mechanism
review focused on changing how sparse successful trajectories affect the full-noise policy update,
not on collecting more overlapping guided trajectories.
