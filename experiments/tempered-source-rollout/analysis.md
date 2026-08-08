# Analysis: Tempered-Source Rollout Bridge for Can

## Result

The tempered subset supplied high-success trajectories and improved the official full-noise subset
slightly, but it failed both the magnitude and final-random gates. Over actor-update iterations 2
through 5, full-noise success was 44/484 (9.09%) versus the control's 53/604 (8.77%), a gain of only
0.32 percentage points (two-sided Fisher `p=0.915`) and below the required 10.77%.

| Iteration | Scale-0.5 success | Scale-1 success | Overall success |
|---:|---:|---:|---:|
| 1 (critic only) | 24/38 (63.16%) | 12/120 (10.00%) | 36/158 (22.78%) |
| 2 | 21/37 (56.76%) | 19/124 (15.32%) | 40/161 (24.84%) |
| 3 | 26/38 (68.42%) | 7/120 (5.83%) | 33/158 (20.89%) |
| 4 | 18/35 (51.43%) | 9/120 (7.50%) | 27/155 (17.42%) |
| 5 | 21/37 (56.76%) | 9/120 (7.50%) | 30/157 (19.11%) |
| Iterations 2-5 pooled | 86/147 (58.50%) | 44/484 (9.09%) | 130/631 (20.60%) |

The tempered subset passed its mechanism gate by a wide margin. Final evaluation was 42/50 (84%)
zero and 3/50 (6%) random, versus 42/50 (84%) and 5/50 (10%) for the control. Training losses stayed
finite. Thus continuous source overlap recovered the H8 full-noise deficit, but did not produce a
stable useful improvement and degraded final random evaluation.

## Interpretation

A continuously overlapping lower-variance source transfers better than a delta-zero source: the
full-noise collection metric moved from H8's 7.08% to 9.09%. The effect is still too small and
inconsistent to justify a scale or mixture sweep. The cumulative valid-CFM fraction also fell to
93.91% by iteration 5, lower than the official control's 95.81%, because the high-success guided
subset creates more episode boundaries. This motivates one explicit interaction test with per-rollout
mask reset, not further source tuning.

## Artifacts

- `~/workspace/outputs/fpo-control/results/can_tempered05_smoke_seed0`
- `~/workspace/outputs/fpo-control/results/can_tempered05_5iter_seed0`
- `~/workspace/outputs/fpo-control/logs/can_tempered05_smoke_seed0.log`
- `~/workspace/outputs/fpo-control/logs/can_tempered05_5iter_seed0.log`
