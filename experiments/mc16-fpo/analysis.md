# Analysis: Increased CFM Monte Carlo Samples for Can FPO++

## Result

The matched seed-0 pilot completed without NaNs or loss spikes. Increasing the number of CFM samples
per action from 8 to 16 did not improve early reward learning. Over actor-update iterations 2 through
5, the 8-sample control collected 55 successes in 603 episodes (9.12%), while the 16-sample candidate
collected 48 in 604 (7.95%). The candidate therefore decreased the preregistered primary metric by
1.17 percentage points instead of improving it by at least two points.

| Iteration | MC8 success | MC16 success |
|---:|---:|---:|
| 1 (critic only) | 13/151 (8.61%) | 14/150 (9.33%) |
| 2 | 18/151 (11.92%) | 11/151 (7.28%) |
| 3 | 17/151 (11.26%) | 7/150 (4.67%) |
| 4 | 10/150 (6.67%) | 18/152 (11.84%) |
| 5 | 10/151 (6.62%) | 12/151 (7.95%) |
| Iterations 2-5 pooled | 55/603 (9.12%) | 48/604 (7.95%) |

Final 50-episode evaluation also showed no advantage. MC8 versus MC16 was 40/50 (80%) versus 39/50
(78%) under zero sampling and 7/50 (14%) versus 6/50 (12%) under random sampling. These evaluation
differences are small relative to binomial uncertainty, but both point estimates are unfavorable and
the random-sampling non-degradation gate also fails.

## Decision

H5 is refuted at its screening budget. Do not test 32 samples, add confirmation seeds, or start a
5M-step run. More Monte Carlo samples increase compute but do not address the dominant sparse-reward
problem in this Can setting.

## Artifacts

- `~/workspace/outputs/fpo-control/results/can_fpopp_mc8_5iter_seed0`
- `~/workspace/outputs/fpo-control/results/can_fpopp_mc16_5iter_seed0`
- `~/workspace/outputs/fpo-control/logs/can_fpopp_mc8_5iter_seed0_retry.log`
- `~/workspace/outputs/fpo-control/logs/can_fpopp_mc16_5iter_seed0.log`

