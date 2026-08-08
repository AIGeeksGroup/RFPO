# Conditional Reflow Pilot Analysis

## Matched training

Both students started from the released Can EMA checkpoint with fresh optimizers and used seed
20260808, batch size 64, and 100 updates. The control used ordinary conditional flow matching on
dataset actions. Reflow used exact Gaussian-source / frozen-64-step-teacher-endpoint pairs.

## Geometry gate

Evaluation used 128 fixed observation/source pairs.

| Metric | CFM control | Reflow | Relative change |
|---|---:|---:|---:|
| 64-step normalized straightness error | 0.00286483 | 0.00220895 | -22.9% |
| 64-step path-length ratio | 1.009963 | 1.007790 | lower |
| 4-step endpoint MSE vs. 64-step endpoint | 0.00068647 | 0.00049552 | -27.8% |
| 2-step endpoint MSE vs. 64-step endpoint | 0.00257312 | 0.00191850 | -25.4% |
| 1-step endpoint MSE vs. 64-step endpoint | 0.00971683 | 0.00739813 | -23.9% |

The 20% straightness threshold was passed. Diversity did not collapse: mean element standard
deviation was 0.293177 for the control and 0.294709 for reflow; mean pairwise distance was 6.160114
and 6.179751, respectively.

## Rollout screen

Each cell used 20 Can episodes, 10 parallel environments, EMA weights, Gaussian sources, and seed
20260808.

| Euler steps | Sampling mode | CFM control | Reflow | Difference |
|---:|---|---:|---:|---:|
| 10 | zero | 11/20 (55%) | 14/20 (70%) | +3 successes |
| 10 | random | 1/20 (5%) | 2/20 (10%) | +1 success |
| 4 | zero | 14/20 (70%) | 15/20 (75%) | +1 success |
| 4 | random | 2/20 (10%) | 1/20 (5%) | -1 success |

The rollout gate passed. Ten-step zero success was 15 percentage points higher rather than more
than 15 points lower; four-step zero success improved by the required one success; pooled random
success was unchanged at 3/40. The screen is too small to establish a stable reward improvement.
An independent evaluation seed is required before extending training or integrating reflow into
online FPO++.

## Independent environment-seed confirmation

The same checkpoints were evaluated with seed 20260809 under the same 20-episode-per-cell protocol.

| Euler steps | Sampling mode | CFM control | Reflow | Difference |
|---:|---|---:|---:|---:|
| 10 | zero | 14/20 (70%) | 17/20 (85%) | +3 successes |
| 10 | random | 3/20 (15%) | 3/20 (15%) | equal |
| 4 | zero | 15/20 (75%) | 14/20 (70%) | -1 success |
| 4 | random | 1/20 (5%) | 2/20 (10%) | +1 success |

Across both environment seeds, 10-step zero success was 25/40 (62.5%) for the control and 31/40
(77.5%) for reflow. Four-step zero success was exactly tied at 29/40 (72.5%). Random success across
both step counts was 7/80 for the control and 8/80 for reflow. Thus the official-step improvement
direction replicated within this trained checkpoint, while the claimed four-step improvement did
not; the defensible low-step conclusion is preservation, not improvement. A second matched training
seed is the next gate for a stable benchmark-improvement claim.

Remote evidence:

- `~/workspace/outputs/fpo-control/results/can_reflow_control100_seed20260808/geometry.json`
- `~/workspace/outputs/fpo-control/results/can_reflow_stage1_100_seed20260808/geometry.json`
- `~/workspace/outputs/fpo-control/logs/can_reflow_control100_eval_driver_retry.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_stage1_100_eval_driver_retry.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_control100_confirm_seed20260809_driver.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_stage1_100_confirm_seed20260809_driver.log`
