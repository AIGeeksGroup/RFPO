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

## Independent training-seed confirmation

A second matched control/reflow pair used training seed 20260809 with all other training settings
unchanged. Reflow again passed the 128-pair geometry gate: 64-step normalized straightness error
fell from 0.00282171 to 0.00219042 (-22.4%), path-length ratio fell from 1.010476 to 1.008058, and
four-step endpoint MSE fell from 0.00058784 to 0.00044354 (-24.5%). The 64-step mean element
standard deviation was 0.288502 for the control and 0.288833 for reflow.

One matched rollout seed (20260808) produced:

| Euler steps | Sampling mode | CFM control | Reflow | Difference |
|---:|---|---:|---:|---:|
| 10 | zero | 15/20 (75%) | 16/20 (80%) | +1 success |
| 10 | random | 5/20 (25%) | 3/20 (15%) | -2 successes |
| 4 | zero | 14/20 (70%) | 17/20 (85%) | +3 successes |
| 4 | random | 1/20 (5%) | 2/20 (10%) | +1 success |

Across all three matched rollout comparisons from two training seeds, 10-step zero success was 40/60
for controls and 47/60 for reflow; four-step zero success was 43/60 and 46/60. Random success pooled
over both step counts was exactly 13/120 for each method. The geometry mechanism and zero-sampling
task advantage now both replicate across training seeds. The next experiment is an official-scale
200-episode, 50-environment, 10-step confirmation using a new environment seed.

## Official-scale confirmation

The second training-seed pair was evaluated at 10 Euler steps with seed 20260810, 200 episodes, and
50 environments per cell.

| Sampling mode | CFM control | Reflow | Difference |
|---|---:|---:|---:|
| zero | 147/200 (73.5%) | 150/200 (75.0%) | +3 successes (+1.5 pp) |
| random | 24/200 (12.0%) | 23/200 (11.5%) | -1 success (-0.5 pp) |

The zero result did not meet the preregistered +10-success (+5 pp) gate. Its approximate unpooled
95% difference interval was [-7.1 pp, +10.1 pp], and a two-sided Fisher exact test gave p=0.819.
Random sampling passed the non-inferiority gate (Fisher p=1.0). The large evaluation therefore
supports performance preservation but not a stable benchmark improvement. Do not proceed to online
FPO++ integration based on pure one-stage reflow.

Remote evidence:

- `~/workspace/outputs/fpo-control/results/can_reflow_control100_seed20260808/geometry.json`
- `~/workspace/outputs/fpo-control/results/can_reflow_stage1_100_seed20260808/geometry.json`
- `~/workspace/outputs/fpo-control/logs/can_reflow_control100_eval_driver_retry.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_stage1_100_eval_driver_retry.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_control100_confirm_seed20260809_driver.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_stage1_100_confirm_seed20260809_driver.log`
- `~/workspace/outputs/fpo-control/results/can_reflow_control100_trainseed20260809/geometry.json`
- `~/workspace/outputs/fpo-control/results/can_reflow_stage1_100_trainseed20260809/geometry.json`
- `~/workspace/outputs/fpo-control/logs/can_reflow_control100_trainseed20260809_eval20260808_driver.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_stage1_100_trainseed20260809_eval20260808_driver.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_control100_trainseed20260809_official200_eval20260810_driver.log`
- `~/workspace/outputs/fpo-control/logs/can_reflow_stage1_100_trainseed20260809_official200_eval20260810_driver.log`
