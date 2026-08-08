# Mixed Reflow Analysis

## Training and geometry

The 50/50 endpoint mixture completed 100 updates from the released EMA checkpoint with seed 20260808.
Logged per-batch teacher fractions varied around the configured 0.5, the run remained finite, and the
final checkpoint was saved at step 99.

On 128 fixed pairs, mixed reflow reduced normalized 64-step straightness error from the control's
0.00286483 to 0.00254834 (-11.0%) and reduced four-step endpoint MSE from 0.00068647 to 0.00061628
(-10.2%). Mean pairwise action distance increased slightly from 6.1601 to 6.1907. The geometry gate
passed, retaining roughly half of pure reflow's straightening effect.

## Rollout screens

| Evaluation seed | Sampling mode | CFM control | Mixed reflow | Difference |
|---:|---|---:|---:|---:|
| 20260808 | zero | 11/20 | 15/20 | +4 |
| 20260808 | random | 1/20 | 5/20 | +4 |
| 20260809 | zero | 14/20 | 16/20 | +2 |
| 20260809 | random | 3/20 | 0/20 | -3 |

Zero-sampling improvement repeated, but the independent-seed random result violated the preregistered
non-collapse gate. H4 is stopped without a mixture-probability sweep or official-scale evaluation.
The mixed objective is not suitable for online FPO++ while its exploration behavior is unstable.

Remote evidence:

- `~/workspace/outputs/fpo-control/results/can_mixed_reflow50_100_seed20260808/geometry.json`
- `~/workspace/outputs/fpo-control/logs/can_mixed_reflow50_100_seed20260808.log`
- `~/workspace/outputs/fpo-control/logs/can_mixed_reflow50_100_seed20260808_screen_driver.log`
- `~/workspace/outputs/fpo-control/logs/can_mixed_reflow50_100_seed20260808_confirm20260809_driver.log`

