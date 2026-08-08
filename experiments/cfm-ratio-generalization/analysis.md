# Analysis: Stored-versus-Held-Out CFM Ratio Generalization

## Result

H16 is refuted. The locked two-iteration step-6000 audit completed with finite, nonzero gradients
and 1,754 positive-advantage chunks available. The seeded audit used 64 chunks. Held-out MC8 draws
were somewhat closer to the behavior policy, but the effect was too small and their replay-gradient
direction was not stable enough to support resampling the CFM variables every PPO epoch.

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Held-out active-fraction gain | at least 15 points | 6.45 points | fail |
| Held-out/stored median absolute log-ratio | at most 0.8 | 0.77843 | pass |
| Held-out pre/post gradient cosine | at least 0.8 | 0.74273 | fail |
| Held-out ratio ESS | at least 80% | 99.63% | pass |

After the official update, 54.69% of held-out positive ratios and 48.24% of stored positive ratios
remained active. Their median absolute log-ratios were 0.04237 and 0.05442, respectively. The
held-out gradient norm changed from 9.140 to 6.672; the stored gradient norm changed from 9.540 to
6.212.

An earlier 8-environment smoke gave the same failure pattern: active-fraction gain was only 2.34
points and held-out gradient cosine was 0.401, while the log-ratio and ESS gates passed. The formal
result is therefore not an isolated inversion of the smoke result.

## Interpretation

Reusing fixed MC8 variables does create a measurable stored-versus-held-out difference, but fixed
draw overfitting is not large enough to explain FPO++ clipping. Fresh draws retain only 6.45 points
more active positive terms, less than half the preregistered minimum, and their post-update gradient
still rotates beyond the stability gate. Epoch-level resampling would therefore change a noisy
objective without establishing that it preserves useful update direction.

Stop without implementing epoch-resampled behavior losses. Do not tune epoch count, MC count,
learning rate, clipping, source distribution, or the held-out seed around this result.

## Runtime Note

The audit used OSMesa software rendering because physical GPU 1's EGL renderer had developed a
repeatable multi-step stall. With a matched MuJoCo state, OSMesa versus EGL reset observations had
mean absolute pixel difference 0.00812 and identical low-dimensional state. The step-6000 policy's
deterministic first action differed by 0.00322 on average and 0.00740 at most (action-vector L2
difference 0.01025 versus reference L2 1.02928). This is sufficient for the ratio-mechanism audit,
but any final reward benchmark should be reconfirmed with healthy EGL rendering.

Remote evidence:

- `~/workspace/outputs/fpo-control/results/can_step6000_cfmratio_audit_osmesa_formal_seed20260815/cfm_ratio_generalization_audit.json`
- `~/workspace/outputs/fpo-control/logs/can_step6000_cfmratio_audit_osmesa_formal_seed20260815.log`
- `~/workspace/outputs/fpo-control/logs/can_step6000_cfmratio_audit_osmesa_smoke_seed20260815.log`
