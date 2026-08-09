# Analysis: H48 Residual Flow Steering on Square

## Validity

The one-iteration smoke and five-iteration formal run completed. The formal run produced 323-327
macro transitions per iteration, finite losses and gradient norms, and a final checkpoint. The frozen
base actor passed the bitwise state check after every update. At the final iteration, latent and
residual mean-head deltas from initialization were 0.418 and 0.355, so both modulation branches
were active.

## Training Diagnostics

Collection successes across the five iterations were `4/4`, `2/15`, `4/16`, `2/16`, and `6/15`.
These are censored within-rollout diagnostics and were not used for checkpoint selection. The exact
joint 224-dimensional PPO ratio had a high clip fraction in every iteration: 72.6%, 76.0%, 75.0%,
75.1%, and 76.0%. This does not invalidate H48's locked implementation, but it indicates that most
joint-ratio samples stop contributing normally after several PPO epochs.

## Balanced Reward Result

| Condition | Mean / zero | Sampled / Gaussian | Pooled |
|---|---:|---:|---:|
| released control | 10/20 | 10/20 | 20/40 |
| RFS candidate | 8/20 | 8/20 | 16/40 |
| delta | -2/20 | -2/20 | -4/40 |

H48 fails all locked reward gates: pooled gain is `-4` rather than at least `+3`, sampled gain is
`-2` rather than at least `+2`, and mean-mode loss is `-2`, exceeding the allowed one-success loss.
RFS is refuted at this budget and implementation. Do not run another seed, select an intermediate
checkpoint, extend training, or tune residual scale, latent scale, widths, learning rates, or epochs.

The high joint-ratio clipping is a method-level observation that motivates a separately registered
temporal-factorization hypothesis; it does not authorize an H48 neighbor or reinterpret this result.
