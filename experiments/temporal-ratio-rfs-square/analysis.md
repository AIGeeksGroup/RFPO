# Analysis: H49 Temporally Factorized RFS Ratios

## Mechanism Result

The unit test established per-parameter equality of the temporal and joint PPO gradients at the
on-policy point. The one-iteration server screen exactly reproduced H48's first collection with 323
transitions and 4/4 completed successes. Temporal ratio clipping was 15.01%, versus the frozen
joint-ratio control's 72.58%, a reduction of 57.56 percentage points. The base remained bitwise
frozen, both branches updated, and all metrics were finite. H49 therefore passed the locked mechanism
gate and authorized the conditional reward run.

Across the five-iteration run, temporal clip fractions were 15.0%, 17.2%, 22.5%, 24.9%, and 26.8%,
remaining far below H48's 72.6-76.0%. Final latent/residual mean-head deltas were 0.606/0.512,
larger than H48's 0.418/0.355, consistent with less update deactivation.

## Balanced Reward Result

| Condition | Mean | Sampled | Pooled |
|---|---:|---:|---:|
| frozen released control | 10/20 | 10/20 | 20/40 |
| temporal-ratio RFS | 5/20 | 6/20 | 11/40 |
| delta | -5/20 | -4/20 | -9/40 |

H49 fails all reward gates. Reducing clipping did not rescue RFS; it amplified modulation updates
and worsened both modes beyond H48's already negative 8/20 and 8/20. The evidence rejects joint
ratio clipping as the main cause of H48's failure and instead indicates that the learned modulation
direction itself is harmful in this short Square regime.

Stop without temporal grouping, clip/LR/epoch changes, branch weighting, another seed, checkpoint
selection, longer training, or confirmation. Close RFS-style modulation and return to methods with a
different reward-improvement mechanism.
