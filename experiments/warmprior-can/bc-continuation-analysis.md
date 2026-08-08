# H3b Analysis: Matched WarmPrior BC Continuation on Can

Date: 2026-08-08

Both conditions resumed the released `95j3noe4_step_1000` checkpoint and optimizer for 100 updates
with seed 20260808 and batch size 64. Both saved step 1100 with finite losses and gradients. WP-Past
used valid eight-action histories for 95.3% to 98.4% of samples at logged steps; remaining samples
correctly used the episode-boundary Gaussian fallback.

| Training source | Evaluation source | Sampling | Successes | Episodes | Success rate |
|---|---|---|---:|---:|---:|
| Gaussian | Gaussian | Zero | 14 | 20 | 70% |
| Gaussian | Gaussian | Random | 0 | 20 | 0% |
| WP-Past, sigma 0.5 | WP-Past, sigma 0.5 | Zero | 0 | 20 | 0% |
| WP-Past, sigma 0.5 | WP-Past, sigma 0.5 | Random | 0 | 20 | 0% |

The final logged Gaussian loss/gradient norm was 7.1005/17.0833. The final logged WP-Past
loss/gradient norm was 6.6613/16.0977. These losses are not directly comparable because their source
distributions and regression targets differ.

## Decision

Stop H3. WP-Past did not improve the primary random-sampling metric and reduced deterministic
success by 70 percentage points relative to the matched continuation, violating both preregistered
gates. The earlier inference-only failure therefore was not fixed by a short faithful training-time
adaptation. This result does not rule out from-scratch WarmPrior training, but it is sufficient to
reject additional compute on this branch under the current improvement-first policy. Proceed to
one-stage conditional reflow.

Remote checkpoints, full training logs, and evaluation summaries are retained under
`~/workspace/outputs/fpo-control/{results,logs}` with the `can_bc100_*_seed20260808` tags.

