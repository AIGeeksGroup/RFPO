# Analysis: Advantage-Stratified CFM Monte Carlo Audit

## Outcome

H17 is refuted under the locked protocol. The formal iteration-2 audit used the step-6000 Can
initialization, seed 20260816, two disjoint 32-chunk batches, eight repeats, and independent MC64
references. There were 147 fully valid chunks available, of which 64 were selected before ranking
within each batch.

| Batch | Uniform MC8 MSE | Stratified MC12/4 MSE | Relative change | Uniform cosine | Stratified cosine | Cross-method mean cosine |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.38506 | 0.24829 | -35.52% | 0.83600 | 0.89257 | 0.97683 |
| 1 | 0.42016 | 0.28554 | -32.04% | 0.81971 | 0.87454 | 0.97373 |

All gradients and metrics were finite and nonzero. The candidate passed the first two gates in both
batches: normalized gradient MSE fell by more than 15%, and mean cosine to the MC64 reference
improved. It failed the third gate in both batches because the candidate-versus-control
repeat-average gradient cosine was below 0.99.

## Interpretation

Allocating samples by absolute advantage is an effective variance-reduction device at fixed CFM
sample count, but the locked 12/4 allocation does not preserve the uniform-MC8 average update
direction closely enough. This mirrors the antithetic audit: estimator variance can be reduced while
the finite-sample intervention also changes the realized policy-gradient direction. The result does
not justify an online update, and the split, sample counts, advantage transform, reference size, and
seed will not be tuned after observing the audit.

The audit initially exceeded the frozen ViT attention launch limit because MC64 redundantly expanded
32 images to 2,048 images. Encoding each base observation once and repeating its frozen embedding
removed that implementation bottleneck without changing the locked sample allocation or actor
gradient target. The completed run then used 147 valid chunks and released GPU 1 cleanly.

## Decision

Stop H17 without implementing online 12/4 allocation or running the five-update reward screen.
Future candidates must change the quality of the learning signal or objective rather than merely
reallocate fixed-draw Monte Carlo effort.
