# Analysis: Held-Out Ratio Early Stopping

## Outcome

H18 is refuted under the locked protocol. The step-6000 Can audit used seed 20260817, 128
positive fully valid chunks available, and two disjoint held-out MC8 batches of 32. The pooled active
positive-ratio fraction was 85.55% after epoch 1 and 72.85% after epoch 2, so the locked 0.80 rollback
rule selected epoch 1.

| Batch | Epoch-1 gradient cosine | Epoch-10 gradient cosine | Epoch-1 surrogate gain | Epoch-10 surrogate gain | Retained gain |
|---|---:|---:|---:|---:|---:|
| 0 | 0.75446 | 0.20894 | -0.00742 | +0.01245 | wrong sign |
| 1 | 0.26305 | 0.08007 | +0.00196 | +0.03597 | 5.45% |

The selected epoch was earlier than epoch 10 and improved gradient cosine over epoch 10 by more than
0.10 in both batches. It failed the absolute 0.85 gradient-cosine gate in both batches. It also
failed the surrogate-retention gate: batch 0 moved in the wrong direction, and batch 1 retained only
5.45% of the full-update gain instead of the required 50%.

## Interpretation

Repeated epochs worsen FPO++ ratio clipping and gradient rotation, but the harmful direction change
is already substantial after the first actor epoch. The active-ratio fraction is therefore too weak
a sensor: 85.55% of positive ratios remained active while one batch's gradient cosine had already
fallen to 0.263. Adjusting the 0.80 threshold cannot recover an acceptable pre-update snapshot because
the only earlier snapshot is epoch 0, which performs no actor learning.

## Decision

Stop H18 without implementing rollback early stopping or running a five-update reward screen. Do not
tune the active-ratio threshold or replace it with a fixed epoch count. The next candidate should
improve the advantage or critic signal that determines the first actor step, rather than constrain
later movement along an already unstable direction.
