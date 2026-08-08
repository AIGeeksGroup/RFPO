# Formal Analysis: Cross-Iteration Discounted-Success Critic Audit

## Pipeline Smoke

The two-environment smoke completed the full cross-iteration pipeline and produced finite value and
gradient metrics. Candidate value MSE was 0.24797 versus 0.12908 for control; the two four-chunk
gradient-cosine changes were -0.396 and +0.420. These small, deliberately underpowered batches are
pipeline evidence only and do not satisfy the locked 64-chunk protocol.

An initial interpretation incorrectly claimed that the side critic had not trained because a
standalone check used the `cosine` scheduler. The experiment actually uses
`lr_scheduler_name="constant"`; Diffusers' constant scheduler keeps the configured `1e-4` learning
rate from construction and ignores the warmup count. A paired parameter audit confirmed that two
identical MSE critics receive identical nonzero updates.

## Formal Result

The locked seed-20260818 audit completed on fresh iteration-2 data with 1,837 uncensored value
targets and two disjoint 32-chunk gradient batches. The candidate reduced value MSE from 0.117965 to
0.103595, a 12.18% improvement, and therefore passed the first gate. Its Spearman correlation fell
from 0.34416 to 0.21107, failing the second gate.

Gradient alignment was also inconsistent. In batch 0, cosine to the Monte Carlo outcome reference
fell from 0.82898 to 0.75583, a change of -0.07315. In batch 1, it rose from 0.34294 to 0.51110, a
change of +0.16816. Both candidate cosines were positive, but the required improvement of at least
0.05 did not hold in every batch, so the third gate failed.

## Decision

H19 is refuted. A bounded discounted-success BCE target improves scalar calibration in this audit but
loses rank information and does not reliably improve the actor-gradient direction. Per the locked
protocol, do not integrate the candidate critic, run the five-update reward screen, or tune its loss,
discount, censoring rule, architecture, or seed.
