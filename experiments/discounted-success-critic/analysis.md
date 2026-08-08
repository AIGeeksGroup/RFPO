# Interim Analysis: Cross-Iteration Discounted-Success Critic Audit

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

## Decision

H19 remains active. Run the locked seed-20260818, 64-chunk formal audit with the already validated
implementation. Judge only that formal result against the preregistered gates.
