# H59 AdamW Extragradient Audit

## Result

H59 is refuted. The preregistered step-6000 Can audit completed with 249 eligible labeled chunks and
two fixed 32-chunk batches. Every validity check passed: the live actor was restored bitwise after
each virtual branch, the real actor optimizer retained zero state entries, all gradients and
displacements were finite and nonzero, and candidate/control displacement ratios were 0.948 and
0.943.

| Metric | Batch 0 control | Batch 0 candidate | Batch 1 control | Batch 1 candidate |
|---|---:|---:|---:|---:|
| Update cosine to outcome reference | 0.16871 | 0.09722 | 0.06067 | 0.09046 |
| Candidate update-cosine gain | - | -0.07149 | - | +0.02980 |
| Post-update outcome-gradient cosine | 0.50646 | 0.51432 | 0.73215 | 0.24383 |
| Candidate post-cosine gain | - | +0.00786 | - | -0.48832 |
| Held-out GAE surrogate gain | +0.11464 | +0.00395 | +0.10350 | -0.01290 |
| Displacement norm ratio | - | 0.94777 | - | 0.94331 |

Neither batch achieved the required `+0.05` update-direction gain. The post-update direction improved
by only `+0.0079` in batch 0 and deteriorated by `-0.4883` in batch 1. More decisively, the candidate
retained only 3.45% of batch 0's positive control surrogate gain and reversed its sign in batch 1.

The virtual control gradients were clipped from norms 14.35 and 12.48 to 5.0, while the lookahead
gradients had norms 3.03 and 3.81 and were not clipped. This confirms that the lookahead actively
changed the update rather than reproducing control, but it changed it toward a much weaker and
inconsistent learning signal.

## Decision

Stop H59 without online integration, a Square reward run, lookahead-step scaling, extra lookaheads,
another seed, or optimizer variants. Exact one-step extragradient does not correct the measured
FPO++ gradient-rotation bottleneck under the frozen protocol.

The experiment used OSMesa as a method audit. It is not an official EGL benchmark or a real-robot
result. Raw results are in `audit_seed20261050.json`; the remote log is
`~/workspace/outputs/fpo-control/logs/can_h59_adamw_extragradient_audit_osmesa_seed20261050.log`.
