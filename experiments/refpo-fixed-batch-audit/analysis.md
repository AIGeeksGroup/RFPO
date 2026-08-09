# Analysis: ReFPO Fixed-Batch Regularization Audit

## Outcome

H61 is refuted. The paired audit was valid, but `lambda=0.04` failed the locked ratio-drift and clip-fraction gates in both held-out batches. No reward screen or coefficient, seed, or epoch variant is authorized.

The control and candidate used the same 223 eligible chunks and exact pairing fingerprint `687734f74cfd006e475342786ebfe083ae1535804a29c5f85b5e9ff46b8ef4ca`. All reported quantities were finite, and the candidate added no actor forward evaluations. Before updating, the weighted CFM gradient norm was only 0.2815% and 0.2499% of the GAE gradient norm in batches 0 and 1.

## Locked Gates at Epoch 10

| Batch | Metric | Control | Candidate | Candidate / control or change | Gate | Result |
|---:|---|---:|---:|---:|---|---|
| 0 | Median absolute log ratio | 0.121859 | 0.117377 | 96.32% | at most 80% | Fail |
| 0 | Ratio standard deviation | 0.255700 | 0.232127 | 90.78% | at most 80% | Fail |
| 0 | Clip fraction | 91.02% | 91.02% | 0.00 pp | at least -10 pp | Fail |
| 0 | Surrogate gain | 0.167327 | 0.171748 | 102.64% | positive and at least 50% | Pass |
| 0 | Outcome-gradient cosine | 0.329936 | 0.435117 | +0.105181 | no lower than -0.02 | Pass |
| 1 | Median absolute log ratio | 0.140664 | 0.127384 | 90.56% | at most 80% | Fail |
| 1 | Ratio standard deviation | 0.241157 | 0.214276 | 88.85% | at most 80% | Fail |
| 1 | Clip fraction | 93.36% | 91.41% | -1.95 pp | at least -10 pp | Fail |
| 1 | Surrogate gain | 0.176984 | 0.164941 | 93.20% | positive and at least 50% | Pass |
| 1 | Outcome-gradient cosine | 0.442348 | 0.412627 | -0.029721 | no lower than -0.02 | Fail |

## Interpretation

The published coefficient produced a small ratio-stabilizing tendency without making the update inert: surrogate retention exceeded 93% in both batches. The effect was nevertheless much smaller than the preregistered threshold, did not materially reduce the already extreme clip fraction, and slightly worsened outcome-gradient preservation in one batch. This is insufficient evidence that current-CFM anchoring addresses the FPO++ update failure observed in this benchmark.

The result closes H61 at the tested paper coefficient. Per protocol, do not run a reward screen and do not tune the coefficient, seed, number of epochs, or audit population.

## Artifacts

- `results/control.json`
- `results/candidate.json`
- `results/analysis.json`
- `logs/can_h61_refpo_control.log`
- `logs/can_h61_refpo_candidate.log`
- `logs/can_h61_refpo_analysis.log`
- `logs/can_h61_refpo_control_osmesa_path_failure.log`

Local and remote SHA-256 checksums match for all seven artifacts. The initial control launch failed before training because the wrapper omitted the rootless OSMesa library path; that infrastructure failure is preserved separately and was corrected before the valid paired run.
