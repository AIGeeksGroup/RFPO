# H29 Zero-Endpoint Anchor Conflict Projection Audit

## Result

H29 passes every locked gate in both 32-chunk batches. Replacing the generic velocity-field anchor
with the deterministic zero-source action endpoint weakened RL/BC conflict enough to retain 73.7%
and 72.8% of control surrogate progress while reducing endpoint drift to 33.5% and 23.6% of control.

| Metric | Batch 0 | Batch 1 | Gate | Pass |
|---|---:|---:|---:|:---:|
| RL/endpoint gradient cosine | -0.5128 | -0.5223 | conflict active | Yes |
| Candidate/control endpoint-MSE increase | 33.45% | 23.57% | <= 50% | Yes |
| Candidate/control surrogate gain | 73.71% | 72.77% | >= 70% | Yes |
| Candidate vs control post-gradient cosine | equal at 0.9999998 | equal at 0.9999999 | candidate >= control | Yes |
| Candidate gradient-norm retention | 85.85% | 85.27% | diagnostic | - |

There were 190 eligible positive fully valid chunks and all 64 locked chunks were audited. All
gradients and metrics were finite and nonzero. The narrower endpoint objective therefore resolves
H28's overconstraint at the one-step mechanism level. This is sufficient to authorize a matched
short reward screen, but is not yet evidence of a benchmark improvement.

## Artifacts

- `results.json`: authoritative audit metrics
- `audit.log`: complete remote log
- Remote result: `~/workspace/outputs/fpo-control/results/can_step6000_zero_endpoint_pcgrad_audit_osmesa_seed20260827/`
- Remote log: `~/workspace/outputs/fpo-control/logs/can_step6000_zero_endpoint_pcgrad_audit_osmesa_seed20260827.log`

