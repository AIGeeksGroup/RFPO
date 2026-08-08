# H28 BC-Anchor Conflict Projection Audit

## Result

H28 is refuted by the locked surrogate-retention gate. Both 32-chunk batches showed strong conflict
between the FPO++ gradient and the frozen BC velocity-field gradient, and projection substantially
reduced behavior drift. However, it retained only 34.0% and 46.1% of the control surrogate gain,
below the required 70% in both batches. No optimizer integration or reward screen was run.

| Metric | Batch 0 | Batch 1 | Gate | Pass |
|---|---:|---:|---:|:---:|
| RL/BC gradient cosine | -0.8121 | -0.7337 | conflict active | Yes |
| Candidate/control BC-MSE increase | 2.22% | 6.75% | <= 50% | Yes |
| Candidate/control surrogate gain | 34.00% | 46.13% | >= 70% | No |
| Candidate vs control post-gradient cosine | 1.000000 vs 1.000000 | 1.000000 vs 1.000000 | candidate >= control | Yes |
| Candidate gradient-norm retention | 58.35% | 67.95% | diagnostic | - |

All gradients and metrics were finite and nonzero. There were 216 eligible positive fully valid
chunks, of which the locked 64 were audited. The candidate reduced BC-MSE increase from
`2.277e-8` to `5.063e-10` in batch 0 and from `1.713e-8` to `1.156e-9` in batch 1. This confirms the
mechanism is active but shows that protecting the velocity field over generic Gaussian query points
removes too much reward-improving motion.

## Artifacts

- `results.json`: authoritative audit metrics
- `run.log`: complete remote log
- Remote result: `~/workspace/outputs/fpo-control/results/can_step6000_bc_anchor_pcgrad_audit_osmesa_formal2_seed20260827/`
- Remote log: `~/workspace/outputs/fpo-control/logs/can_step6000_bc_anchor_pcgrad_audit_osmesa_formal2_seed20260827.log`

The two earlier launches produced no H28 metrics: one stopped after iteration 1 because of an
environment-step budget mismatch, and one stopped before metric computation because of a helper
name collision. Neither informed the locked outcome gates.

