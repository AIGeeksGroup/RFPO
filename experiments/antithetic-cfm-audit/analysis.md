# Analysis: Antithetic CFM Gradient Estimator

## Result

H14 is refuted under its locked gate. All gradients were finite and nonzero, and both modes used
eight CFM samples at identical model-evaluation cost. Joint antithetic sampling reduced normalized
gradient MSE in both fixed-data batches, but the mode-average gradient directions were not similar
enough to satisfy the preregistered no-shift requirement.

| Batch | Metric | IID MC8 | Joint antithetic MC8 | Candidate change |
|---:|---|---:|---:|---:|
| 0 | Normalized gradient MSE | 0.48808 | 0.31315 | -35.84% |
| 0 | Mean gradient cosine | 0.82523 | 0.87606 | +0.05083 |
| 0 | Scalar-loss CV | 0.04250 | 0.04052 | -4.65% |
| 0 | Cross-mode average cosine | - | - | 0.96366 |
| 1 | Normalized gradient MSE | 0.46826 | 0.35665 | -23.84% |
| 1 | Mean gradient cosine | 0.82742 | 0.86147 | +0.03405 |
| 1 | Scalar-loss CV | 0.02120 | 0.01936 | -8.68% |
| 1 | Cross-mode average cosine | - | - | 0.95465 |

The candidate passed the per-batch variance, within-mode cosine, and scalar-loss CV gates. It failed
the required cross-mode average-gradient cosine of at least 0.99 in both batches. The finite-repeat
average may include estimation noise, but increasing repeats after observing the result would change
the locked audit. The result therefore cannot justify an online policy update.

## Interpretation

Joint pairing is a promising variance-reduction device in isolation, but this audit did not establish
that its finite-MC update estimates preserve the IID gradient direction closely enough for FPO++.
Stop without online training and do not test epsilon-only pairing, time-only pairing, other sample
counts, or quasi-Monte Carlo variants. Move to successful standard-Gaussian on-policy replay, whose
first gate must audit stale-policy ratios and update bias before any training integration.

Remote evidence:

- `~/workspace/outputs/fpo-control/results/antithetic_cfm_audit_seed20260813.json`
- `~/workspace/outputs/fpo-control/logs/antithetic_cfm_audit_seed20260813.log`
