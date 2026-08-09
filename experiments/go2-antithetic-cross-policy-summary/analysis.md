# H76 Result: Policy-Level Statistical Summary

H76 retrospectively summarizes the four completed official Go2 training seeds,
treating each policy as one equally weighted independent unit. It does not use
the thousands of nested evaluation episodes as independent evidence for
training-seed generalization.

| Estimand | Seed effects | Mean (SD) | Policy-level 95% t interval | Two-sided t p | One-sided sign p |
|---|---|---:|---:|---:|---:|
| Antithetic32 - random64, equal 64 NFE | 0.981, 0.903, 1.008, 0.888 | 0.945 (0.059) | [0.852, 1.038] | 0.000066 | 0.0625 |
| Antithetic64 - IID-pair64, equal 128 NFE | 0.779, 0.357, 0.574, 0.551 | 0.566 (0.173) | [0.291, 0.840] | 0.0072 | 0.0625 |
| Antithetic32 - zero32 | -0.016, 0.035, -0.046, -0.067 | -0.023 (0.044) | [-0.094, 0.047] | 0.3661 | 0.9375 |

Under a normal random-seed-effects model, the first two policy-level intervals
exclude zero and estimate a roughly `+0.94` equal-total-NFE recovery over one
random endpoint and a `+0.57` symmetry-specific advantage over IID averaging.
The zero32 comparison remains centered near zero and its interval crosses zero,
confirming that antithetic projection is not the best deterministic deployment
rule.

With only four policies, normality cannot be checked reliably. The exact sign
test is therefore reported as a conservative distribution-free complement:
four positive effects out of four gives one-sided `p=1/16=0.0625`. The result is
strong magnitude and consistency evidence, but four training seeds are still a
small generalization sample. H67/H69/H70/H73 retain their paired episode-level
bootstrap intervals as within-policy evidence.

The analysis is fully deterministic and reads its values from archived result
JSON rather than manually transcribed tables. It used NumPy 2.4.3 and SciPy
1.17.1.
