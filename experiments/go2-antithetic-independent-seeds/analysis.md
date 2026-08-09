# H69 Result: Independent Go2 Training Seeds

## Outcome

H69 passed all preregistered primary gates. Official Go2 policies were trained
independently with seeds 43, 44, and 45 for 1,500 iterations and 4,096
environments. Their final checkpoint SHA-256 digests were distinct. Each policy
completed 512 paired episodes for zero, random, IID-pair, and antithetic modes,
for 6,144 total episodes.

| Training seed | Zero | Random | IID pair | Antithetic | Anti - IID | Anti - random |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 41.524 | 40.557 | 41.103 | 41.460 | +0.357 | +0.904 |
| 44 | 41.398 | 40.301 | 40.697 | 41.271 | +0.574 | +0.971 |
| 45 | 41.668 | 40.718 | 41.053 | 41.604 | +0.551 | +0.886 |

The stratified antithetic-minus-IID gain was +0.4942 with bootstrap 95% CI
[+0.3531, +0.6422]. All three seed-level point estimates were positive. The
stratified antithetic-minus-random gain was +0.9203 with CI [+0.7713, +1.0746].
Exact initial-observation and primary-source hashes matched within every seed,
and IID and antithetic modes both used 128 NFE/action.

The stronger deployment gate did not pass. Antithetic-minus-zero was negative
in all three new seeds and pooled to -0.0847 with CI [-0.2120, +0.0406]. Seed
45's individual interval was strictly negative. Together with archived seed
42, the result supports stable recovery toward deterministic central behavior,
not superiority over zero-source inference.

## Decision

Confirm the H70 equal-total-NFE construction on these three fixed checkpoints.
Only if that screen passes should official cross-task training begin. Preserve
zero as the stronger deployment baseline in every table.
