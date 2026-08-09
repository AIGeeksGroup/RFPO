# H76: Retrospective Cross-Policy Statistical Summary

## Purpose and status

Summarize the four already completed independent Go2 training seeds at the
policy level. This is a retrospective publication analysis: the underlying
H67/H69/H70/H72/H73 outcomes have already been observed, so H76 is not a new
confirmatory hypothesis and has no pass/fail gate.

## Fixed estimands

Treat each training seed (42, 43, 44, 45) as one equally weighted independent
unit. Extract exactly these three per-policy paired-return differences:

1. `antithetic32 - random64` from H70 for seed 42 and H73 for seeds 43-45;
2. `antithetic64 - iid_pair64` from H67 for seed 42 and H69 for seeds 43-45;
3. `antithetic32 - zero32` from H72 for seed 42 and H73 for seeds 43-45.

Do not pool episodes across policies for the primary cross-policy uncertainty.
For each estimand report the four raw effects, arithmetic mean, sample SD, SEM,
two-sided 95% Student-t interval with three degrees of freedom, one-sample
t statistic and two-sided p-value, and the exact one-sided sign-test p-value.
The sign test is reported because normality cannot be assessed reliably with
four policies. Episode-level paired bootstrap intervals remain the within-policy
evidence in the source experiments.

## Interpretation

Lead with effect magnitude and interval, not significance labels. A positive
t interval may support consistency under a normal random-seed-effects model,
but four policies are still a small generalization sample. An exact sign-test
value of `1/16 = 0.0625` for four same-direction effects is not conventionally
significant and must be reported rather than omitted.
