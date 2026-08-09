# Outer Loop Cycle 47: Target Actor Inner-Loop Drift Directly

## Problem Boundary

H60 shows that substantially better flow integration is not reward-relevant on the reproduced Go2
checkpoint. Across manipulation audits, the more persistent failure is actor inner-loop drift:
held-out CFM gradients rotate and positive ratios clip while interventions that strongly preserve the
policy usually discard most useful surrogate progress. The next intervention should therefore act
on the training objective that creates this drift, not on sampler geometry alone.

## Candidates

| Rank | Candidate | Fast falsification | Decision |
|---:|---|---|---|
| 1 | ReFPO fixed-batch CFM regularizer | paired two-iteration Can audit | Select H61 |
| 2 | VINE-style value-free credit assignment | small trajectory-level estimator audit | Next if H61 fails |
| 3 | QGF/QPILOTS-style value guidance | frozen-policy action ranking | Deprioritize; requires a reliable critic |
| 4 | More solver/reflow variants | endpoint geometry audit | Reject; H1, H1b, and H60 close geometry as a reward proxy |

## Why H61 Is New Evidence Rather Than Repeating H2

H2 tested whether offline conditional reflow made gradients from different Monte Carlo CFM draws
agree. It failed and correctly stopped an auxiliary justified by that gradient-disagreement mechanism.
ReFPO, published subsequently, specifies a different direct intervention and claim: add the current,
unweighted fixed-batch CFM loss during the same FPO update to reduce proxy-ratio sensitivity. H61
tests that exact fixed-batch claim and keeps H2's negative cross-Monte-Carlo result as a required
non-degradation check.

## Selection

Use the paper's MuJoCo coefficient `lambda=0.04` once, without a sweep. First compare it with the
official update on two exactly paired 32-chunk held-out batches. The candidate must reduce held-out
ratio drift while retaining at least half of control surrogate progress and must not worsen outcome-
gradient preservation. A failure stops ReFPO before reward training; a pass permits only a short
matched reward screen.
