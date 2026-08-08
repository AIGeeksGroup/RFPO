# H32 Analysis: Advantage-Sign Conflict Projection

## Outcome

H32 is refuted at its fixed-batch mechanism audit. No online optimizer integration, reward screen, or
projection-order variant is authorized.

## Audit Health

- Iteration-1 and iteration-2 random-source collection success was 8/16 and 13/16.
- Iteration-2 valid CFM actions: 4,948/5,120 (96.64%).
- Fully valid chunks with observed terminal labels: 208; audited chunks: 64.
- Each 32-chunk batch contained 12 positive and 20 negative normalized-GAE chunks, passing the
  minimum eight-per-sign support condition.
- Training, ratios, and all gradient norms were finite; no NaN, OOM, or rendering failure occurred.

## Gate Results

| Metric | Batch 0 | Batch 1 | Gate |
|---|---:|---:|---:|
| Positive/negative gradient cosine | -0.00524 | +0.05487 | conflict in both |
| Projection active | yes | no | yes in both |
| Control cosine to outcome reference | 0.94806 | 0.73389 | diagnostic |
| Candidate cosine to outcome reference | 0.94797 | 0.73389 | at least 0.75 |
| Candidate cosine gain | -0.00009 | 0.00000 | at least +0.10 |
| Candidate/control norm ratio | 1.00261 | 1.00000 | 0.70 to 1.30 |

The candidate passed norm retention and remained effectively identical to the official update. It
failed the required outcome-reference gain in both batches. Batch 1 additionally failed the
non-vacuous-conflict and absolute-reference-cosine gates.

The logged control/candidate self-cosines are slightly above one because the diagnostic used
float32 cosine reduction over a very large flattened gradient vector. Norm ratios and all primary
control-to-reference comparisons remain finite, and the pass/fail conclusion has margins orders of
magnitude larger than this numerical reduction error.

## Interpretation

The hypothesized cancellation is absent: the two advantage-sign branches are nearly orthogonal in
one batch and weakly aligned in the other. Removing the tiny opposing component cannot repair the
actor-update direction. Positive-only optimization, alternate projection ordering, and per-sign
gradient surgery would revisit the same rejected mechanism and should not be tested next.
