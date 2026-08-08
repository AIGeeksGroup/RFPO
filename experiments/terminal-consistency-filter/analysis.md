# H35 Analysis: Terminal-Consistency GAE Filter

## Outcome

H35 is refuted at the fixed two-replica mechanism audit. Normalized GAE signs already agreed with
centered observed terminal returns on more than 93% of labeled chunks. The candidate was therefore
nearly identical to control and slightly reduced outcome-gradient alignment in both replicas. No
online integration or filter variant is authorized.

## Audit Health

- Iteration-1/2 Gaussian collection success: 8/16 and 9/16.
- Iteration-2 valid CFM actions: 4,939/5,120 (96.46%).
- Fully valid chunks with observed terminal labels: 221; audited chunks: 128.
- Both replicas retained more than eight positive and eight negative weights.
- All control, candidate, and reference gradients were finite and nonzero.

## Gate Results

| Metric | Replica 0 | Replica 1 | Gate |
|---|---:|---:|---:|
| Retained chunks | 61/64 (95.31%) | 60/64 (93.75%) | 40% to 90% |
| Control/outcome cosine | 0.72328 | 0.78459 | diagnostic |
| Candidate/outcome cosine | 0.72305 | 0.78219 | >= 0.75 |
| Candidate cosine gain | -0.00023 | -0.00240 | >= +0.10 |
| Candidate/control norm ratio | 0.99864 | 0.99456 | 0.50 to 1.20 |

The candidate failed the active-filter and direction-gain gates twice; replica 0 also failed the
absolute outcome-cosine gate. Candidate/control cosine printed slightly above one because of float32
reduction error on the large flattened vectors, but the norm and outcome metrics make the negative
decision unambiguous.

## Interpretation

The high sign agreement independently confirms that official GAE already assigns the correct broad
direction to most terminal-labeled chunks at step 6000. Its remaining outcome-alignment error is not
concentrated in a useful set of sign contradictions. Soft masks, return thresholds, recentering, and
mixing coefficients would change the frozen hypothesis and are stopped.
