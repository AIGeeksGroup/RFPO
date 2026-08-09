# Analysis: Replay-Exact Vine Return Audit

## Outcome

H62 is refuted. The audit was valid: all 184 scheduled suffix rollouts replayed their branch states exactly, all outcomes and lengths were valid, candidate actions were active, candidate Gaussian sources matched the locked population checks, and policy parameters remained bitwise unchanged. The branch-return rankings nevertheless failed six of seven signal gates, so no actor update, branch-step or continuation tuning, checkpoint change, hindsight state selection, or additional seed block is authorized.

Eight roots yielded 23 active vine states and 184 suffix rollouts rather than the maximum 24 states and 192 rollouts because one root had already terminated at its final scheduled branch point.

## Locked Gates

| Gate | Observed | Requirement | Result |
|---|---:|---:|---|
| Exact prefix replays | 184 / 184 | all | Pass |
| Candidate source mean | 0.000469 | absolute value at most 0.05 | Pass |
| Candidate source standard deviation | 0.992498 | 0.95 to 1.05 | Pass |
| Median candidate normalized RMS | 0.277857 | at least 0.05 | Pass |
| Nonzero states, block A | 2 | at least 8 | Fail |
| Nonzero states, block B | 3 | at least 8 | Fail |
| Common nonzero states | 1 | at least 6 | Fail |
| Pearson correlation | -0.437798 | at least 0.35 | Fail |
| Common sign agreement | 0.000000 | at least 0.75 | Fail |
| A-selected top quartile in B | 0.166667 | strictly positive | Pass |
| B-selected top quartile in A | -0.083333 | strictly positive | Fail |

## Interpretation

Exact low-level prefix replay is technically viable in this environment, and independently sampled Gaussian candidates produce materially different action chunks. Sparse terminal success is too insensitive at most intermediate states, however: 20 of 23 states in each block produced tied candidate outcomes, only one state was informative in both blocks, and its ranking reversed. The negative full-state correlation and failed reverse cross-fit show that the few observed differences are continuation-noise effects rather than reproducible local action credit.

This result rejects replay-exact binary branch returns as a practical critic-free ranking signal under the locked Square setup. Per protocol, the branch-return FPO reward screen is not implemented.

## Artifacts

- `results/config.json`
- `results/roots.json`
- `results/records.json`
- `results/results.json`
- `logs/audit.log`

Local and remote SHA-256 checksums match for all five artifacts. The audit released physical GPU 3 normally after completion.
