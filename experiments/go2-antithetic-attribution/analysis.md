# H67 Analysis: Equal-Compute Source Attribution

## Outcome

H67 passed its preregistered mechanism and reward gates. On 2,048 fixed Go2
states, IID endpoint averaging retained 70.51% of a single random endpoint's
normalized displacement from the zero action. Exact antithetic averaging
retained only 12.45%. All endpoint means were bitwise equal to the specified
float32 expressions, zero dispatch was bitwise exact, and actor parameters were
unchanged.

The paired 256-environment screen completed 1,024 episodes with exact initial
observation hashes and exact shared primary source-stream hashes. Mean returns
were:

| Method | NFE/action | Mean return |
|---|---:|---:|
| Zero source | 64 | 41.3748 |
| One random source | 64 | 40.2003 |
| IID source pair | 128 | 40.8501 |
| Antithetic source pair | 128 | 41.6294 |

At equal 128 NFE, antithetic pairing beat IID averaging by `+0.7793`; paired
SEM was `0.1896` and the locked bootstrap 95% interval was
`[+0.4913, +1.2095]`. Thus H66's gain is not explained by generic two-sample
test-time averaging. IID averaging itself beat one random endpoint by
`+0.6499`, while antithetic pairing beat it by `+1.4292`.

Antithetic pairing also had a positive `+0.2547` point estimate over the matched
zero-source policy, but its interval `[-0.0267, +0.6194]` crosses zero. This is
suggestive rather than evidence that H66 beats the best deterministic deployment
mode. The next authorized step is cross-checkpoint screening, followed by
independent training seeds only if the effect is not isolated to the final
checkpoint.

## Reproducibility

Raw artifacts are in `results/`. Re-running `analyze.py` on the four evaluation
JSON files reproduces `results/analysis.json` byte-for-byte. The local focused
unit tests pass (`4 passed`); Ruff, Python compilation, remote Isaac imports,
and shell syntax checks also pass. The unrelated full local suite cannot collect
because the active local Python lacks the package-declared `GitPython`
dependency.

