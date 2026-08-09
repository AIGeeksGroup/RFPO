# H51 Stage A Analysis: Potential-Based Square Shaping

## Result

The frozen-policy audit passed every locked mechanism gate. Sixteen deterministic Square episodes
completed from seeds `20260958..20260973`, including 3 successes and 13 failures, with 5,871 recorded
primitive transitions.

| Metric | Locked gate | Result | Outcome |
|---|---:|---:|---|
| Maximum absolute telescoping residual | `<= 1e-5` | `3.75e-16` | Pass |
| Active failure nonterminal shaping | `>= 20%` | `76.02%` | Pass |
| Positive failure nonterminal shaping | `>= 5%` | `23.23%` | Pass |
| Negative failure nonterminal shaping | `>= 5%` | `52.79%` | Pass |
| Finite bounded potentials and complete episodes | required | valid | Pass |

The 5,187 audited nonterminal transitions from failed episodes had shaping-term mean `-0.000366`,
standard deviation `0.03325`, minimum `-0.49570`, and maximum `0.42067`. The signal is therefore
nontrivial in both directions rather than a nearly constant offset.

## Interpretation

The discounted candidate return differs from sparse return only by the locked initial-state
potential to numerical precision. This empirically validates the intended policy-invariant reward
transformation on complete wrapper episodes and avoids raw dense reward's hover-stalling incentive.
At the same time, most failed-episode transitions expose local progress or regression to GAE.

Stage A is mechanism evidence only. It authorizes the already-preregistered five-iteration paired
training screen; it is not reward-improvement or official-renderer benchmark evidence.

## Artifacts

- `results/config.json`
- `results/results.json`
- `results/records.jsonl`
- `results/run.log`

