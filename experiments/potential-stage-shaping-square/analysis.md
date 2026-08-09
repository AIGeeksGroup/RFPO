# H51 Analysis: Potential-Based Square Shaping

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

## Stage B Paired Screen

Both methods completed the locked five iterations and 25,600 environment interactions from the
released Square checkpoint. The first two collection fingerprints matched exactly; fingerprints
diverged only after the candidate update could affect behavior. All five candidate shaping records
contained 5,120 finite terms. Active shaping ranged from 70.76% to 79.82% per iteration.

| Evaluation mode | Control | Candidate | Delta | Locked gate | Outcome |
|---|---:|---:|---:|---:|---|
| Zero source | 12/20 (60%) | 13/20 (65%) | +1/20 | >= -1/20 | Pass |
| Gaussian source | 9/20 (45%) | 12/20 (60%) | +3/20 | >= +2/20 | Pass |
| Pooled | 21/40 (52.5%) | 25/40 (62.5%) | +4/40 | >= +3/40 | Pass |

The candidate passed every preregistered screen gate. This is positive small-sample reward evidence
and authorizes a separately committed independent confirmation. It is not yet a stable improvement:
the screen contains only 20 episodes per cell, uses OSMesa rather than the official EGL renderer,
and shares one training/evaluation seed pair.

## Artifacts

- `results/config.json`
- `results/results.json`
- `results/records.jsonl`
- `results/run.log`
- `screen-results/control_collection_fingerprints.json`
- `screen-results/candidate_collection_fingerprints.json`
- `screen-results/candidate_potential_shaping_history.json`
- `screen-results/*_train.log`
- `screen-results/*_summary.txt`
- `screen-results/control_{zero,random}.log`
- `screen-results/candidate_{zero,random}.log`
