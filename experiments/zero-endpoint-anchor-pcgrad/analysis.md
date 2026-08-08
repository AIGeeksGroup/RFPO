# H29 Zero-Endpoint Anchor Conflict Projection Audit

## Result

H29 passes every locked gate in both 32-chunk batches. Replacing the generic velocity-field anchor
with the deterministic zero-source action endpoint weakened RL/BC conflict enough to retain 73.7%
and 72.8% of control surrogate progress while reducing endpoint drift to 33.5% and 23.6% of control.

| Metric | Batch 0 | Batch 1 | Gate | Pass |
|---|---:|---:|---:|:---:|
| RL/endpoint gradient cosine | -0.5128 | -0.5223 | conflict active | Yes |
| Candidate/control endpoint-MSE increase | 33.45% | 23.57% | <= 50% | Yes |
| Candidate/control surrogate gain | 73.71% | 72.77% | >= 70% | Yes |
| Candidate vs control post-gradient cosine | equal at 0.9999998 | equal at 0.9999999 | candidate >= control | Yes |
| Candidate gradient-norm retention | 85.85% | 85.27% | diagnostic | - |

There were 190 eligible positive fully valid chunks and all 64 locked chunks were audited. All
gradients and metrics were finite and nonzero. The narrower endpoint objective therefore resolves
H28's overconstraint at the one-step mechanism level. This is sufficient to authorize a matched
short reward screen, but is not yet evidence of a benchmark improvement.

## Artifacts

- `results.json`: authoritative audit metrics
- `audit.log`: complete remote log
- Remote result: `~/workspace/outputs/fpo-control/results/can_step6000_zero_endpoint_pcgrad_audit_osmesa_seed20260827/`
- Remote log: `~/workspace/outputs/fpo-control/logs/can_step6000_zero_endpoint_pcgrad_audit_osmesa_seed20260827.log`

## H30 Reward Screen

H30's reward screen was invalidated before evaluating the candidate. Both matched five-iteration
training runs completed with finite losses. Across the candidate's 320 actor optimizer steps,
projection was active on 36 steps (11.25%), passing the locked 10-90% mechanism gate. The four
actor-update iteration rates were 5.00%, 8.75%, 20.00%, and 11.25%; pooled mean RL/anchor cosine was
0.2024 and mean projected-gradient norm retention was 99.954%.

The final control actor appeared to score 20/20 under both modes, but the protocol requested only 20
episodes from 50 asynchronous environments. The evaluator stopped after the first 20 completions;
successful episodes terminate much earlier than horizon-length failures, so this is informative
censoring rather than a representative 20-episode sample. Candidate evaluation and confirmation were
not run. H30 therefore provides no reward comparison in either direction; only its finite training
and projection-frequency diagnostics remain valid.

The first control-zero evaluation attempt exposed an evaluation-only division by zero when fewer
episodes than parallel environments leave some environments with no completed episode. Commit
`83ce3c6` makes the per-environment dispersion statistic ignore those empty environments. The main
episode-level success metric and rollout behavior are unchanged; 64 local tests and 5 remote focused
tests pass.

### H30 Artifacts

- `h30_control_train.log` and `h30_candidate_train.log`: complete training logs
- `h30_candidate_training.json`: all 320 projection diagnostics
- `h30_control_zero_eval_summary.txt`: 20/20 zero-source control result
- `h30_control_random_eval_summary.txt`: 20/20 random-source control result
