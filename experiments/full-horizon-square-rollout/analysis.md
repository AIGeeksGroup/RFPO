# Analysis: H47 Full-Horizon Square Rollouts

## Training Validity

The candidate completed four 400-step rollouts at the same 25,600-environment-step budget as the
five-by-320-step H43 control. Its initial collection prefix exactly reproduced H43: at step 300,
both runs had completed two successful episodes. By step 400, the candidate had observed 17 episode
terminations, comprising three successes and 14 failures. This passes the locked requirement that
all 16 initial environment episodes receive an observed terminal before the rollout ends.

All four candidate iterations and checkpoints were finite. Completed episodes by iteration were
`[17, 16, 16, 16]`, successes were `[3, 8, 7, 6]`, and valid CFM action fractions were
`[99.50%, 98.58%, 97.94%, 97.06%]`. The intervention therefore changed terminal-label availability
as intended rather than failing as an inactive configuration.

## Balanced Reward Screen

All cells used deterministic `20260923 + env_id` environment seeds, 20 environments contributing
one episode each, Euler-10, 16 executed actions, no EMA, and OSMesa.

| Condition | Zero | Random | Pooled |
|---|---:|---:|---:|
| Five-by-320 control | 10/20 (50%) | 6/20 (30%) | 16/40 (40%) |
| Four-by-400 candidate | 8/20 (40%) | 6/20 (30%) | 14/40 (35%) |
| Difference | -2/20 | 0/20 | -2/40 |

The candidate fails every improvement requirement: random does not gain the required 2/20, zero
loses two rather than at most one, and pooled success falls by two rather than gaining three.

## Interpretation and Decision

Extending Square collection through the complete environment horizon successfully converts censored
initial failures into observed zero-return terminals. That extra outcome information does not improve
the final policy under the equal-interaction budget; removing one actor-update rollout while using
larger, terminal-complete batches leaves Gaussian success unchanged and harms deterministic success.
This distinguishes label completeness from reward utility: the released bootstrap is imperfect but
not the dominant bottleneck in this screen.

Refute H47. Do not test 360/384-step neighbors, add actor updates or interactions, choose an
intermediate checkpoint, change the seed, or run confirmation.
