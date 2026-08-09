# H41 Square Zero-Endpoint PCGrad Analysis

## Validity

The candidate-only smoke completed one critic-only and one actor-update iteration, saved finite
checkpoints at steps 128 and 256, and emitted finite endpoint-PCGrad metrics. Its actor batch had no
reward signal, so zero projection activity was diagnostic rather than formal evidence.

The formal control and candidate used the same released EMA checkpoint, seed 20260913, five actual
iterations, and official Square FPO++ hyperparameters. Their iteration-1 and iteration-2 collection
records matched exactly before the first candidate intervention: 9 then 4 completed successes and
valid-CFM counts 5047 then 5022. Both finished at step 25,600 with finite losses and checkpoints.

The official training command overrides `n_action_steps` from the base checkpoint's 8 to 16. All
four H41 evaluation summaries therefore report 16 action steps, whereas H40's direct base-policy
screen reported 8. This means H40's absolute rates are not a matched H41 control estimate; it does
not invalidate H41 because the formal control and candidate both use the same locked official
16-step setting and the control itself remains non-saturated.

## Mechanism Activity

| Actor iteration | Active projection steps | Activity | Mean RL/anchor cosine | Mean norm retention |
|---:|---:|---:|---:|---:|
| 2 | 7/80 | 8.75% | 0.19870 | 99.96% |
| 3 | 17/80 | 21.25% | 0.10736 | 99.59% |
| 4 | 8/80 | 10.00% | 0.21531 | 99.89% |
| 5 | 12/80 | 15.00% | 0.12927 | 98.69% |
| pooled | 44/320 | 13.75% | - | - |

The projection was neither vacuous nor dominant and passed the locked 5-90% activity gate.

## Reward Results

| Mode | Control | Candidate | Difference | Gate | Result |
|---|---:|---:|---:|---:|---|
| zero | 9/20 (45%) | 9/20 (45%) | 0 | no worse than -1 | pass |
| random | 5/20 (25%) | 6/20 (30%) | +1 | at least +2 | fail |
| pooled | 14/40 (35%) | 15/40 (37.5%) | +1 | at least +3 | fail |

Every mode completed 20 finite episodes with one episode per environment. The candidate retained
zero-source behavior and moved random success in the desired direction, but the effect is below both
locked reward thresholds and is only one episode.

## Decision

H41 is refuted under its preregistered screen. Zero-endpoint PCGrad now has the same qualitative
pattern on Can and Square: active conflict projection, preserved deterministic behavior, and only a
one-episode random-source gain. This is not enough to justify confirmation. Stop without another
seed, a larger budget, checkpoint selection, projection variants, endpoint variants, or tuning.
