# Outer-Loop Cycle 33: Uncensor Square Failure Outcomes

## Reflection

H46 reliably lowered flow curvature but reduced corrected Square random success. Geometry, source
ranking, success replay, critic replacement, optimizer stabilization, and behavior anchoring are now
closed. The next candidate should change the availability of reward evidence without changing the
policy source or inventing another proxy.

The H43 logs expose a task-boundary mismatch: Square has a 400-step episode horizon, while each
training rollout ends after 320 steps. Successful episodes terminate early, but initial failures are
still alive at the collection boundary and GAE bootstraps them from the critic. Extending each
rollout to 400 steps supplies observed failure terminals while keeping the total environment-step
budget fixed by using four rather than five rollouts.

## Candidates

| Rank | Candidate | Rationale | Decision |
|---:|---|---|---|
| 1 | Full-horizon 400-step Square rollouts | exposes terminal failure labels; same total interaction budget and no new loss coefficient | Select H47 |
| 2 | Learned action-outcome best-of-two | directly reward-relevant, but H21 reversed outcome ranking and a new predictor adds substantial machinery | Park |
| 3 | More frequent fresh rollouts | addresses repeated-epoch drift but requires more environment interaction at equal optimizer work | Park |
| 4 | Exact CNF likelihood ratios | could improve trust-region fidelity but requires expensive divergence estimation | Park |
| 5 | Natural-gradient blocks | may control parameter-space drift but adds block and damping choices | Park |
| 6 | Leave-environment-out baseline | may reduce shared rollout noise, but states are unmatched across environments | Park |
| 7 | Self-imitation CFM auxiliary | reward-conditioned, but H15 found success-only and positive-advantage gradients nearly orthogonal | Reject |
| 8 | Checkpoint early stopping | H43 training SR is censored and identically 100%, so it cannot select without test leakage | Reject |
| 9 | Alternate curvature score | forbidden H46 neighbor after the corrected negative screen | Reject |
| 10 | Five-epoch actor updates | adjacent to the closed H18 actor-stopping route and introduces a post-failure epoch choice | Reject |

## Selection

H47 changes only the grouping of the same 1,600 per-environment interaction steps: control uses
five 320-step rollouts and four actor updates, while the candidate uses four 400-step rollouts and
three actor updates. A positive result would show that complete outcome information is worth more
than one additional repeated-data actor update; a negative result closes the horizon-censoring route
without a large run.
