# Outer-Loop Cycle 29: Nonstationary On-Policy Optimization

## Reflection

H42 confirms that changing execution or source behavior can improve Gaussian exploration while
damaging the deterministic mode. The next method should leave the policy's rollout distribution and
benchmark contract unchanged. Prior optimizer experiments changed learning rate, minibatch
accumulation, clipping, or rollback within one rollout, but none tested optimizer state carried
between independently collected on-policy rollouts.

FPO++ keeps one AdamW instance across all rollout iterations. With actor betas `(0.9, 0.99)`, its
moments mix gradients from policies and state distributions that are no longer current. Clearing
actor moments at each new rollout boundary is coefficient-free, preserves the official objective,
and is directly testable in the validated Square screening regime.

## Candidates

| Rank | Candidate | Lens and rationale | Decision |
|---:|---|---|---|
| 1 | Rollout-local actor Adam state | decomposition; isolate stale cross-rollout moments while preserving each rollout's ten update epochs | Select H43 |
| 2 | Natural-block gradient agreement | composition; preserves within-block structure, but block definitions and a consensus threshold add choices | Park |
| 3 | Leave-environment-out return baseline | decomposition; can reduce shared rollout noise, but states are not matched across environments | Park |
| 4 | Extragradient FPO++ | control analogy; anticipates local rotation, but doubles actor passes and adds a virtual-step choice | Park |
| 5 | Learned outcome-based chunk reranking | reward-relevant inference, but needs a new action-value model and labels | Park |
| 6 | Actor EMA evaluation | smoothing; cheap if trained explicitly, but current manipulation checkpoints do not save an EMA state | Park |
| 7 | Mode-dependent 8/16 action horizon | post-hoc composition of H42 cells | Reject by H42 stopping rule and source-mode-specific policy selection |
| 8 | State-dependent source scale | adaptive exploration | Reject because source-scaling and guided-source transfer are closed |
| 9 | Endpoint-PCGrad variants | local refinement | Reject by H41 stopping rule |
| 10 | Potential-based task shaping | denser credit | Reject because it changes the official benchmark reward contract |

## Selection

H43 tests whether AdamW's cross-rollout state, rather than its within-rollout update path, contributes
to unstable FPO++ adaptation. The candidate clears only actor optimizer state immediately before the
actor epochs of each iteration. The LR scheduler and critic optimizer remain persistent. A paired
five-iteration Square screen can reject the idea before any official-scale run.
