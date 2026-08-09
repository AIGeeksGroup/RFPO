# Outer-Loop Cycle 32: Within-Observation Flow Confidence

## Reflection

H45 produced the largest recent Gaussian-source point improvement, but its pooled gain remained
below the locked gate and all neighboring batching variants are closed. Across H38, H42, H43, H44,
and H45, changes that globally smooth the sampler or online trajectory repeatedly trade deterministic
and Gaussian-source behavior. The next candidate should preserve the checkpoint and zero-source
policy exactly while acting only on ambiguous Gaussian samples at inference.

Conditional reflow established that flow straightness is measurable, although globally straightening
the model did not yield a stable Can reward gain. A distinct question remains: at one fixed
observation, does the released policy's own ODE path identify the more self-consistent of two
Gaussian action candidates? Selecting the lower-curvature candidate is coefficient-free and can be
rejected in one balanced Square screen without retraining.

## Candidates

| Rank | Candidate | Lens and rationale | Decision |
|---:|---|---|---|
| 1 | Lower-curvature best-of-two Gaussian chunks | within-observation confidence; preserves weights and zero mode | Select H46 |
| 2 | Reward-model best-of-two chunks | direct reward relevance, but prior DAE could not learn action effects from the short sparse rollout | Park |
| 3 | More-frequent fresh-rollout updates | targets stale update epochs, but requires substantially more environment interaction for equal optimizer work | Park |
| 4 | Length-normalized CFM ratios | changes trust-region scale and requires a new matched clip coefficient | Park |
| 5 | Scalar behavior-anchor regularization | H28-H30 already close fixed-reference behavior anchoring and coefficient variants | Reject |
| 6 | Advantage quantile minibatches | forbidden neighbor of H45 | Reject |
| 7 | Best-of-two distance to zero action | reopens source shrinkage and explicitly favors the deterministic mode | Reject |

## Selection

H46 draws exactly two independent standard-Gaussian sources whenever a new action chunk is needed,
integrates both with the unchanged Euler-10 sampler, and computes normalized straightness error over
the 16 actions that will actually be executed. Each environment receives the candidate with lower
error. The zero-source code path is unchanged. A candidate-only smoke must first establish finite
paths, exact per-environment argmin selection, and non-degenerate use of both exchangeable branches.

