# Outer Loop Cycle 18: Reward-Gradient Candidates

## Diagnosed Bottleneck

The step-6000 policy supplies enough successful Gaussian rollouts, and official GAE already ranks
observed outcomes well. Prior experiments reject changing source distributions, inference smoothing,
CFM Monte Carlo allocation, critic targets, advantage transforms, optimizer step size, and generic BC
anchors. The remaining problem is how opposing reward-gradient components combine during an FPO++
update.

## Candidate Set

| Rank | Candidate | Distinct mechanism | Fast falsification | Decision |
|---:|---|---|---|---|
| 1 | Advantage-sign conflict projection | Preserve positive-GAE progress when the negative-GAE branch directly opposes it | Two-batch outcome-gradient audit | Select H32 |
| 2 | Layerwise sign-conflict masking | Localize conflicts to parameter blocks | Same audit, but introduces arbitrary layer granularity | Park |
| 3 | Leave-environment-out baseline | Remove shared rollout noise without changing ranks | Offline variance audit | Park; weakly tied to observed drift |
| 4 | State-dependent PPO clipping | Allocate trust region by critic uncertainty | Requires calibration and a clipping schedule | Reject for current pilot |
| 5 | Advantage-gated endpoint anchor | Protect BC behavior only on low-confidence states | Combines two unproven gates | Reject until a simpler signal works |
| 6 | Potential-based reward shaping | Densify Can credit | Changes the benchmark reward contract | Reject |

## Selection

H32 composes the existing coefficient-free projection operator with a decomposition already defined by
the FPO++ objective. Unlike H11, it does not discard negative-advantage samples or replace signed GAE.
Unlike H28-H30, it does not use a behavior-cloning teacher. It removes only the component of the
negative-gradient branch that directly cancels the positive branch, and only when their dot product is
negative.

The strongest objection is that positive GAE is not itself ground truth. Therefore the audit is not
judged by alignment to the positive branch alone: both control and candidate are compared with a
centered, observed Monte Carlo terminal-return gradient on the same chunks. No training is authorized
unless the candidate improves that independent direction in both fixed batches while remaining close
to the original signed update.
