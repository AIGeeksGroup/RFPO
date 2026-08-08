# Focused Literature Survey

## Core Papers

| Paper | Mechanism | Relevance |
|---|---|---|
| Flow Policy Gradients for Robot Control (2026) | Per-sample CFM ratios and ASPO | Target benchmark and baseline |
| Flow Straight and Fast (2022) | Reflow straightens model-induced transport couplings | Candidate for lower-NFE policies |
| WarmPrior (2026) | Temporal action prior with residual Gaussian noise | Candidate for better manipulation exploration and reward |
| Generalized Advantage Estimation (2015) | Exponentially weighted advantage estimation with a bias-variance parameter | Candidate for propagating sparse Can success farther through long episodes |

## Additional Candidates To Audit After Baseline

- One-Step Flow Policy Mirror Descent, arXiv:2507.23675: directly targets one-step online flow policies.
- Trajectory-Consistent Flow Matching, arXiv:2605.08511: uses trajectory consistency and velocity smoothness to close the train-inference gap.
- Truncated Rectified Flow Policy, arXiv:2604.09159: combines flow straightening with entropy-regularized RL and one-step sampling.

These are not yet active hypotheses. They will be promoted only if the official baseline reproduces and the simpler reflow/WarmPrior pilots expose a relevant failure mode.
