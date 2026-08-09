# Paired Antithetic Rollouts for Flow-Policy Training

## Retrieval scope

Targeted search on 2026-08-09 for direct precedents to paired `z/-z` closed-loop
rollouts, common-random-number trajectory comparisons, and antithetic policy-
gradient estimation. This is a novelty screen, not an exhaustive systematic
review.

## Closest precedents

### Reducing Variance in Gradient Bandit Algorithm using Antithetic Variates Method

Liu et al. (2018), DOI `10.1145/3209978.3210068`, applies antithetic variates to
a multi-armed-bandit policy-gradient estimator. It proves variance reduction
for its Antithetic-Arm Bandit construction. This is direct precedent for the
general claim that antithetic sampling can reduce an RL gradient estimator's
variance, but it does not study continuous control, flow policies, matched
closed-loop trajectories, or FPO++.

### Evolution Strategies as a Scalable Alternative to Reinforcement Learning

Salimans et al. (2017), arXiv:1703.03864, uses common random numbers and mirrored
parameter perturbations in evolution strategies. It prevents novelty claims
for mirrored sampling as a generic policy-search primitive. The perturbations
operate in policy-parameter space rather than a flow policy's action-source
space.

### Using Common Random Numbers for Simulation-based Planning with Rollouts

Yadav et al. (2026), arXiv:2605.04732, proves that common random numbers can
reduce variance in relative rollout utilities for simulation-based planning.
It studies action comparison in a sampling model, not on-policy learning or
antithetic flow latents, but is a close trajectory-level variance-reduction
precedent.

### Antithetic Noise in Diffusion Models

Jia et al. (2025), arXiv:2506.06185, directly precedes `z/-z` endpoint pairing
and approximate affine antisymmetry in diffusion and normalizing-flow models.
Any new contribution must therefore arise from how mirrored flow actions are
used in closed-loop control and policy optimization, not from endpoint
antisymmetry itself.

## Search outcome and claim boundary

The targeted sources did not return a paper combining all of: a conditional
flow control policy, matched closed-loop `z/-z` trajectories from the same
initial state, and pair-centered FPO/PPO updates. Semantic Scholar returned
HTTP 429 and contributed no results, so absence there is not evidence of
exhaustive novelty.

A defensible future claim would be narrowly phrased: approximate odd symmetry
of a learned flow policy can create useful closed-loop return covariance, and
that covariance can be converted into a lower-variance FPO++ update. The first
empirical prerequisite is that separately executed mirrored trajectories
actually reduce source-induced return variance; immediate action symmetry from
H75 alone does not imply this after feedback dynamics.

## Reproducible provenance

- OpenAlex `GET /works`, queries `antithetic sampling reinforcement learning
  policy gradient`, `mirrored sampling policy gradient reinforcement learning`,
  `paired trajectories common random numbers reinforcement learning variance
  reduction`, and exact search `"antithetic variates" "policy gradient"`;
  first 20/20/20/50 results respectively.
- OpenAlex single-work lookups `W2811175972` and DOI
  `10.48550/arxiv.1703.03864`.
- arXiv `GET /api/query`, query `(all:"common random numbers" OR all:"paired
  rollouts" OR all:"mirrored sampling") AND (all:"reinforcement learning" OR
  all:"policy optimization")`, first 30 results; six indexed matches.
- arXiv single-ID lookup `2605.04732`.
- Semantic Scholar paper search `antithetic policy gradient reinforcement
  learning`, first 20 requested; HTTP 429, no usable records.
