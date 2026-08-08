# Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning

- Authors: Xue Bin Peng, Aviral Kumar, Grace Zhang, Sergey Levine
- Year: 2019
- Identifier: arXiv:1910.00177v3
- URL: https://arxiv.org/abs/1910.00177
- Retrieved: 2026-08-08 from `https://export.arxiv.org/api/query?id_list=1910.00177`

## Relevant Mechanism

AWR replaces a signed policy-gradient update with supervised policy regression weighted by an
exponentiated advantage. High-advantage actions receive more likelihood mass, while low-advantage
actions are suppressed by relative weighting rather than by explicitly pushing their likelihood
down. This supports replay and produces a stable maximum-likelihood subproblem.

## Relevance to FPO++

FPO++ already provides a CFM-loss likelihood surrogate. An exponentiated advantage can therefore
weight its per-sample ratios while retaining the existing PPO trust region. For sparse Can rewards,
this may concentrate the actor update on the few successful standard-Gaussian trajectories without
introducing the source-distribution mismatch seen in H8-H10.
