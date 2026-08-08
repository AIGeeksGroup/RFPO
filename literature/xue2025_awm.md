# Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models

- Authors: Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, Zhi-Ming Ma
- Year: 2025
- Identifier: arXiv:2509.25050v1
- URL: https://arxiv.org/abs/2509.25050
- Retrieved: 2026-08-08 from the arXiv API and PDF

## Relevant Mechanism

AWM treats score or flow matching as a sequence-policy likelihood surrogate and weights its loss by
the sample advantage. Positive advantages lower the matching loss for rewarded samples and negative
advantages raise it. A frozen-reference velocity penalty is optional. The method preserves the
pretraining objective and decouples rollout sampling from forward-process training noise.

## Relevance to FPO++

The signed on-policy gradient is closely related to FPO++, so AWM alone is not a novel replacement.
Its useful lesson is that clean forward-process CFM regression is a low-variance policy update. The
remaining design choice for Can is how to weight rare successful trajectories without using biased
guided-source rollouts.
