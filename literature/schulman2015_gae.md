# High-Dimensional Continuous Control Using Generalized Advantage Estimation

- Authors: John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel
- Year: 2015
- Identifier: arXiv:1506.02438v6
- URL: https://arxiv.org/abs/1506.02438
- Retrieved: 2026-08-08 from `https://export.arxiv.org/api/query?id_list=1506.02438&max_results=1`

## Relevant Mechanism

Generalized Advantage Estimation uses an exponentially weighted estimator analogous to TD(lambda).
The parameter `lambda` controls the bias-variance tradeoff: stronger bootstrapping reduces variance
but adds bias, while `lambda` closer to one propagates observed returns farther through a trajectory.

## Relevance to Can FPO++

Can provides a sparse terminal success reward and has a 300-step horizon. The official configuration
uses `gamma=0.99` and `lambda=0.99`, so the approximate coefficient on a terminal residual 300 steps
earlier is `(0.99 * 0.99)^300 = 0.0024`. Holding the task discount fixed and setting only
`lambda=1.0` raises it to `0.99^300 = 0.049`, exposing early actions to substantially more success
credit without altering the reward, policy, or evaluation. The risk is increased gradient variance.

