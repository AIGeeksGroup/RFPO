# Direct Advantage Estimation

- Authors: Hsiao-Ru Pan, Nico Gurtler, Alexander Neitz, Bernhard Scholkopf
- Venue/year: NeurIPS 2022; arXiv revision 2023
- Identifier: arXiv:2109.06093
- Open-access URL: https://arxiv.org/pdf/2109.06093
- Retrieved: 2026-08-09
- Discovery provenance: OpenAlex `/works?search="Direct Advantage Estimation" reinforcement learning`

## Main Idea

DAE learns the advantage directly by decomposing trajectory return into a discounted sum of
per-action effects. It enforces the defining centering constraint
`E_{a~pi(.|s)}[A(s,a)] = 0`, rather than first fitting a value function and constructing TD residuals.
The paper integrates this estimator with PPO and reports improved sample efficiency over GAE on a
majority of its discrete-control environments.

## Relevance To FPO++

H18 and H19 show that the first FPO++ actor gradient can be unreliable even when positive-ratio
activity is high or next-rollout value MSE improves. DAE attacks this failure at the advantage
estimator rather than through another actor trust-region or value-calibration change.

The released DAE method is not directly applicable: its experiments use discrete actions, and the
paper explicitly identifies continuous-action centering as an open limitation. FPO++ also uses an
implicit flow policy. A faithful adaptation therefore needs Monte Carlo centering with multiple
flow-policy action chunks sampled at the same observation. Any result must be described as a
continuous-action approximation inspired by DAE, not as the original algorithm unchanged.

## Current Decision

Promote the continuous-action approximation to H21 only as a frozen-actor mechanism audit. Do not
run online training unless fresh-rollout advantage ranking and outcome-reference gradient direction
both improve under locked gates.
