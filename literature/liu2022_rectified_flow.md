# Flow Straight and Fast

- Authors: Xingchao Liu, Chengyue Gong, Qiang Liu
- Year: 2022
- Identifier: arXiv:2209.03003v1
- URL: https://arxiv.org/abs/2209.03003
- Retrieved: 2026-08-08 via `https://export.arxiv.org/api/query?id_list=2209.03003`

## Relevant Mechanism

Rectified Flow fits the velocity of linear interpolants between samples from two endpoint distributions. Reflow then samples endpoint pairs from the learned ODE and fits a new rectified flow to that induced coupling. Repeated rectification has non-increasing convex transport costs and tends to straighten trajectories, reducing numerical integration error.

## Relevance to FPO++

FPO++ already trains a conditional linear flow from Gaussian noise to robot actions, so the base least-squares objective is not new here. The nontrivial transfer is conditional reflow: for each observation, retain the sampled initial noise and the action produced by the current policy, then refit or regularize the policy using this model-induced coupling. The immediate prediction is reduced conditional trajectory curvature and better low-NFE action fidelity. Any reward gain must be established separately.

## Main Risk

Straightening may make the transport more deterministic and reduce useful multimodal exploration. Reflow must therefore be evaluated under both zero-noise inference and random-noise training distributions.

