# Flow Policy Gradients for Robot Control

- Authors: Brent Yi et al.
- Year: 2026
- Identifier: arXiv:2602.02481v1
- URL: https://arxiv.org/abs/2602.02481
- Official code: https://github.com/amazon-far/fpo-control
- Retrieved: 2026-08-08

## Relevant Results

FPO++ adds per-sample CFM ratios and an asymmetric PPO/SPO trust region to vanilla FPO. The official release provides IsaacLab locomotion and G1 motion tracking, plus five image-based manipulation fine-tuning tasks. The published Go2 target is approximately 40 final training return after 1500 iterations with 4096 environments. Manipulation reproduction uses pretrained base policies, four fine-tuning methods, three seeds, and zero-sampling success rate.

## Reproduction Gate

Before proposing improvements, reproduce one from-scratch locomotion task (Go2) and one pretrained manipulation task (Can). Accept the baseline when learning trends match and the final metric is within a tolerance justified from released seed variation, rather than requiring pointwise curve identity.

