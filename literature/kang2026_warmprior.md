# WarmPrior: Straightening Flow-Matching Policies with Temporal Priors

- Authors: Sinjae Kang, Chanyoung Kim, Kaixin Wang, Li Zhao, Kimin Lee
- Year: 2026
- Identifier: arXiv:2605.13959v1
- URL: https://arxiv.org/abs/2605.13959
- Retrieved: 2026-08-08 from the arXiv PDF

## Mechanism

WarmPrior replaces the standard Gaussian source with a Gaussian centered on recent action history. WP-Past uses the previously executed action chunk with residual noise scale 0.5. WP-Preview predicts two chunks, executes the first, and uses the second as the next prior mean. The network, interpolant, and flow-matching loss remain unchanged.

## Reported Evidence

The paper reports straighter paths and improved behavior-cloning success across Robomimic and MimicGen, especially at low numbers of function evaluations. In prior-space RL, a bounded residual around the warm mean improves sample efficiency and final success over DSRL baselines on Square and Transport.

## Relevance to FPO++

FPO++ manipulation uses action chunks and suffers when a base policy has high zero-sampling but low random-sampling success. WP-Past may make stochastic exploration more task-aligned without changing the FPO++ ratio objective. Unlike post-hoc reflow, this requires base-policy pretraining with the altered source distribution.

## Pilot Choice

Test WP-Past before WP-Preview because it does not double the prediction horizon. First measure base-policy zero/random success and flow straightness; only then fine-tune with FPO++ under an equal rollout budget.

