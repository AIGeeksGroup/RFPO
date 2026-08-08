# Research Findings

## Research Question

Can the official FPO++ robot-control results be reproduced, and can conditional reflow make the learned policy flow straighter enough to improve return, robustness, or sampling efficiency?

## Current Understanding

FPO++ stabilizes likelihood-free policy gradients through per-CFM-sample ratios and an asymmetric trust region for from-scratch locomotion. Its policy still uses an Euler-integrated conditional flow with up to 64 steps. Rectified Flow suggests that regenerating paired endpoints from an existing flow and fitting those pairs can straighten trajectories, but it does not imply a reward improvement by itself. The first defensible target is therefore lower curvature and preserved task return at fewer sampling steps; online return improvements are a second-stage hypothesis.

## Key Results

The released Can step-1000 checkpoint completed an early 20-episode validation in each sampling mode. Zero sampling succeeded on 16/20 episodes (80%), while random sampling succeeded on 2/20 episodes (10%). The latter exactly matches the low stochastic-success regime described in the paper; the zero-sampling estimate is compatible with the reported 73.76% base-policy success but remains too small for a final reproduction claim. A 200-episode evaluation is in progress.

## Patterns and Insights

- FPO++ already uses the linear conditional flow-matching objective, so simply renaming it rectified flow is not a contribution.
- The transferable mechanism from arXiv:2209.03003 is recursive reflow using model-induced endpoint coupling.
- A reflow method that only improves inference speed is still useful, but it must not be presented as an RL performance gain.
- WarmPrior (arXiv:2605.13959) offers a more direct reward-improvement hypothesis for chunked manipulation: center the flow source on recent actions and retain residual Gaussian noise for exploration.

## Lessons and Constraints

- Reproduce released baselines before changing objectives.
- Use short validation runs before full 1500-iteration or multi-seed jobs.
- Stop a candidate when it degrades the primary metric beyond seed noise or fails to improve its claimed mechanism.

## Open Questions

- Does conditional reflow preserve the multimodal exploration that gives flow policies their advantage?
- Is curvature correlated with FPO ratio variance or gradient disagreement?
- Should reflow be an offline post-training stage, an auxiliary online loss, or both?
- Can WP-Past close the large gap between zero-sampling and random-sampling success of the released Can base policy?

## Optimization Trajectory

Pending baseline measurements.
