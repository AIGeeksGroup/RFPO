# Research Findings

## Research Question

Can the official FPO++ robot-control results be reproduced, and can conditional reflow make the learned policy flow straighter enough to improve return, robustness, or sampling efficiency?

## Current Understanding

FPO++ stabilizes likelihood-free policy gradients through per-CFM-sample ratios and an asymmetric trust region for from-scratch locomotion. Its policy still uses an Euler-integrated conditional flow with up to 64 steps. Rectified Flow suggests that regenerating paired endpoints from an existing flow and fitting those pairs can straighten trajectories, but it does not imply a reward improvement by itself. The first defensible target is therefore lower curvature and preserved task return at fewer sampling steps; online return improvements are a second-stage hypothesis.

## Key Results

The released Can step-1000 checkpoint reproduced its reported behavior. Two 200-episode runs gave zero-sampling success rates of 70.5% and 71.0%; both Wilson intervals include the paper's 73.76% result. Random sampling produced 9.5% and 14.0%, or 11.75% pooled over 400 episodes, consistent with the approximately 10% low-success exploration regime highlighted by the paper. The official parallelism setting is 50 environments; pooled episode counts are retained as the authoritative metric.

The Isaac Go2 training pipeline also passed a 64-environment, 5-iteration smoke run. It wrote initial and final checkpoints plus five TensorBoard records, reached 2400 FPS after warm-up, and completed without base-contact terminations. This validates the training path but is not a learning-curve reproduction.

A 256-environment, 50-iteration Go2 validation then completed with episode length increasing from 17.3 to 905.8 and value loss decreasing from 0.0252 to 0.0121. The official 4096-environment, 1500-iteration seed is running. Separately, one official-budget Can FPO++ update completed from the released checkpoint, including collection, policy/value updates, checkpointing, and both evaluation modes. Its 20-episode post-update scores are treated only as pipeline evidence.

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

Can released-checkpoint reproduction is complete. Go2 full-seed reproduction is in progress; improvement pilots begin from the validated Can training pipeline.
