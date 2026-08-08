# Research Findings

## Research Question

Can the official FPO++ robot-control results be reproduced, and can conditional reflow make the learned policy flow straighter enough to improve return, robustness, or sampling efficiency?

## Current Understanding

FPO++ stabilizes likelihood-free policy gradients through per-CFM-sample ratios and an asymmetric trust region for from-scratch locomotion. Its policy still uses an Euler-integrated conditional flow with up to 64 steps. Rectified Flow suggests that regenerating paired endpoints from an existing flow and fitting those pairs can straighten trajectories, but it does not imply a reward improvement by itself. The first defensible target is therefore lower curvature and preserved task return at fewer sampling steps; online return improvements are a second-stage hypothesis.

## Key Results

The released Can step-1000 checkpoint reproduced its reported behavior. Two 200-episode runs gave zero-sampling success rates of 70.5% and 71.0%; both Wilson intervals include the paper's 73.76% result. Random sampling produced 9.5% and 14.0%, or 11.75% pooled over 400 episodes, consistent with the approximately 10% low-success exploration regime highlighted by the paper. The official parallelism setting is 50 environments; pooled episode counts are retained as the authoritative metric.

The official Go2 configuration reproduced successfully with 4096 environments, 1500 iterations, and seed 42. Final training return was 40.620. All 31 saved checkpoints completed post-training evaluation; the final zero-sampling return was 41.529 and random-sampling return was 40.523, matching the paper's approximately 40-return regime. Separately, one official-budget Can FPO++ update completed from the released checkpoint, including collection, policy/value updates, checkpointing, and both evaluation modes. Its 20-episode post-update scores are treated only as pipeline evidence.

An exploratory inference-only WP-Past intervention failed decisively on Can. Under a matched seed,
the Gaussian source achieved 17/20 zero-sampling and 2/20 random-sampling successes, while replacing
the source prefix with the previous action chunk produced 0/20 in both modes. This is expected input-
distribution shift for a model trained exclusively on Gaussian sources. It rules out the inexpensive
checkpoint-only intervention, not the published method, which changes the source during BC training.

A faithful short training-time WP-Past adaptation also failed its screening gate. Matched 100-update
continuations from the same released checkpoint gave 14/20 zero and 0/20 random successes for the
Gaussian control, versus 0/20 in both modes for WP-Past. The warm run used valid previous-action
histories in 95-98% of logged batches, so the failure is not explained by universal Gaussian fallback.
This refutes H3 at the allocated screening budget and shifts the main effort to conditional reflow.

One-stage conditional reflow passed both pilot gates. On 128 fixed observation/source pairs, it
reduced normalized straightness error by 22.9%, reduced four-step endpoint MSE by 27.8%, and retained
the control's action diversity. In a matched 20-episode-per-cell Can screen, reflow improved zero-
sampling success from 11/20 to 14/20 at 10 Euler steps and from 14/20 to 15/20 at four steps. Random
success pooled across the two step counts was 3/40 for both methods. These results support the claimed
straightening and provide an initial task-level improvement signal, but the rollout sample is not yet
large enough to establish a stable benchmark gain.

## Patterns and Insights

- FPO++ already uses the linear conditional flow-matching objective, so simply renaming it rectified flow is not a contribution.
- The transferable mechanism from arXiv:2209.03003 is recursive reflow using model-induced endpoint coupling.
- A reflow method that only improves inference speed is still useful, but it must not be presented as an RL performance gain.
- WarmPrior (arXiv:2605.13959) offers a more direct reward-improvement hypothesis for chunked manipulation: center the flow source on recent actions and retain residual Gaussian noise for exploration.
- WarmPrior cannot be grafted onto a Gaussian-trained FPO checkpoint at inference time; training-time source adaptation is necessary on Can.
- A short 100-update training-time adaptation is also insufficient: it preserves neither deterministic success nor random exploration.
- Conditional reflow's lower integration error transferred to a positive low-step rollout screen; this is the first improvement candidate to pass both mechanism and task gates.

## Lessons and Constraints

- Reproduce released baselines before changing objectives.
- Use short validation runs before full 1500-iteration or multi-seed jobs.
- Stop a candidate when it degrades the primary metric beyond seed noise or fails to improve its claimed mechanism.
- Keep inference-only warm starts separate from faithful WarmPrior training in claims and experiment labels.

## Open Questions

- Does conditional reflow preserve the multimodal exploration that gives flow policies their advantage?
- Is curvature correlated with FPO ratio variance or gradient disagreement?
- Should reflow be an offline post-training stage, an auxiliary online loss, or both?
- Does the positive one-stage conditional-reflow rollout screen replicate under an independent evaluation seed?

## Optimization Trajectory

Can released-checkpoint and Go2 official-seed reproduction are complete. WarmPrior screening is closed after two negative variants. Conditional reflow passed its geometry and first rollout screens and is awaiting one independent-seed confirmation before any larger experiment.
