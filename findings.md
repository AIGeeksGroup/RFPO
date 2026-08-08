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

An independent environment-seed evaluation narrowed this conclusion. Reflow again gained three
10-step zero-sampling successes (17/20 versus 14/20), yielding 31/40 versus 25/40 pooled over two
seeds. At four steps, however, the second seed reversed the one-success pilot advantage; the pooled
result was tied at 29/40. Random success remained comparable at 8/80 versus 7/80. Conditional reflow
therefore supports lower integration error and four-step performance preservation, while its apparent
official-step reward gain still needs a second training seed.

The independent training seed repeated both parts of the result. Reflow reduced normalized
straightness error by 22.4% and four-step endpoint MSE by 24.5%, with unchanged diversity. Its matched
rollout gained one 10-step zero success and three four-step zero successes. Across three comparisons
from two training seeds, reflow/control zero success was 47/60 versus 40/60 at 10 steps and 46/60
versus 43/60 at four steps; random success was tied at 13/120. This is sufficient evidence to move
from screening to one official-scale confirmation, but not yet to claim final benchmark improvement.

The official-scale confirmation separated the efficiency result from the reward claim. At 10 Euler
steps and 200 episodes, reflow/control zero success was 150/200 versus 147/200 (+1.5 percentage
points), far below the preregistered +5-point gate and statistically indistinguishable (Fisher
p=0.819). Random success was 23/200 versus 24/200. Pure one-stage reflow therefore preserves Can
performance while reliably straightening the flow, but it does not establish a benchmark reward
improvement and should not be integrated into online FPO++ on that premise.

A 50/50 mixture of dataset and teacher endpoints tested whether action grounding was the missing
ingredient. It retained an 11.0% straightness reduction and improved zero success in both small
screens, but random success changed from 5/20 versus 1/20 on the first seed to 0/20 versus 3/20 on
the confirmation seed. The deterministic signal is interesting, but unstable exploration violates
the online-policy requirement. The mixed-endpoint direction is stopped without a probability sweep.

Increasing FPO++ Monte Carlo compute did not solve the sparse-reward problem. In matched five-update
Can runs, 16 CFM samples per action produced 48/604 (7.95%) pooled collection success over actor
updates, compared with 55/603 (9.12%) for the official 8-sample control. Final 50-episode zero/random
evaluation was also slightly lower at 78%/12% versus 80%/14%. The candidate failed both the primary
+2-point improvement gate and random-sampling non-degradation gate, so larger sample counts are not
being pursued.

The rollout implementation also accumulates its `cfm_value_invalid_stored` mask across independent
iterations. Resetting the mask held valid CFM actions near 98%, whereas the released behavior fell
from 98.11% to 95.81% over five iterations. This is a real bookkeeping defect, but fixing it increased
pooled actor-update success only from 53/604 (8.77%) to 56/604 (9.27%, Fisher `p=0.841`). Final
zero/random evaluation was 82%/10% versus 84%/10%. Preserving these additional samples is correct,
but it does not establish a useful short-budget reward gain.

## Patterns and Insights

- FPO++ already uses the linear conditional flow-matching objective, so simply renaming it rectified flow is not a contribution.
- The transferable mechanism from arXiv:2209.03003 is recursive reflow using model-induced endpoint coupling.
- A reflow method that only improves inference speed is still useful, but it must not be presented as an RL performance gain.
- WarmPrior (arXiv:2605.13959) offers a more direct reward-improvement hypothesis for chunked manipulation: center the flow source on recent actions and retain residual Gaussian noise for exploration.
- WarmPrior cannot be grafted onto a Gaussian-trained FPO checkpoint at inference time; training-time source adaptation is necessary on Can.
- A short 100-update training-time adaptation is also insufficient: it preserves neither deterministic success nor random exploration.
- Conditional reflow's lower integration error transferred to a positive low-step rollout screen; this is the first improvement candidate to pass both mechanism and task gates.
- Small 20-episode screens overestimated pure reflow's reward effect; the 200-episode result retained only +1.5 points while confirming non-degradation.
- Mixing real and reflow endpoints can retain partial straightening and deterministic gains, but does not reliably preserve random-source exploration.
- Doubling CFM samples does not create more reward information; in this pilot it added compute while pooled early success fell by 1.17 points.
- CFM invalid-step state must be reset between independent rollouts, but the resulting 2.3-point increase in valid samples by iteration 5 was not enough to improve Can reward materially.

## Lessons and Constraints

- Reproduce released baselines before changing objectives.
- Use short validation runs before full 1500-iteration or multi-seed jobs.
- Stop a candidate when it degrades the primary metric beyond seed noise or fails to improve its claimed mechanism.
- Keep inference-only warm starts separate from faithful WarmPrior training in claims and experiment labels.

## Open Questions

- Does conditional reflow preserve the multimodal exploration that gives flow policies their advantage?
- Is curvature correlated with FPO ratio variance or gradient disagreement?
- Should reflow be an offline post-training stage, an auxiliary online loss, or both?
- Which reward-aware FPO++ mechanism can improve Can success without relying on unstable BC-source exploration changes?

## Optimization Trajectory

Can released-checkpoint and Go2 official-seed reproduction are complete. WarmPrior, mixed-endpoint reflow, increased Monte Carlo sampling, and validity-mask reset screening are closed. Pure conditional reflow is supported as a geometry and sampling-efficiency method but refuted as a five-point Can reward improvement. The next direction must strengthen long-horizon sparse credit assignment rather than add BC distillation, sample reuse, or Monte Carlo compute.
