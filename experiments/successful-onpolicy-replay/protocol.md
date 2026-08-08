# Protocol: Successful On-Policy Replay Staleness Audit

## Hypothesis

Recent successful trajectories collected from the official standard-Gaussian policy can be reused
for one additional FPO++ update without excessive stale-policy ratio error or a materially conflicting
gradient direction. This would increase the information extracted from successes while preserving
the benchmark source distribution.

## Method

Add an opt-in diagnostic to the existing FPO++ loop. Mark every action chunk belonging to an episode
that later terminates successfully, using terminal rewards and episode boundaries from the rollout.
On actor-update iteration 2, select at most 64 successful valid chunks with a fixed seeded permutation
and reuse their stored CFM times, noises, and behavior losses.

Before the official actor update, compute a unit-positive-advantage replay gradient and compare it to
the gradient from fresh positive-advantage chunks in the same rollout. After the unmodified official
update, recompute the successful-chunk CFM ratios and replay gradient. The diagnostic must use
`torch.autograd.grad`, must not call an optimizer, and must not alter model parameters or accumulated
training gradients.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights
- Training seed: 20260814
- Budget: two 48k-environment-step iterations; iteration 1 remains critic-only
- Can environments: 30; collection steps: 1600; CFM samples: 8
- Rollout source: official scale-1 Gaussian only
- Audit point: iteration 2, immediately before and after its standard FPO++ update
- Replay sample: up to 64 seeded successful valid chunks; require at least 32
- No replay optimizer step, final benchmark evaluation, or second seed at this stage

## Gates

All sampled ratios and gradients must be finite and gradients nonzero. After the standard update:

1. at least 70% of replay ratios must remain below the positive-advantage PPO upper clipping boundary
   `1 + clip_coef`, so most samples can still provide an unclipped positive update;
2. replay-ratio effective sample size must be at least 80%;
3. cosine between pre-update and post-update successful replay gradients must be at least 0.8;
4. before-update cosine between the success-only replay gradient and the same-rollout fresh
   positive-advantage gradient must be at least 0.5.

Stop without implementing replay training if any gate fails. If all gates pass, add one opt-in
one-iteration replay candidate to the existing five-update step-6000 paired screen. Do not tune replay
age, capacity, ratio clipping, replay weight, success threshold, learning rate, source distribution,
or mask behavior before that screen.
