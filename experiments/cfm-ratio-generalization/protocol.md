# Protocol: Stored-versus-Held-Out CFM Ratio Generalization

## Hypothesis

Ten FPO++ update epochs overfit the eight CFM time/noise draws stored during rollout, so an independent
MC8 estimate on the same positive-advantage action chunks changes less and retains more unclipped
policy-gradient signal after the update.

## Method

Extend the opt-in iteration-2 audit without changing the optimizer. Select 64 seeded valid chunks
with positive fresh on-policy advantages. The control variables are their stored rollout MC8 times,
noises, and behavior losses. Before the actor update, independently draw one held-out MC8 set with a
separate fixed generator and compute its behavior losses under the unchanged rollout policy. Run the
official ten-epoch FPO++ update using only the stored draws. Then compute stored and held-out ratios
and unit-positive-advantage gradients under the updated actor.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights
- Training seed: 20260815
- Budget: two 48k-environment-step iterations; iteration 1 remains critic-only
- Can environments: 30; collection steps: 1600; CFM samples: 8
- Rollout source: official scale-1 Gaussian only
- Audit point: iteration 2, 64 seeded positive-advantage valid chunks; require at least 32 available
- Held-out generator seed: 20260815 plus a fixed audit offset
- No held-out sample enters the official policy update

## Gates

All losses, ratios, and pre-update gradients must be finite and nonzero. The fixed-draw overfitting
mechanism is supported only if, after the official update:

1. held-out positive-active ratio fraction exceeds the stored-draw fraction by at least 15 points;
2. held-out median absolute log-ratio is at most 80% of the stored-draw value;
3. the held-out post-update replay gradient is nonzero and its cosine to the held-out pre-update
   gradient is at least 0.8;
4. held-out ratio ESS is at least 80%.

Stop without implementing epoch resampling if any gate fails. If all gates pass, implement a frozen
behavior actor that recomputes old losses on fresh MC8 draws per epoch, keeping eight samples per
forward, and run one paired five-update step-6000 screen. Do not tune epoch count, sample count,
learning rate, clipping, source distribution, or held-out seed after this audit.
