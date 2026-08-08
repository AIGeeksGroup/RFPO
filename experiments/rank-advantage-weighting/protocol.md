# Protocol: Rank-Preserving Advantage Weight Audit

## Hypothesis

Official GAE already orders fresh Can outcomes well, but its learned magnitudes can overemphasize
poorly calibrated chunks. Replacing magnitudes with centered empirical percentile ranks will retain
all GAE ordering information while improving the first FPO++ actor-gradient direction toward
uncensored discounted Monte Carlo outcomes.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260820; official scale-1 Gaussian source
- Budget: 8 environments, 320 collection steps, and two iterations; iteration 1 remains the released
  critic-only warmup
- Actor: frozen for the iteration-2 audit; candidate weights are never used for an optimizer step
- Eligible data: fully valid iteration-2 chunks with an observed terminal before rollout truncation
- Control weights: unchanged raw official GAE advantages
- Candidate transform: compute stable average ranks over all eligible control weights, including
  average ranks for exact ties, then map ranks linearly to `[-1, 1]`. This transform is centered and
  strictly preserves every non-tied ordering.
- Reference weights: uncensored discounted Monte Carlo returns centered over the complete eligible
  audit set
- Gradient audit: two seeded, disjoint batches of 32 chunks using their rollout-stored MC8 CFM
  variables
- Actor-gradient parameters: all trainable non-vision actor parameters

## Gates

Require finite weights and gradients, at least 64 eligible chunks, both signs and nonzero variance
in the candidate weights, a nonzero reference gradient in each batch, and exact non-tied order
preservation. Both gradient batches must pass:

1. candidate actor-gradient cosine to the Monte Carlo reference is positive;
2. candidate cosine is at least 0.05 higher than the control cosine.

Stop without online integration if either batch fails. If both pass, add rank weighting as an
opt-in mode and run one matched five-update step-6000 screen. Do not tune the rank formula, scope,
batch selection, seed, chunk count, or gates after observing the audit.
