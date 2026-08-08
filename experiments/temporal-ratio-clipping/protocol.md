# Protocol: H33 Temporal Per-Sample Ratio Clipping

## Hypothesis H33

Manipulation FPO++ sums CFM losses across all 16 action-chunk timesteps before exponentiating and
clipping the ratio. Thus, one shifted timestep can clip the gradient contribution of every other
timestep in the chunk. Retaining a separate ratio for each valid action timestep and clipping each
independently should provide a finer trust region and preserve the initially useful GAE gradient
after an actor step.

The temporal surrogate sums its per-timestep contributions before averaging over chunks and MC
samples. At the behavior policy, its gradient must therefore equal the official summed-loss-ratio
gradient in both direction and norm. There is no new coefficient and invalid padded timesteps do not
contribute.

## Fixed Audit

- Checkpoint: released Can `95j3noe4_step_6000` EMA actor under OSMesa.
- Seed: 20260903.
- Collection: 16 environments, 320 steps, one critic-only warmup plus a fresh iteration-2 Gaussian
  rollout, with official MC8, 10 Euler steps, and 16-step predicted chunks.
- Select 64 seeded, fully valid chunks with observed terminal labels and split into two fixed
  32-chunk batches. Normalize GAE independently within each batch as in official minibatch training.
- Reuse the identical stored CFM times and noises for control and candidate.
- Control: one ratio per chunk from the sum of 16 timestep losses.
- Candidate: 16 independently clipped timestep ratios, with timestep surrogate terms summed.
- For each batch, compute both gradients at the behavior policy. Then apply the same clipped virtual
  SGD step along their common initial gradient, using actor LR `1e-5` and max norm 5, and recompute
  both gradients and positive-advantage active-ratio fractions at the shared updated parameters.

## Gates

All gates must pass in both batches:

1. Pre-update candidate/control gradient cosine at least 0.999 and norm ratio in [0.99, 1.01].
2. Candidate post-step gradient cosine to the common pre-update gradient at least 0.75 and at least
   0.10 higher than control.
3. Candidate positive active-ratio fraction at least 0.10 higher than control.
4. Candidate post-step gradient norm between 70% and 130% of control.
5. All ratios, losses, and gradients finite and nonzero.

Stop without online integration or temporal-group variants if any gate fails. Only a full pass
authorizes a matched short training protocol committed separately before any reward rollout.
