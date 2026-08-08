# Protocol: H37 Geometric Median-of-Means Gradient

## Hypothesis H37

FPO++ microbatch gradients contain heavy-tailed block outliers, but their within-vector parameter
structure is useful. Aggregating four equal-size block gradients with their geometric median should
improve alignment to observed terminal outcomes without the joint-structure destruction measured for
H34's coordinate-wise median.

## Locked Method

- Retain official GAE, per-minibatch advantage normalization, stored MC8 CFM variables, ratio loss,
  clipping, source distribution, actor parameters, and arithmetic-mean control.
- Split each replica into four ordered, equal 24-chunk blocks and compute one flat gradient per block.
- Candidate is the minimizer of the sum of Euclidean distances to the four complete gradient vectors.
- Compute it with Weiszfeld iterations initialized at their arithmetic mean, stopping at relative
  iterate displacement `1e-6` or after 100 iterations. Use `1e-12` only to handle exact zero distance.
- Do not normalize block gradients, tune the iteration rule, change the number of blocks, combine
  GMOM with clipping changes, or include H34's coordinate-wise median as a selectable variant.

## Fixed Audit

- Checkpoint: released Can `95j3noe4_step_6000` EMA actor under OSMesa.
- Seed: `20260908`.
- Collection: 16 environments, 320 steps, one critic-only warmup plus a fresh iteration-2 official
  Gaussian rollout; total budget 10,240 environment steps.
- Select 192 seeded, fully valid chunks with observed terminal labels. Split them into two independent
  96-chunk replicas, each containing four ordered 24-chunk blocks.
- Reference: one gradient on all 96 replica chunks using centered discounted returns to observed
  terminal outcomes and the same stored MC8 variables.
- Control: arithmetic mean of the four block gradients. Candidate: their full-vector geometric median.

## Gates

All gates must pass in both replicas:

1. Every block, control, candidate, and reference gradient is finite and nonzero; the geometric
   median converges within 100 iterations.
2. Candidate cosine to the outcome-reference gradient is at least `0.75` and at least `0.10` higher
   than control.
3. Candidate cosine to the control gradient is at least `0.75`.
4. Candidate norm is between 50% and 120% of control norm.
5. Candidate outcome-reference cosine is at least the median of the four constituent block cosines.

Stop without optimizer integration, block-count changes, gradient normalization, approximate-median
variants, another seed, or online reward training if either replica fails any gate. A full pass only
authorizes a separately committed matched short-training protocol.

