# Protocol: H34 Coordinate-Wise Median Microbatch Gradient

## Hypothesis H34

The initial FPO++ gradient varies across shuffled microbatches. Full-batch accumulation averages these
gradients, but the mean retains large coordinates produced by a minority of microbatches. Aggregating
four equal-size microbatch gradients with a coordinate-wise median should suppress such outliers and
align the first update more closely with observed terminal outcomes.

This changes only gradient aggregation. It retains official GAE, per-minibatch advantage
normalization, stored MC8 CFM variables, loss, clipping, source, and actor parameters. For four
gradients, the median is the mean of the two middle coordinate values and has no threshold or tuned
coefficient.

## Fixed Audit

- Checkpoint: released Can `95j3noe4_step_6000` EMA actor under OSMesa.
- Seed: 20260904.
- Collection: 16 environments, 320 steps, one critic-only warmup plus a fresh iteration-2 Gaussian
  rollout using official settings.
- Select 192 seeded, fully valid chunks with observed terminal labels. Split them into two independent
  96-chunk replicas, each containing four ordered 24-chunk microbatches.
- Within each microbatch, normalize GAE exactly as official non-DDP training and compute its fixed-MC8
  FPO++ gradient at the unchanged behavior policy.
- Control: arithmetic mean of the four flattened microbatch gradients.
- Candidate: coordinate-wise median of the same four gradients.
- Reference: one gradient on all 96 replica chunks using centered discounted returns to observed
  terminal outcomes and the same stored MC8 variables.

## Gates

All gates must pass in both replicas:

1. Every microbatch, control, candidate, and reference gradient is finite and nonzero.
2. Candidate cosine to the outcome-reference gradient is at least 0.75 and at least 0.10 higher than
   control.
3. Candidate cosine to the control gradient is at least 0.75.
4. Candidate norm is between 50% and 120% of control norm.
5. Candidate outcome-reference cosine is at least the median of the four individual microbatch
   cosines, ensuring aggregation is not worse than a typical constituent update.

Stop without online integration, trimming thresholds, sign voting, or microbatch-count variants if
any gate fails. A full pass authorizes a separately committed matched short-training protocol only.
