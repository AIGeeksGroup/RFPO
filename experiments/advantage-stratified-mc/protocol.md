# Protocol: Advantage-Stratified CFM Monte Carlo Audit

## Hypothesis

At a fixed average of eight CFM samples per action chunk, assigning twelve samples to the upper half
of absolute-advantage chunks and four to the lower half estimates the reward-weighted FPO++ policy
gradient more accurately than uniform MC8.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260816; official scale-1 Gaussian source
- Data: iteration-2 fresh on-policy chunks after the critic-only first iteration
- Audit: two seeded, disjoint batches of 32 valid chunks
- Weights: signed rollout advantages, with no extra normalization or clipping change
- Control: 8 independent CFM samples for every chunk, 256 total samples per batch
- Candidate: 12 independent samples for the upper 16 chunks by absolute advantage and 4 for the
  lower 16 chunks, 256 total samples per batch
- Reference: an independent MC64 estimate for every chunk
- Repeats: 8 independent fixed-budget estimates per method and batch
- Parameters: all trainable non-vision actor parameters
- No candidate sample enters the official policy update

For every sample set, compute the behavior loss and differentiable current loss using the same CFM
times and noises. Average ratios within each chunk before applying its signed advantage, so the 12/4
allocation changes estimator variance but not chunk weighting.

## Gates

For each of both 32-chunk batches, require finite, nonzero gradients and:

1. at least 15% lower mean normalized gradient squared error to the MC64 reference;
2. mean gradient cosine to the MC64 reference no lower than uniform MC8;
3. candidate-versus-control repeat-average gradient cosine at least 0.99.

Stop without online training if any gate fails. If all gates pass, implement the locked 12/4
allocation and run one matched five-update step-6000 screen. Do not tune the split, sample counts,
advantage transform, reference size, or seed after observing this audit.
