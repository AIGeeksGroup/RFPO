# Protocol: H30b Balanced Zero-Endpoint PCGrad Reward Screen

## Rationale

H30 training is valid, but its first evaluation used 20 requested episodes across 50 asynchronous
environments. Successful Can episodes finish earlier than failures, so the first 20 completions were
censored and cannot compare the trained policies. H30b reuses the frozen H30 checkpoints and changes
only episode accounting.

## Fixed Evaluation

- Checkpoints: final non-EMA actors from the completed seed-20260828 H30 control and candidate runs.
- Modes: official 10-step zero source and scale-1 Gaussian source.
- Seed: 20260906 shared across both conditions and modes.
- Budget: 20 environments and exactly one completed episode per environment, for 20 balanced episodes
  per condition per mode.
- Renderer: the same OSMesa stack used by both conditions.
- No retraining, checkpoint selection, anchor changes, or evaluation retries.

## Gates

Retain the original H30 reward gates:

1. candidate random-source success exceeds control by at least 2/20;
2. candidate zero-source success is no worse than control by more than 1/20;
3. candidate pooled success over the 40 zero/random episodes exceeds control by at least 2 episodes;
4. the already-recorded candidate projection-active fraction remains between 10% and 90%.

Stop if any gate fails. If all pass, run one independently seeded balanced confirmation with 50
environments and one episode per environment per condition/mode before making a benchmark-improvement
claim.
