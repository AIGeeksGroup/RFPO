# Protocol: H45 Square Advantage-Sign-Stratified Minibatches

## Hypothesis

Balancing raw positive- and non-positive-GAE chunks across each actor minibatch reduces update-path
noise from random minibatch composition and improves Gaussian-source Square success without
degrading zero-source behavior.

## Locked Training

- Initialization: released Square `trc7rbt0_step_110000`, EMA actor.
- Control: frozen official persistent-Adam H43 control
  `square_h43_control_osmesa_seed20260916`.
- Candidate seed: 20260916, five actual iterations, 16 environments, 320 collection steps, Euler-10,
  16 executed actions, MC8, eight minibatches, ten epochs, actor LR `1e-5`, and all remaining H43
  official settings unchanged.
- Iteration 1 is critic-only and retains the official random shuffle.
- In actor-active iterations, define a chunk as positive when its raw first-timestep GAE is greater
  than zero. Shuffle positive and non-positive partitions, assign all chunks without replacement to
  the eight existing fixed-size minibatches, make positive counts differ by at most one, then shuffle
  within each minibatch.
- Do not resample, discard, duplicate, or reweight chunks. Keep minibatch-local advantage
  normalization, actor/critic objectives, optimizer state, and scheduler unchanged.
- OSMesa is accepted for this paired screen only and is not an official benchmark claim.

## Validity Gates

1. Candidate iteration-1 and iteration-2 collection records must exactly match the frozen H43
   control successes and valid-CFM counts: successes `[2, 9]`, valid counts `[5097, 5036]`.
2. Every actor epoch must contain all 320 unique chunk indices exactly once in eight 40-chunk
   minibatches.
3. In every actor epoch, maximum minus minimum raw-positive count across minibatches must be at most
   one, with both signs represented in every minibatch.
4. Training and the final non-EMA checkpoint must remain finite.

## Locked Evaluation and Reward Gates

- Compare the frozen H43 final control checkpoint with the candidate final checkpoint.
- Square, Euler-10, 16 executed actions, seed 20260920 shared across all cells.
- Evaluate zero and independent standard-Gaussian sources separately using 20 environments and
  exactly one episode per environment.
- Require all of: candidate random success at least `+2/20`; zero success no worse than `-1/20`;
  pooled success at least `+3/40`.

Failure closes sign-stratified minibatches without return-label or quantile strata, altered batch
count, another seed, longer training, checkpoint selection, or nearby normalization variants.
Success requires a separately registered 50-episode-per-mode confirmation.
