# H45 Square Advantage-Sign-Stratified Minibatch Analysis

## Validity

The two-iteration smoke exercised one complete actor update. Its ten audit records each covered 40
unique chunks exactly once in eight five-chunk minibatches; raw-positive counts were two or three per
minibatch, and the final checkpoint was finite.

The formal candidate completed five iterations with finite losses and checkpoints. Iteration-1/2
collection exactly matched the frozen H43 control: successes were `[2, 9]` and valid-CFM counts were
`[5097, 5036]`. Across 40 actor epochs in iterations 2-5, every epoch covered all 320 unique chunks
once in eight 40-chunk minibatches. The raw-positive count range was exactly one in every epoch, and
both signs occurred in every minibatch. All preregistered validity gates passed.

## Reward Results

| Mode | Official-shuffle control | Sign-stratified candidate | Difference | Gate | Result |
|---|---:|---:|---:|---:|---|
| zero | 8/20 (40%) | 7/20 (35%) | -1 | no worse than -1 | pass |
| random | 4/20 (20%) | 7/20 (35%) | +3 | at least +2 | pass |
| pooled | 12/40 (30%) | 14/40 (35%) | +2 | at least +3 | fail |

The candidate produced the desired Gaussian-source direction without the larger modal-policy shift
seen in H43-H44. However, the one-success zero loss left pooled improvement one success below the
locked confirmation threshold. A 40-episode paired screen is too small to reinterpret this near miss
after observing it.

## Decision

H45 is refuted under its preregistered joint gates. Do not run another seed or confirmation and do
not try return-label, quantile, batch-count, longer-training, checkpoint-selection, or normalization
variants. Retain the result as evidence that minibatch composition can move Gaussian-source reward,
but not as a stable benchmark gain.
