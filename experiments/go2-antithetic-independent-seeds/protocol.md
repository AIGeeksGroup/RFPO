# H69: Independent-Training-Seed Go2 Confirmation

## Question

Does the equal-compute antithetic advantage generalize to independently trained
official Go2 policies, and does it improve over deterministic zero-source
deployment on average?

## Training

Train three new policies with seeds 43, 44, and 45 using the unmodified official
Go2 configuration:

- 4,096 parallel environments
- 1,500 iterations
- official FPO++ hyperparameters and Euler-64 inference configuration
- fixed final checkpoint `model_1499.pt`; no checkpoint selection
- disable the runner's post-training sweep over all 31 saved checkpoints because
  H69 performs its separately seeded fixed-final evaluation; this flag does not
  affect training or checkpoint creation

Training must finish with finite metrics and a complete final checkpoint. No run
is excluded for low return.

## Evaluation

For each trained seed, evaluate `zero`, `random`, `iid_pair`, and `antithetic`
with 512 environments per method and one episode per environment. Use:

| Train seed | Evaluation seed | Primary source seed | Secondary source seed |
|---:|---:|---:|---:|
| 43 | 20261243 | 20261343 | 20261443 |
| 44 | 20261244 | 20261344 | 20261444 |
| 45 | 20261245 | 20261345 | 20261445 |

The bootstrap seed is `20261500` with 20,000 stratified paired resamples. Within
each training seed, all methods must share initial-observation hashes and all
stochastic methods must share the complete primary source stream. IID and
antithetic must each use exactly 128 NFE/action; zero and random use 64.

## Decisions

H69's primary equal-compute claim passes only if `antithetic - iid_pair` is
positive in all three new training seeds and its pooled stratified bootstrap 95%
lower bound is above zero. `Antithetic - random` must also have a positive pooled
lower bound to retain the official stochastic-source improvement claim.

The stronger deployment claim is evaluated separately: it passes only if the
pooled stratified `antithetic - zero` 95% lower bound is above zero. Failure of
this secondary gate restricts the paper claim to stochastic-source robustness
and equal-compute attribution; it does not invalidate H69's primary hypothesis.

The existing seed-42 H67 result is reported descriptively but is not included in
the three-seed confirmatory gates. Cross-task experiments are authorized only if
the primary H69 claim passes.

## Archival reanalysis

The analysis CLI may expose an explicit archival mode so the committed JSON
artifacts can be reanalyzed on a machine that does not hold the remote training
checkpoints. Archival mode must still require a fixed `model_1499.pt` checkpoint
name, a lowercase 64-character SHA-256 digest, a positive recorded checkpoint
size, distinct digests across training seeds, and exact agreement between each
evaluation artifact and its training manifest. It may skip only the local file
existence, byte-size, and digest recomputation checks.

The default analysis mode remains the validation path for fresh experiments: it
must require each checkpoint file, compare its byte size with the manifest, and
recompute its SHA-256 digest. Archival mode does not create new empirical
evidence or change any statistical decision; it only makes the committed H69
analysis reproducible from its archived records.
