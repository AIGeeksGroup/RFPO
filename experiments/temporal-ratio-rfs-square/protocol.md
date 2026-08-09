# Protocol: H49 Temporally Factorized RFS Ratios

## Hypothesis

RFS failed partly because its exact 224-dimensional joint PPO ratio clipped 72.6-76.0% of samples.
Clipping the explicit latent/residual likelihood ratio separately at each of the 16 action times,
then summing the temporal objectives, preserves the exact on-policy policy gradient while preventing
unrelated time steps from jointly deactivating the entire action chunk.

## Locked Mechanism Screen

- Frozen control: H48 formal iteration 1 at seed `20260924`, with 323 transitions, 4/4 collection
  success, finite updates, and joint clip fraction `0.7257500066`.
- Candidate: the same released EMA actor, environment seed, RFS initialization, 16 environments,
  320-step rollout, 323 expected transitions, network, optimizer, PPO settings, and one update.
- Sole method change: compute latent-plus-residual log probability per action time, apply PPO 0.2
  clipping to each temporal ratio, and sum the 16 clipped objectives.
- A unit test must establish equality of joint and temporal policy gradients at the on-policy point.
- Base actor remains bitwise frozen and both modulation branches must update with finite metrics.

H49 passes the mechanism screen only if the pre-update rollout exactly reproduces H48's 323
transitions and 4/4 success, temporal clip fraction is at least 20 percentage points below the frozen
joint control, and all validity checks pass. Failure stops without grouping, clip, epoch, LR, scale,
or seed variants.

## Conditional Reward Screen

If the mechanism passes, run the locked five-iteration 25,600-step candidate with temporal ratios
and no other changes. Compare its final policy against the already frozen H48 controls at evaluation
seed `20260925`: 10/20 mean and 10/20 sampled. Evaluate candidate mean then sampled, 20 environments
contributing one episode each.

H49 passes only with at least 23/40 pooled success, at least 12/20 sampled success, and at least
9/20 mean success. A pass authorizes independent confirmation; failure stops without another seed,
checkpoint selection, longer training, or temporal-grouping variants. OSMesa remains a paired screen.
