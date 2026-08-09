# Outer-Loop Cycle 31: Stabilize Actor Minibatch Composition

## Reflection

H42-H44 repeatedly moved zero- and Gaussian-source success in opposite directions. In particular,
rollout-local Adam and actor-trajectory SWA favored the deterministic mode while hurting stochastic
source behavior. The next intervention should leave source sampling, policy weights at inference,
and the FPO++ objective unchanged while directly reducing optimization noise inside Gaussian-source
training.

The released actor loop randomly shuffles all chunks before eight sequential Adam steps, and it
normalizes advantages independently inside each minibatch. H32 found that positive- and negative-GAE
gradients are nearly orthogonal rather than strongly conflicting; therefore deleting or projecting a
branch is not justified. The remaining testable question is whether uneven sign composition across
sequential minibatches destabilizes the update path.

## Candidates

| Rank | Candidate | Rationale | Decision |
|---:|---|---|---|
| 1 | Advantage-sign-stratified actor minibatches | exact sample coverage; stabilizes each Adam step and local normalization without reweighting | Select H45 |
| 2 | Return-success-balanced minibatches | uses a much sparser binary label and risks implicit reward oversampling | Reject |
| 3 | Advantage-quantile stratification | introduces arbitrary bin count and magnitude boundaries before sign stratification is tested | Park |
| 4 | Source-norm-balanced minibatches | conditions optimization on inference noise magnitude and repeats source-specific interventions | Reject |
| 5 | Online EMA policy | H44 already refuted within-trajectory averaging and EMA adds a decay hyperparameter | Reject |
| 6 | Entropy-preserving auxiliary | Appendix D.5 reports ASPO as detrimental in pretrained manipulation | Reject |

## Selection

H45 partitions the complete actor-update chunk set by the sign of its raw first-step GAE, shuffles
within each partition, and distributes positive chunks across all eight fixed-size minibatches so
their positive counts differ by at most one. Every index appears exactly once per epoch. The
critic-only first iteration retains the official random shuffle, allowing exact pairing with the
frozen H43 control through iteration-2 collection.
