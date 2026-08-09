# Outer Loop Cycle 38: Direct Episodic Search Beyond FPO++ Gradients

## Failure Boundary

H0-H52 repeatedly separate useful behavior from a reliable FPO++ update. Curvature, source priors,
CFM Monte Carlo estimators, advantage transformations, critics, clipping, optimizer schedules,
gradient surgery, robust aggregation, residual policies, reward shaping, and parameter restriction
can change their intended mechanisms, but none yields a stable confirmed reward gain. H52 is the
clearest boundary: restricting the parameter subspace preserves direction and behavior by about 99%
while retaining only about 2% of the useful surrogate gain. Continuing to regularize the same
critic/GAE/CFM-ratio gradient is unlikely to resolve the core reward-estimation problem.

## Diverged Candidates

1. Fixed-subspace mirrored evolution strategies on complete Square episode success.
2. Full-parameter isotropic evolution strategies.
3. Augmented random search on only the final output layer.
4. CMA-ES over a learned low-dimensional actor adapter.
5. Bayesian optimization over checkpoint interpolation coefficients.
6. Cross-entropy search over Gaussian source mean and scale.
7. Population-based training over FPO++ learning rate and clip threshold.
8. Simultaneous perturbation stochastic approximation on the actor.
9. REINFORCE on explicit parameter-noise seeds.
10. Direct policy-gradient estimation from common-random-number action perturbations.
11. A stage-classification auxiliary head on frozen visual features.
12. Contrastive success/failure representation learning before FPO++.
13. Model-predictive selection with a learned terminal success classifier.
14. Offline decision-transformer fine-tuning on the accumulated Square rollouts.

## Convergence

Select H53, fixed-subspace mirrored parameter search. It addresses the measured failure directly:
complete episodic success supplies the objective, so there is no critic, temporal discount, GAE,
CFM likelihood surrogate, or action-frequency dependence. A fixed 16-direction subspace makes the
high-dimensional search falsifiable and reusable; mirrored perturbations and common environment
seeds reduce variance. Salimans et al. (2017) establish ES as a delayed-reward, value-free alternative
to policy gradients, and Mania et al. (2018) show that simple parameter-space random search can be
competitive on continuous-control benchmarks.

Reject full isotropic ES before signal evidence because 13 million dimensions would require far more
directions. Output-layer search is too close to H52's nearly inert subspace, source-space search and
explicit adapters overlap H48-H49, and population tuning violates the project's mechanism-first
budget discipline. Learned classifiers or sequence models reintroduce the value/data prerequisites
that failed H15/H19/H50. H53 therefore begins with a no-update cross-seed signal audit; binary reward
that cannot rank fixed perturbation directions is an immediate stop condition.

Two-sentence pitch: sparse-reward FPO++ currently relies on an unstable critic/GAE/implicit-ratio
chain, so stabilizing its gradient repeatedly removes useful learning. H53 searches a fixed actor
parameter subspace using paired complete-episode success, which is delayed-reward invariant and
requires neither a critic nor a flow-policy density.
