# Outer Loop Cycle 39: Closed-Loop Use of Overlapping Action Predictions

## Failure Boundary

H0-H53 close broad training-time changes to the source, flow coupling, estimator, critic, optimizer,
gradient, explicit residual policy, reward signal, trainable subspace, and direct parameter search.
The recurring failure is not inactivity: many interventions improve their local geometry, stability,
credit, or likelihood target but trade away Gaussian-source behavior or retain too little useful
policy movement. H53 additionally shows that complete binary success cannot reproducibly rank local
parameter perturbations at the available budget.

## Diverged Candidates

1. Ensemble the previous prediction's unexecuted continuation with the new prediction's overlapping
   prefix at each official eight-step chunk boundary.
2. Learn a state-dependent gate between old and new overlapping chunks.
3. Use action-prediction disagreement as a trigger for early replanning.
4. Execute the old continuation only when the new prediction is close.
5. Fit an offline boundary-correction residual from successful trajectories.
6. Average three generations from the current observation.
7. Use a phase classifier to vary the number of executed actions.
8. Project the new chunk onto a bounded acceleration set.
9. Use a short model-predictive action smoother with simulator rollouts.
10. Distill overlapping predictions into a recurrent policy.

## Convergence

Select H54, a fixed ACT-inspired temporal ensemble at chunk boundaries. It exploits information that
the released policy already computes and then discards: actions 8-15 from a 16-action prediction.
Unlike H27, it does not increase replanning frequency. Unlike H26, it combines predictions from
different observations rather than antithetic sources under one observation. Unlike H31, it does not
correlate Gaussian sources across episodes or replans. The intervention changes neither weights nor
inference-call count and has a direct falsifiable mechanism: the overlap should reduce boundary jump
while retaining a nontrivial coefficient on the new visual prediction.

Reject a learned gate, phase classifier, offline residual, or distillation stage before this fixed
mechanism is supported because each adds training data and a new fitted model. Reject thresholded
replanning and acceleration projection because they introduce arbitrary task-dependent thresholds.
Reject current-observation multi-sampling because H26 and H46 already show that averaging or ranking
same-observation flow samples does not transfer reliably to reward.

Two-sentence pitch: chunked flow policies throw away the second half of every predicted horizon even
though it overlaps the next visual replan. H54 reuses that continuation through a fixed ACT-inspired
ensemble, aiming to smooth only the chunk boundary without reducing feedback cadence or adding policy
calls.
