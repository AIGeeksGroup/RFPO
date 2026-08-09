# Outer Loop Cycle 36: Credit Without a Learned Action Value

## Evidence Being Explained

Square has non-saturated success and reproducible sparse-reward headroom. Frozen-policy curvature
ranking, source smoothing, residual steering, longer rollouts, and a simulator-state critic all
failed. H50 specifically shows that adding privileged state to a short-budget value model does not
reliably improve held-out return ranking. The next candidate should improve the reward information
seen by the existing on-policy pipeline without requiring an action-value critic or changing the
Gaussian flow source.

## Diverged Candidates

1. Evaluate FPO++ with its existing DPPO fixed-noise SDE sampler.
2. Learned-noise DPPO initialized from the released Square checkpoint.
3. Best-of-three Gaussian chunks selected by action-space medoid distance.
4. Candidate selection by forward/reverse ODE cycle consistency.
5. Candidate selection by distance to the zero-source endpoint.
6. Temporal overlap selection against the preceding action chunk.
7. Episodic novelty bonuses from frozen visual features.
8. Random-network-distillation bonuses on simulator state.
9. Raw RoboSuite reach/grasp/lift/hover dense rewards.
10. Potential-based shaping from RoboSuite stage progress.
11. Stage classification as an auxiliary critic objective.
12. Full RLDT with double-Q replay and SVGD transport.

## Convergence

- Fixed and learned SDE sampling are the paper's already-tuned DPPO baselines, and both underperform
  FPO++. They are not a new FPO++ or Rectified Flow intervention.
- Medoid, cycle-consistency, and zero-distance selection are confidence-ranking variants adjacent to
  H26/H46 and tend to contract rather than preserve Gaussian exploration.
- Temporal overlap repeats H31's premise; H31 reduced chunk jitter by 68% but collapsed random Square
  success.
- Novelty bonuses introduce an arbitrary reward coefficient and may reward task-irrelevant visual or
  simulator novelty before any evidence that novelty is the bottleneck.
- Raw dense stage rewards can make prolonged hovering more valuable than insertion under repeated
  per-step accumulation.
- Stage classification and full RLDT reintroduce learned value prerequisites that H50 did not support;
  the latter also lacks public code and has a much larger compute budget.

## Selected Direction

Select H51: potential-based stage shaping with
`r' = r_sparse + gamma * Phi(s_next) - Phi(s)`, unit coefficient, and terminal potential zero. `Phi`
is RoboSuite's existing maximum reach/grasp/lift/hover stage reward. This directly addresses sparse
credit, preserves the original success objective under the discounted-MDP assumptions, leaves policy
inputs and Gaussian flow sampling unchanged, and can be rejected with a frozen-policy signal audit
before short paired training.

