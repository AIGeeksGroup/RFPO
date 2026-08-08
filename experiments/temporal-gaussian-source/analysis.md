# H31 Analysis: Temporal Gaussian Source

## Outcome

H31 is refuted. The intervention passed every mechanism gate but failed the locked balanced reward
screen decisively, so no confirmation run or `rho` tuning is permitted.

## Mechanism Audit

- Source mean: 0.00405 (gate: absolute value at most 0.03).
- Source standard deviation: 0.99895 (gate: 0.97 to 1.03).
- Lag-one source correlation: 0.89986 (target: 0.9, tolerance: 0.03).
- Mean action-chunk change: 2.55754 IID versus 0.81765 AR(1), a 68.03% reduction
  (gate: at least 10%).
- Action-diversity retention: 75.25% (gate: at least 70%).
- All audited sources and actions were finite.

## Balanced Reward Screen

The corrected protocol used 20 environments and exactly one completed episode from each environment.
At seed 20260831, the official IID source scored 13/20 (65%), consistent with the prior 34/50 (68%)
step-6000 baseline. The `rho=0.9` AR(1) source scored 0/20, missing the required +2/20 gate by 15
successes relative to control. Every candidate episode ran to the 299-step horizon, and evaluation
completed without NaN or infrastructure errors.

## Interpretation

AR(1) preserves the standard-normal marginal at each replan, but it does not preserve the joint
distribution across an episode. At high correlation, the first stochastic source strongly constrains
later sources. The policy therefore receives less independent opportunity to change its behavior mode
after new observations. Lower boundary jitter is real, but in this task it is harmful persistence rather
than useful temporal coherence.

This result closes temporal-source smoothing as an inference improvement. Sweeping the correlation
would violate the locked stopping rule and would revisit a core hypothesis already contradicted by a
65-point absolute degradation.
