# Outer Loop 2: Candidate Mechanisms

The observed failure is specific: successful zero or tempered rollouts are plentiful, but their
source distribution does not match the official standard-Gaussian policy. The next method must
improve how valid on-policy successes affect the actor, not merely collect more guided successes.

## Diverged Candidates

1. ESS-normalized exponential advantages inside the FPO++ ratio objective.
2. Positive-only success-trajectory CFM auxiliary loss.
3. Successful-episode replay with a frozen-policy velocity penalty.
4. Conditional-reflow auxiliary regularization during FPO++ updates.
5. Reflow-conditioned reduction of CFM Monte Carlo gradient disagreement.
6. Exact source-mixture importance correction using known Gaussian noise densities.
7. Multi-candidate action ranking with a new Q critic, following full FMER.
8. Analytic flow-entropy regularization to protect random-source diversity.
9. Adaptive positive/negative PPO clipping based on success scarcity.
10. Curriculum from tempered to full noise with an explicitly changed target policy.
11. Higher-order integration during rollout to reduce action endpoint error.
12. Success-conditioned offline BC followed by standard FPO++ recovery.

## Convergence

Candidate 1 ranks first. It directly addresses sparse on-policy success weighting, needs no new
critic, preserves the official rollout distribution, and can be falsified in the existing five-update
gate. Candidate 2 is similar but introduces an arbitrary auxiliary coefficient. Candidate 4 retains
the positive geometry finding, but pure reflow did not improve reward and should first show a link
to gradient disagreement. Candidates 3, 7, 8, and 12 require substantially heavier infrastructure.
Candidate 6 is principled but would assign vanishing target-policy weight to concentrated guided
samples in the 112-dimensional action-chunk source. Candidates 9-11 lack direct evidence for the
observed transfer failure.

The selected method uses normalized `exp(A/tau)` weights, with `tau` chosen by bisection to a fixed
50% effective-sample-size fraction. It replaces the signed advantage in the actor surrogate while
retaining FPO++ per-sample ratios and PPO clipping. No temperature or ESS sweep is allowed in the
pilot.
