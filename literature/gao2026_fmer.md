# Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints

- Authors: Ting Gao et al.
- Year: 2026
- Identifier: arXiv:2603.17685v3
- URL: https://arxiv.org/abs/2603.17685
- Retrieved: 2026-08-08 from the arXiv API and ar5iv full text

## Relevant Mechanism

FMER minimizes CFM regression under normalized exponential weights
`w(s,a) = softmax(A(s,a) / tau)`. This is a surrogate for a mirror-descent policy-improvement
target. Its normalization keeps total update mass constant per state, and the paper reports that
unnormalized weights and hard top-1 selection converge prematurely. FMER also adds explicit entropy
control and normally ranks multiple actions with a Q critic.

## Relevance to FPO++

The full FMER algorithm would require a new action-value critic, multi-candidate collection, and
entropy estimator. The minimal testable transfer is its normalized exponential weighting: apply it
to standard-Gaussian on-policy Can actions, select the temperature from a fixed effective-sample-size
target, and retain FPO++'s existing per-CFM-sample PPO trust region. This isolates whether soft
success prioritization improves sparse-reward learning before adding a heavier critic or entropy
pipeline.
