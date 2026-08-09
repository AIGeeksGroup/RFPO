# Protocol: Replay-Exact Vine Return Audit

## Hypothesis

At fixed intermediate states of the released Square policy, paired standard-Gaussian action chunks have reproducible terminal-success rankings across independent continuation-noise blocks. If supported, these rankings can provide critic-free local advantages for a later branch-return FPO update.

## Locked Setup

- Hypothesis ID: `H62`.
- Policy: Square checkpoint `trc7rbt0_step_110000`, EMA weights.
- Sampler: official Euler integration, 10 velocity steps, 16 executed actions per chunk.
- Root environment seeds: `20261070` through `20261077`.
- Root rollout source: stateless unit Gaussian keyed by environment seed and replan index.
- Vine points: after environment steps `80`, `160`, and `240`, provided the root episode is still active.
- Candidates: two independently keyed unit-Gaussian source chunks per vine point, labeled `0` and `1`.
- Continuations: four per candidate. Block A contains continuation replicas `0,1`; block B contains replicas `2,3`. Within a replica, candidate 0 and 1 receive the same future source tensor at every matching continuation replan index.
- Maximum complete suffix rollouts: `8 roots * 3 states * 2 candidates * 4 continuations = 192`.
- Renderer: rootless OSMesa for this mechanism audit. No official EGL or benchmark claim follows from renderer-dependent reward values.

## Replay Construction

Collect and store every low-level root action. For each vine rollout, recreate Square with the original environment seed and replay the exact action prefix to the locked step. The candidate chunk is then executed for up to 16 steps, followed by the frozen policy until success or the 400-step horizon.

State restoration through `sim.set_state_from_flattened` alone is forbidden because it omits controller, observation-history, and RNG state. Prefix replay is valid only if every replay lane matches its root reference at the branch point in all of the following:

1. complete processed-observation SHA-256;
2. 26-dimensional privileged-state SHA-256;
3. terminated and truncated flags remain false through the prefix;
4. branch step index is exact.

## Locked Metrics and Gates

All numerical values must be finite, all outcomes binary, and all policy parameters must remain bitwise unchanged. The audit passes only if every validity gate and every signal gate passes:

1. every scheduled root that is active at a vine point is replayed exactly in all eight candidate/continuation lanes;
2. aggregate candidate source absolute mean is at most `0.05` and standard deviation is in `[0.95, 1.05]`;
3. median normalized RMS difference between paired candidate action chunks is at least `0.05`;
4. each continuation block has at least eight vine states with a nonzero candidate success difference;
5. at least six vine states are nonzero in both blocks;
6. candidate-difference Pearson correlation across all vine states is at least `0.35`;
7. sign agreement across common-nonzero vine states is at least `0.75`;
8. the top quartile selected by block A has strictly positive mean candidate difference in block B, and vice versa.

Candidate difference is `mean_success(candidate_1) - mean_success(candidate_0)` within a continuation block. Stable sorting by `(root_seed, branch_step)` resolves top-quartile ties.

## Decision Rule

Any failed gate refutes H62. Do not add actor updates, tune branch steps, add continuation samples, change the checkpoint, select states by hindsight task stage, or rerun another seed block. If every gate passes, separately register a two-update Square branch-return FPO reward screen before implementation. The reward screen must compare both zero- and Gaussian-source official metrics; this audit alone cannot establish a benchmark improvement.
