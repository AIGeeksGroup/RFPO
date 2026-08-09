# H65 Fixed-Tail Go2 Actor Averaging Analysis

## Decision

H65 is refuted under its locked 256-environment-per-mode reward screen. The fixed-tail actor average
was valid and active, but its return was statistically indistinguishable from the final checkpoint
and its equally weighted two-mode gain was slightly negative. Do not run the 4,096-environment
confirmation or change the averaging window, weights, source run, seed, or payload policy.

## Construction Validity

The candidate uniformly averaged the eight `actor.*` tensors from checkpoints 1300, 1350, 1400,
1450, and 1499 in float64. Its relative actor L2 displacement from the final checkpoint was
`0.035143831`, inside the locked `[0.01, 0.06]` interval. Reload validation confirmed exact retention
of every non-actor model tensor and every non-model payload. The candidate checkpoint SHA-256 is
`bc9cbd581f831736598612282b42f5f563ddccf9a865cc0950f55a83b14ad593`.

## Paired Reward Screen

All four method-mode evaluations completed exactly 256 episodes with finite actions and returns.
Control and candidate initial-observation SHA-256 values matched exactly in both modes.

| Mode | Final control | Tail average | Paired gain | Paired bootstrap 95% |
|---|---:|---:|---:|---:|
| Zero | 41.64152 | 41.63355 | -0.00797 | [-0.02489, 0.01178] |
| Random | 40.56500 | 40.56672 | +0.00172 | [-0.09433, 0.18210] |

The equally weighted mode gain was `-0.003125`, below the required `+0.10`. The pooled paired
bootstrap interval was `[-0.05459, 0.09139]`, whose lower bound failed the strictly-positive gate.
The per-mode non-degradation gates passed, but both improvement gates failed.

## Interpretation

Uniform averaging moved the actor materially in parameter space without changing its behavior in a
detectable favorable direction. The late Go2 checkpoints appear to occupy a functionally flat local
region rather than complementary reward optima. This closes coefficient-free fixed-tail weight
averaging on the reproduced run; reward-selected soups and window tuning would reuse the same screen
for selection and are not justified by this result.

Raw evaluator JSON, construction manifest, analysis JSON, and complete logs are archived under
`results/`.
