# Protocol: Official Go2 Baseline

## Claim

The official FPO++ Go2 configuration can reproduce the reported learning trend and approximately 40 final return on the H200 server.

## Stages

1. Environment/import check with no training.
2. Short run with 64 or 256 environments and 5-10 iterations to validate rollout, update, logging, and checkpointing.
3. One official-scale run with 4096 environments and 1500 iterations.
4. If the first full run is credible, run two additional seeds for mean and variability.

## Primary Metric

Zero-sampling evaluation return at the final checkpoint. Training return is secondary because it includes stochastic exploration.

## Acceptance Criterion

The curve must reach the same high-return regime as the released reference and the final evaluation must fall within 10% of the reference mean or overlap its seed variation. A failed smoke run is an infrastructure result, not evidence against FPO++.

