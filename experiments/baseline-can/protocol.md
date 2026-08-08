# Protocol: Official Can Baseline

## Claim

The released Can base checkpoint and FPO++ fine-tuning pipeline reproduce the reported zero-sampling evaluation behavior on H200.

## Stages

1. Download and evaluate the released base checkpoint for 200 episodes.
2. Run a reduced-timestep FPO++ fine-tuning smoke test.
3. Run the official 5M-step seed only after the checkpoint metric and smoke test are valid.

## Primary Metric

Zero-sampling success rate over 200 evaluation episodes. Random-sampling success is retained as a diagnostic for exploration quality.

## Acceptance Criterion

The released base checkpoint must match the documented success regime within binomial uncertainty. Fine-tuning must show the released upward trend before additional seeds are scheduled.

