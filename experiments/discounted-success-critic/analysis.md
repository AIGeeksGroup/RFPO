# Analysis: Cross-Iteration Discounted-Success Critic Audit

## Outcome

H19 is invalidated before its formal audit because the locked training protocol cannot train either
critic during iteration 1. The remote scheduler check showed:

```text
initial 0.0 0
after_optimizer 0.0 0
after_scheduler 0.0001 1
```

`get_scheduler(..., num_warmup_steps=1)` sets the optimizer learning rate to zero at construction,
but `finetune_online_rl.py` advances the scheduler only once after the entire rollout/update
iteration. All ten critic-only epoch optimizer steps therefore run at learning rate zero. This also
affects the released control critic, not only the H19 side critic.

The two-environment pipeline smoke completed and produced finite metrics, but the candidate behaved
like an untrained sigmoid output: value MSE was 0.24797 versus 0.12908 for control. These numbers are
not a confirmatory H19 result because the protocol states that the candidate is trained during
iteration 1, which did not occur.

## Decision

Do not run the locked formal H19 seed and do not interpret the smoke as evidence against bounded
success critics. Close H19 as invalidated and isolate the newly discovered zero-learning-rate critic
warmup bug in H20 before reconsidering critic objectives.
