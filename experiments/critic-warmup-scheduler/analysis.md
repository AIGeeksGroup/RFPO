# Analysis: Critic Warmup Scheduler Audit

## Outcome

H20 is invalidated because its premise is false. The released manipulation configuration uses the
Diffusers `constant` scheduler, not `cosine`. Direct remote checks showed:

```text
constant initial 0.0001 last_epoch 0
cosine   initial 0.0    last_epoch 0
```

The diagnostic paired run supplied stronger runtime evidence. After iteration 1, control and
candidate parameter displacement norms were both 1.10998 and their mutual parameter distance was
exactly zero. Their iteration-2 value MSE was identical (0.0149618), and gradient cosine differences
were below `4e-7`. Thus both critics already train at `1e-4`; there is no zero-learning-rate warmup
bug to fix.

## Decision

Do not modify the released critic scheduler and do not run the H20 five-update screen. Restore H19,
whose bounded BCE critic genuinely differs from the official MSE critic under the actual constant
schedule.
