# H23 Lower Actor Learning Rate Audit

## Result

H23 is refuted. Reducing the actor learning rate from `1e-5` to `2.5e-6` did not preserve the
pre-update held-out gradient direction through ten FPO++ optimizer epochs. Both conditions used the
step-6000 Can checkpoint, seed 20260822, 16 environments, 320 collection steps per iteration, two
iterations, ten update epochs, and 64 held-out positive chunks split into two batches.

| Epoch-10 metric | Control `1e-5` | Candidate `2.5e-6` | Candidate gate | Pass |
|---|---:|---:|---:|:---:|
| Pooled active positive ratio | 0.53711 | 0.54492 | >= 0.80 | No |
| Batch-0 gradient cosine | 0.29089 | 0.26763 | >= 0.60 and >= control + 0.20 | No |
| Batch-1 gradient cosine | 0.27988 | 0.30924 | >= 0.60 and >= control + 0.20 | No |
| Batch-0 surrogate gain | 0.01569 | 0.00471 | positive and >= 50% of control | No |
| Batch-1 surrogate gain | 0.02288 | 0.00687 | positive and >= 50% of control | No |

The control and candidate had 166 and 177 eligible positive chunks respectively; both audited the
locked 64 chunks. The candidate retained only about 30% of the control surrogate gain in each batch,
while its cosine changed by -0.0233 and +0.0294. A smaller global Adam learning rate therefore
reduced useful movement without addressing the gradient rotation. No five-update reward screen was
run, as required by the protocol.

## Artifacts

- `control_lr1e-5_seed20260822.json`
- `candidate_lr2.5e-6_seed20260822.json`
- Remote control log: `can_step6000_lrdrift_control16_osmesa_seed20260822.log`
- Remote candidate log: `can_step6000_lrdrift_candidate025_16_osmesa_seed20260822.log`

Both remote jobs exited normally. Physical GPU 1 returned to 0% utilization and 5 MiB after the
candidate run.
