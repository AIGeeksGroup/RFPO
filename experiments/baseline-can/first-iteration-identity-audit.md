# Can First-Iteration Actor Identity Audit

Date: 2026-08-09

## Question

Did `can_fpopp_one_update_seed0_20260808/checkpoints/step_48000` contain a genuine FPO++ actor update,
potentially explaining its exploratory 20/20 zero-source evaluation?

## Audit

The exact released `95j3noe4_step_1000` checkpoint was loaded through the official evaluation loader
with `load_ema=True`. This matters because its ordinary `policy/model.safetensors` is not the actor
used by the official EMA evaluation; the EMA shadow parameters reside in `optimizer.pt`. The loaded
EMA policy state was compared tensor by tensor with the saved first-iteration policy.

The code path and existing experiment protocols independently establish that iteration 1 is
critic-only. The tensor audit supplied the direct checkpoint check:

| Quantity | Result |
|---|---:|
| State tensors compared | 35 |
| Changed tensors | 0 |
| L2 distance | 0.0 |
| Relative L2 distance | 0.0 |
| Maximum absolute difference | 0.0 |
| Bitwise identical | yes |

For completeness, comparing against the non-EMA `model.safetensors` would incorrectly report a
5.35% relative L2 difference. Applying the checkpoint's EMA state removes that apparent difference
exactly.

## Conclusion

The run was a valid end-to-end pipeline smoke but not an actor-update experiment. Its 20/20 result
cannot motivate a larger confirmation because the evaluated actor is exactly the released actor.
Cancel the proposed 50+50 comparison and require at least two iterations for any future short
FPO++ training screen, so that at least one actor update actually occurs.
