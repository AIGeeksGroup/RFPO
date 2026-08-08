# Resolved Runtime Blocker: GPU 1 Offscreen Rendering

The original H16 smoke runs did not reach the iteration-2 audit because Can environment rendering
on physical GPU 1 became pathologically slow on 2026-08-09. An isolated OSMesa fallback restored
normal rollout throughput without changing the FPO conda environment.

## Evidence

- The first 8-environment smoke reached data collection but produced no completion log after more
  than four minutes; matched historical smokes complete the same 320-step collection in about ten
  seconds.
- A fresh 8-environment retry reproduced the stall. Disabling world-size-1 DDP also reproduced it.
- Policy-only inference completed in 0.362 seconds, and policy inference with a live environment
  completed in 0.310 seconds before an environment step.
- A one-environment, zero-action test isolated the failure to rendering: step 0 took 0.279 seconds
  and step 1 took 19.100 seconds without policy inference.
- A generic 4096 by 4096 CUDA matrix multiplication completed in 0.088 seconds with a live EGL
  environment. GPU clocks were at their configured maximum, with no thermal slowdown, ECC error,
  PCIe replay, recovery request, or competing process.
- Killing the render process returned GPU 1 to 0% utilization and 5 MiB, but a new render process
  reproduced the same second-step slowdown.

## Resolution

- Extracted Ubuntu's checksummed `libosmesa6` and matching `libglapi-mesa` packages into
  `~/workspace/scratch/fpo-osmesa/osmesa-root` without root access or changes to the training env.
- A one-environment diagnostic completed four steps in 0.120, 0.070, 0.069, and 0.069 seconds.
- The 8-environment H16 smoke sustained about 70-90 environment steps per second.
- The locked 30-environment audit sustained about 130-190 environment steps per second and
  completed both updates.

Matched-state rendering was close but not bit-identical. Reset low-dimensional state was exact;
mean absolute image difference was 0.00812. The deterministic step-6000 policy's first action had
mean/max absolute differences of 0.00322/0.00740 and an L2 difference of 0.01025, about 1% of the
reference action norm. OSMesa is accepted for mechanism diagnostics, not as final evidence that a
reward benchmark improved.

## Artifacts

Failed remote logs:

```text
~/workspace/outputs/fpo-control/logs/can_step6000_cfmratio_audit_smoke_seed20260815.log
~/workspace/outputs/fpo-control/logs/can_step6000_cfmratio_audit_smoke_retry_seed20260815.log
```

Reproduction scripts:

```text
research_scripts/diagnostics/check_egl_cuda_interop.py
research_scripts/diagnostics/check_egl_policy_interop.py
```

## Remaining Condition

Before any final reward benchmark, reset physical GPU 1 or use another authorized GPU for EGL
rendering while keeping policy compute on physical GPU 1. Rerun the one-environment diagnostic and
require every measured step to complete in under two seconds.
