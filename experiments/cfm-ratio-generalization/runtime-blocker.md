# Runtime Blocker: GPU 1 Offscreen Rendering

H16 remains active. Its smoke runs did not reach the iteration-2 audit because Can environment
rendering on physical GPU 1 became pathologically slow on 2026-08-09.

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
  reproduced the same second-step slowdown. OSMesa is unavailable in the installed environment.

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

## Resume Condition

Either an administrator must reset physical GPU 1, or the user must authorize using another idle
GPU for MuJoCo rendering while policy compute remains on physical GPU 1. After either action, rerun
the one-environment zero-action diagnostic and require every measured step to complete in under two
seconds before restarting the locked H16 smoke.
