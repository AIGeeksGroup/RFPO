# Protocol: Zero-Endpoint Anchor Conflict Projection Audit

## Hypothesis H29

Constraining only the pretrained policy's deterministic zero-source action endpoint, rather than its
velocity field over generic Gaussian flow queries, preserves benchmark-relevant BC behavior while
retaining enough FPO++ surrogate progress.

## Motivation

H28 measured strong RL/BC conflict and reduced velocity-field drift by 93-98%, but retained only
34-46% of the RL surrogate gain. That anchor protects behavior over arbitrary Gaussian flow points,
whereas the high-success Can evaluation follows the zero-source ODE endpoint. H29 narrows the
protected function without introducing a scalar loss coefficient or changing the RL signal.

## Locked Audit

- Reuse H28's initialization, seed 20260827, 16 environments, 320 steps, two iterations, selected 64
  positive fully valid chunks, independent fixed MC8 variables, two 32-chunk batches, virtual step
  rule, and exact branch restoration.
- Replace only the BC objective. Starting from an all-zero normalized source, differentiably execute
  the checkpoint's official 10 Euler flow steps and compute MSE between current and exact pre-update
  normalized action endpoints on each selected chunk-start observation.
- Use the same global conflict projection rule between the FPO++ gradient and this endpoint-MSE
  gradient. No endpoint weighting, per-action weighting, margin, or projection coefficient is allowed.
- Use a fresh output directory. Reusing the rollout seed is intentional: H28 and H29 differ only in
  the anchor, while audit-variable and endpoint computations have separately fixed seeds.

## Gates

All H28 gates remain unchanged and must pass in both batches: finite nonzero gradients, active
conflict, candidate endpoint-MSE increase at most 50% of control, positive candidate surrogate gain
at least 70% of control, and candidate post-step RL-gradient cosine no worse than control.

Stop without optimizer integration or reward training if any batch fails. Do not tune the endpoint,
integration steps, gates, seed, or projection after observing results.

## Conditional Next Step

Only if all gates pass, freeze a matched short step-6000 Can reward-screen protocol before integrating
the endpoint projection into the optimizer.

