# Protocol: BC/FPO++ Weight-Space Interpolation

## Hypothesis

A fixed midpoint interpolation between the behavior-cloned actor and its FPO++-finetuned actor
recovers deterministic task stability without discarding the stochastic-source improvement learned
online. This follows the WiSE-FT principle of interpolating pretrained and finetuned weights to
retain robustness while preserving task adaptation.

## Locked Checkpoints

- Existing run: `can_step6000_advnorm_minibatch_5iter_seed20260812`
- Anchor: `checkpoints/step_48000`. Iteration 1 is critic-only, so this actor is exactly the loaded
  step-6000 EMA actor represented in the finetuning checkpoint format.
- Finetuned endpoint: `checkpoints/step_240000`, after actor updates in iterations 2-5.
- Candidate: tensorwise linear interpolation
  `theta_mid = 0.5 * theta_anchor + 0.5 * theta_finetuned`.
- All floating policy tensors are interpolated in float32 and cast back to their original dtype.
  Non-floating tensors must be identical and are copied unchanged.
- No coefficient sweep, checkpoint selection, or further training is allowed after evaluation.

## Fixed-Seed Screen

- Environment: Can, 10 Euler sampling steps
- Seed: 20260823
- Conditions: anchor, finetuned endpoint, and midpoint
- Modes: zero sampling and scale-1 Gaussian random sampling
- Budget: 20 episodes per condition and mode, 16 environments, OSMesa rendering
- The same seed is used for all six evaluations.

The midpoint passes only if all conditions hold:

1. zero-sampling success is at least 10 percentage points above the finetuned endpoint;
2. random-sampling success is no more than 5 points below the finetuned endpoint;
3. random-sampling success is at least 5 points above the anchor;
4. the mean of zero/random success is at least 2.5 points above the finetuned endpoint.

If any gate fails, stop without changing `alpha`. If all pass, run a fresh 100-episode-per-mode
confirmation for all three policies. OSMesa is acceptable for this screen; a final benchmark claim
still requires healthy EGL or an independently approved rendering GPU.

## Provenance

- Wortsman et al., *Robust Fine-Tuning of Zero-Shot Models*, CVPR 2022 (WiSE-FT,
  arXiv:2109.01903).
- The intervention is post-training and does not alter FPO++ rollout collection or its objective.
