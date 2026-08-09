# Confirmation Protocol: H51 Potential-Based Square Stage Shaping

## Authorization and Claim

The preregistered H51 screen passed at `+1/20` zero-source, `+3/20` Gaussian-source, and `+4/40`
pooled successes. This protocol tests whether that gain survives an independently seeded training and
evaluation pair. It does not authorize an official EGL benchmark or real-robot claim.

## Locked Training Pair

- Initialize control and candidate independently from the released Square
  `trc7rbt0_step_110000` EMA checkpoint.
- Use training seed `20260977`, distinct from the screen's `20260974`.
- Keep the screen's 16 environments, Gaussian source, Euler-10 sampler, 16 executed actions, five
  iterations, 320 primitive collection steps, and 25,600 environment interactions per condition.
- Keep every optimizer, scheduler, CFM, GAE, clipping, and checkpoint setting unchanged.
- Control uses sparse reward. Candidate uses exactly
  `r' = r + 0.995 * Phi(s_next) - Phi(s)`, where
  `Phi = max(reach, grasp, lift, hover)` and terminal or timeout next potential is zero.
- Require exact control/candidate collection fingerprints until actor updates can affect behavior,
  five finite 5,120-term candidate shaping records, and finite losses and ratios.
- Evaluate only the final `step_25600` checkpoint. No checkpoint selection is allowed.

## Locked Evaluation

- Use evaluation seed `20260978`, distinct from all audit, screen, and training seeds.
- Evaluate sparse Square success only, separately under zero and Gaussian source.
- Run 50 episodes per condition and mode with 25 parallel environments, exactly two episodes from
  each environment, Euler-10, 16 executed actions, and no video.
- Run cells serially in this order: control zero, control Gaussian, candidate zero, candidate Gaussian.

## Confirmation Gate

H51 is confirmed only if all of the following hold on the independent 100-episode pair:

- candidate Gaussian success is at least `+3/50` above control;
- candidate zero success is no worse than `-2/50` below control;
- candidate pooled success is at least `+5/100` above control.

Passing authorizes a healthy-EGL official benchmark after renderer recovery. Failing any validity or
reward gate closes H51 without another seed, coefficient, potential definition, training length,
checkpoint, or threshold.

