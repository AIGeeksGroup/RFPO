# Protocol: H48 Residual Flow Steering on Square

## Hypothesis

Freezing the released Square flow actor and learning a joint policy over its initial latent source and
a bounded output residual improves pooled deterministic/stochastic success. The separate residual
branch can correct local control errors while the latent branch changes global modes without
overwriting the pretrained flow field.

## Locked Implementation

- Base actor: released `trc7rbt0_step_110000` checkpoint with EMA weights, frozen exactly.
- Modulation policy: image/state conditioning from the frozen actor; ReLU MLP widths 256, 128, 64;
  zero-initialized latent/residual mean head.
- Latent source: diagonal Gaussian, initial mean zero and standard deviation one.
- Residual: diagonal Gaussian in raw space followed by `0.1 * tanh`, then added to the decoded
  environment-space action and clamped to the environment bounds.
- PPO: exact joint latent/residual log probability, clip 0.2, policy LR `3e-4`, value LR `1e-3`,
  GAE lambda 0.95, max gradient norm 1.0, four minibatches, ten epochs.
- Rollout: Square, seed `20260924`, 16 environments, 320 environment steps per iteration, five
  iterations, 25,600 total environment interactions, Euler-10, 16 executed actions per plan.
- No flow-actor update, CFM loss, FPO ratio, checkpoint selection, parameter sweep, replay, or
  neighboring residual/latent scale.

The implementation is faithful to RFS's joint input/output modulation but adapts it to FPO++'s
image-conditioned action chunks and robosuite Square. It is not an exact reproduction of RFS's
IsaacLab dexterous-hand experiments.

## Mechanism Gate

Before reward evaluation, all of the following must hold:

1. every frozen base-policy state tensor remains bitwise unchanged after each PPO iteration;
2. both latent and residual mean-head parameter blocks have nonzero update norms;
3. latent/residual samples, losses, gradient norms, and checkpoints remain finite;
4. the initial latent distribution matches the base Gaussian source and the deterministic initial
   residual is exactly zero (covered by unit tests).

Failure stops H48 without scale, width, learning-rate, epoch, or seed tuning.

## Balanced Reward Screen

- Evaluation seed: `20260925`, unused in prior Square screens.
- Control: released base actor with zero and Gaussian source.
- Candidate: final RFS policy using mean and sampled modulation respectively.
- Each cell: 20 environments, exactly one completed episode per environment, Euler-10, 16 action
  steps, OSMesa paired rendering.
- Primary metric: pooled success out of 40.

H48 passes only if candidate gains at least 3/40 pooled successes, gains at least 2/20 in sampled
mode, and loses no more than 1/20 in mean mode. A pass authorizes an independent confirmation;
it is not yet an official renderer benchmark claim. Failure stops H48 without another seed,
checkpoint selection, adjacent scale, longer training, or confirmation.
