# Protocol: H41 Square Zero-Endpoint PCGrad Screen

## Hypothesis

Applying the previously mechanism-supported zero-endpoint conflict projection during a short
Square FPO++ continuation improves Gaussian-source success while preserving deterministic behavior.
Square is a stronger test than Can step 6000 because H40 established non-saturated success in both
source modes.

## Locked Pipeline Smoke

Before the paired run, execute one candidate-only smoke with two environments, 64 collection steps,
and two iterations. It must complete one critic-only and one actor-update iteration, save a finite
checkpoint, and log finite endpoint-PCGrad metrics. Smoke reward is diagnostic and cannot alter the
formal seed, budget, gates, or method.

## Locked Training Pair

- Initialization: exact released Square `trc7rbt0_step_110000` EMA actor.
- Conditions: official FPO++ control and zero-endpoint PCGrad candidate.
- Seed: 20260913 shared by both conditions, including deterministic per-environment seeds.
- Per condition: 16 Gaussian-source environments, 320 collection steps per iteration, and 25,600
  total timesteps: exactly five iterations. Iteration 1 is critic-only warmup, followed by four
  actor-update iterations with ten epochs, eight minibatches, and MC8.
- Official Square FPO++ settings: actor LR `1e-5`, GAE `0.99`, discount `0.995`, Huber delta `1`,
  clip coefficient `0.01`, max gradient norm `25`, no log-ratio or old-CFM-loss clamps, PPO trust
  region, 10 Euler flow steps, and 16 predicted action steps of which eight are executed.
- Candidate change only: freeze the exact initial actor; cache its normalized zero-source Euler-10
  endpoint at each rollout chunk-start observation; before each actor optimizer step, project the RL
  gradient against the current endpoint-MSE gradient only when their dot product is negative.
- No anchor coefficient, partial projection, layer selection, schedule, checkpoint selection, or
  additional regularizer is allowed.

## Locked Evaluation

- Evaluate each final non-EMA actor checkpoint with Euler-10 zero and independent standard-Gaussian
  sources.
- Seed: 20260914 shared across conditions and modes.
- Accounting: 20 environments, exactly one completed episode per environment, for 20 episodes per
  condition per mode.
- OSMesa is accepted for this paired screen only. Any final benchmark claim requires an independent
  seed and healthy EGL.

## Gates

Require finite training/evaluation and all of:

1. candidate random-source success exceeds control by at least 2/20;
2. candidate zero-source success is no worse than control by more than 1/20;
3. candidate pooled success over 40 episodes exceeds control by at least 3 successes;
4. projection is active in at least 5% and at most 90% of candidate actor optimizer steps.

If any gate fails, stop this transfer without changing seed, budget, projection, endpoint, or nearby
hyperparameters. If all pass, register a separate independent 50-episode-per-mode confirmation before
running it; do not infer an official benchmark improvement from this screen alone.
