# Protocol: H30 Zero-Endpoint PCGrad Can Reward Screen

## Hypothesis H30

Applying the H29 zero-endpoint conflict projection during FPO++ actor updates improves final
full-Gaussian Can success without materially degrading deterministic zero-source success.

## Locked Training Pair

- Initialization: exact Can `95j3noe4_step_6000` EMA actor.
- Conditions: released FPO++ control versus zero-endpoint PCGrad candidate.
- Seed: 20260828 for both conditions.
- Per condition: 16 full-Gaussian environments, 320 collection steps per iteration, five iterations
  total. Iteration 1 is the released critic-only warmup, followed by exactly four actor-update
  iterations with ten epochs, eight minibatches, MC8, GAE 0.99, and all released clipping and
  optimizer settings.
- Candidate anchor: frozen exact initial actor. For every rollout chunk-start observation, cache its
  normalized zero-source endpoint using 10 Euler steps. Before each actor optimizer step, compute the
  current endpoint-MSE gradient and apply the H29 global projection only on negative gradient dot
  product. Critic gradients and updates are unchanged.
- Log projection-active fraction, RL/anchor cosine, candidate norm retention, and endpoint MSE.
- No scalar anchor coefficient, partial projection, layer selection, or schedule is allowed.

## Locked Evaluation

- Evaluate the final non-EMA actor checkpoint from each condition.
- Modes: official 10-step zero source and scale-1 Gaussian source.
- Seed: 20260829 shared across conditions and modes.
- Budget: 20 episodes per condition per mode, 50 parallel environments as in official evaluation.
- OSMesa is accepted for this paired screen; it does not authorize a final benchmark claim.

## Gates

Require finite training and evaluation plus all of:

1. candidate random-source success exceeds control by at least 2/20;
2. candidate zero-source success is no worse than control by more than 1/20;
3. candidate pooled success over the 40 zero/random episodes exceeds control by at least 2 episodes;
4. projection is active in at least 10% and at most 90% of candidate actor optimizer steps.

Stop without confirmation if any gate fails. If all pass, run one independent 50-episode-per-mode
confirmation at seed 20260830 before any broader benchmark or scale increase. Do not tune the anchor,
projection, learning rate, epochs, gates, or evaluation seed after observing results.

