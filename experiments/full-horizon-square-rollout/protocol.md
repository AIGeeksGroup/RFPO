# Protocol: H47 Full-Horizon Square Rollouts

## Hypothesis

Collecting a complete 400-step Square horizon per FPO++ rollout improves reward learning because
initial failed episodes receive observed terminal labels instead of being bootstrapped from the
critic at the released 320-step collection boundary. The candidate may improve the matched
zero/random evaluation despite using one fewer actor-update rollout at the same total environment
interaction budget.

## Locked Training Comparison

- Initialization: released `trc7rbt0_step_110000` Square checkpoint with EMA weights.
- Control: frozen H43 official-control run `square_h43_control_osmesa_seed20260916`, five 320-step
  rollouts, of which iteration 1 is critic-only and iterations 2-5 update the actor.
- Candidate: seed `20260916`, four 400-step rollouts, of which iteration 1 is critic-only and
  iterations 2-4 update the actor.
- Equal interaction budget: 25,600 environment steps in both conditions (`16 environments x 1,600
  steps per environment`).
- Unchanged settings: standard Gaussian rollout source, 16 executed actions, Euler-10, MC8,
  eight minibatches, ten update epochs, official actor/critic learning rates, GAE, clipping, Huber
  CFM loss, optimizer state, reset behavior, and OSMesa.
- No early stopping, checkpoint selection, extra actor epochs, loss changes, or success replay.

## Training Validity

The candidate is valid only if:

1. the first 300 collection steps reproduce H43 control's initial prefix diagnostics (two completed
   successes and finite rollout state);
2. the first 400-step rollout completes at least 16 episodes, demonstrating that every initial
   environment trajectory received an observed terminal before any reset suffix is censored;
3. exactly four iterations finish and the latest checkpoint is finite.

## Balanced Reward Screen

- Evaluation seed: `20260923`, unused by prior Square evaluation.
- Order: control zero, control random, candidate zero, candidate random.
- Each cell: 20 environments, one completed episode per environment, 16 executed actions,
  Euler-10, no EMA, OSMesa, deterministic `seed + env_id` environment seeds.
- Primary metric: pooled success out of 40; secondary metrics are zero- and random-source success.

H47 passes only if the candidate gains at least 3/40 pooled successes, gains at least 2/20 random
successes, and loses no more than 1/20 zero successes. A pass authorizes a separately locked
independent confirmation, not an official benchmark claim.

If any validity or reward gate fails, stop without a 360/384-step neighbor, longer horizon, another
seed, more actor epochs, larger interaction budget, checkpoint selection, or confirmation. OSMesa
remains a paired screen rather than the official renderer benchmark.
