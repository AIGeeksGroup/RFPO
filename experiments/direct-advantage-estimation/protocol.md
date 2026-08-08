# Protocol: Continuous-Action Direct Advantage Audit

## Hypothesis

The first FPO++ actor update is unreliable because GAE obtains action effects indirectly through a
state-value critic. A chunk-level action-conditioned advantage head, trained with the Direct
Advantage Estimation return decomposition and Monte Carlo centered under the frozen flow policy,
will preserve outcome ranking and align the actor gradient more consistently with observed returns.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260819; official scale-1 Gaussian source
- Actor: frozen during iteration 1 and never updated with candidate weights
- Control: unchanged official critic and GAE advantages
- Candidate input: frozen 1033-dimensional observation encoding and the normalized executed action
  chunk
- Candidate head: two-layer ReLU MLP with widths 512 and 256 and one scalar output
- Continuous-action centering: for every chunk-start observation, subtract the mean candidate output
  over four fixed action chunks sampled from the same frozen flow policy
- Temporal unit: one FPO action chunk; macro reward is the within-chunk discounted reward and macro
  discount is `gamma ** n_action_steps`
- Candidate loss: four-macro-step DAE residual with the post-warmup control critic held fixed as both
  start-state baseline and nonterminal bootstrap target; terminal windows bootstrap to zero
- Candidate optimization: ten epochs, Adam at the official critic learning rate, official gradient
  clipping, no actor or vision-encoder gradients
- Evaluation: fresh iteration-2 rollout before any actor update, with four newly sampled centering
  actions per observation
- Gradient audit: the same two seeded, disjoint batches of 32 fully valid chunks and rollout-stored
  MC8 CFM variables used by H19
- Reference weights: uncensored discounted Monte Carlo returns centered over the complete audit set
- Actor-gradient parameters: all trainable non-vision actor parameters

## Gates

Require finite losses, at least 64 fully valid uncensored chunks, nonzero candidate-weight variance,
and nonzero reference gradients. All gates must pass:

1. candidate Spearman correlation with iteration-2 Monte Carlo returns is at least 0.05 higher than
   control GAE correlation;
2. in each 32-chunk batch, candidate actor-gradient cosine to the Monte Carlo reference is positive
   and at least 0.05 higher than control;
3. the candidate's iteration-1 DAE training loss decreases by at least 20% from its first-epoch mean
   to its final-epoch mean.

Stop without online integration if any gate fails. If all gates pass, integrate only the candidate
advantage weights and run one matched five-update step-6000 screen. Do not tune the centering sample
count, DAE horizon, head architecture, learning rate, epochs, seed, or gates after observing the
audit.
