# Protocol: Cross-Iteration Discounted-Success Critic Audit

## Hypothesis

For binary-terminal Can rewards, a bounded critic trained with a proper Bernoulli loss on observed
discounted success returns predicts the next rollout's return signal more accurately than the
official bootstrapped MSE critic, producing a first-epoch actor gradient closer to a Monte Carlo
outcome reference.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260818; official scale-1 Gaussian source
- Actor: frozen during iteration 1 exactly as in the official critic-only warmup
- Control critic: unchanged official architecture, initialization, GAE-return MSE objective,
  optimizer, minibatches, and ten update epochs
- Candidate critic: identical initial parameters and optimizer schedule; scalar output interpreted as
  a logit and trained with binary cross entropy on soft discounted Monte Carlo returns
- Candidate labels: only rollout steps whose next terminal is observed before collection ends;
  successful labels are `gamma ** steps_to_success`, failed labels are zero, and trailing censored
  segments are excluded
- Evaluation: iteration-2 observations before either critic receives an iteration-2 update
- Gradient audit: two seeded, disjoint batches of 32 fully valid action chunks with uncensored Monte
  Carlo labels, using the rollout-stored MC8 CFM variables
- Control weights: official iteration-2 GAE advantages
- Candidate weights: GAE advantages recomputed from candidate sigmoid values with the same gamma and
  lambda
- Reference weights: iteration-2 discounted Monte Carlo returns centered within the complete audit
  set; no outcome selection
- Parameters: all trainable non-vision actor parameters
- No candidate value or gradient enters the official actor update

## Gates

Require finite labels, values, and nonzero reference gradients. All gates must pass:

1. candidate iteration-2 value MSE to uncensored discounted returns is at least 10% lower than the
   control value MSE;
2. candidate value Spearman rank correlation is no lower than control;
3. in each 32-chunk batch, candidate actor-gradient cosine to the Monte Carlo reference is at least
   0.05 higher than control and is positive.

Stop without online integration if any gate fails. If all gates pass, integrate only the candidate
critic objective and run one matched five-update step-6000 screen. Do not tune the loss, label
discount, censoring rule, architecture, seed, or gates after observing the audit.
