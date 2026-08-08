# Protocol: Critic Warmup Scheduler Audit

## Hypothesis

The released FPO++ manipulation loop nullifies its critic-only warmup by using zero learning rate for
all iteration-1 optimizer steps. Training the identically initialized official MSE critic at its
configured `1e-4` learning rate during iteration 1 improves next-rollout value accuracy and aligns
the first actor gradient more closely with realized Monte Carlo outcomes.

## Locked Audit

- Initialization: `95j3noe4_step_6000`, EMA weights and Huber CFM loss
- Can rollout seed: 20260819; official scale-1 Gaussian source
- Actor: frozen during iteration 1 exactly as in the released loop
- Control critic: unchanged released optimizer and scheduler, whose iteration-1 learning rate is zero
- Candidate critic: identical initial parameters, architecture, GAE-return MSE loss, minibatch order,
  ten epochs, AdamW settings, and clipping; learning rate fixed at the configured `1e-4` during
  iteration 1
- Evaluation: iteration-2 observations before either critic receives an iteration-2 update
- Value reference: uncensored discounted Monte Carlo returns ending at an observed terminal
- Gradient audit: two seeded, disjoint batches of 32 fully valid labeled chunks using stored MC8 CFM
  variables
- Control/candidate weights: GAE advantages from their respective value predictions with identical
  gamma and lambda
- Reference weights: discounted Monte Carlo returns centered over the complete 64-chunk audit set
- No candidate value or gradient enters the released actor update

## Gates

All gates must pass:

1. candidate iteration-2 value MSE is at least 10% lower than control;
2. candidate value Spearman correlation is no lower than control;
3. in each batch, candidate actor-gradient cosine to the Monte Carlo reference is positive and at
   least 0.05 higher than control.

If any gate fails, stop without changing online training. If all pass, set critic scheduler warmup to
zero, run one matched five-update step-6000 control/fix screen, and require at least a two-point pooled
collection-success gain without lower final random success before broader evaluation. Do not tune the
learning rate, epochs, seed, labels, or gates.
