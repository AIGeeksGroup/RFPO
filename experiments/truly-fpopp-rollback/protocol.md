# Protocol: FPO++ Ratio Rollback

## Hypothesis

Replacing FPO++'s flat PPO clipping branch with the PPO-RB rollback branch prevents improving CFM
ratios from continuing outward under other samples' gradients, preserving the useful pre-update
held-out direction across ten actor epochs without discarding most official surrogate progress.

## Locked Method

- Keep the official CFM ratio, chunk aggregation, normalized GAE, optimizer, actor learning rate,
  minibatches, ten epochs, and the established step-6000 audit `clip_coef=0.02` unchanged.
- For each ratio `r`, use the Truly PPO paper's PPO-RB function
  `F_RB(r) = -alpha*r + (1+alpha)*(1 +/- epsilon)` outside the clip interval and `r` inside it.
- Minimize `max(-A*r, -A*F_RB(r))`, exactly matching the paper's `min(A*r, A*F_RB(r))` objective.
- Fix `alpha=0.3`, the paper default for continuous-control PPO-RB tasks other than Humanoid.
- Do not test a KL approximation, coefficient schedule, alpha sweep, asymmetric rollback, or altered
  clip coefficient.

At ratio one, candidate and official losses and gradients must agree exactly up to numerical
tolerance. Rollback is disabled by default and selected explicitly with `trust_region_mode=rollback`.

## Locked Audit

- Initialization: Can `95j3noe4_step_6000`, EMA weights, Huber CFM loss
- Conditions: official FPO++ control and rollback candidate
- Shared rollout seed: `20260907`; official scale-1 Gaussian source
- Budget: two iterations per condition: one critic-only warmup and one ten-epoch actor update
- Environments: 16; all other released step-6000 settings unchanged
- Audit set: 64 positive-advantage fully valid chunks, selected once and split into two seeded,
  disjoint 32-chunk batches
- Held-out estimator: one independent fixed MC8 set per chunk, never used by either optimizer
- Measurements after every epoch: official held-out clipped-gradient cosine to its pre-update value,
  positive active-ratio fraction, unclipped advantage-weighted surrogate gain, and training rollback
  branch activity
- The control and candidate must reproduce the same pre-update held-out ratios, gradients, and
  surrogates within `1e-6` absolute tolerance; otherwise the pair is invalid and rerun only after an
  infrastructure correction.

## Gates

At epoch 10, both disjoint held-out batches must satisfy all of the following:

1. candidate official-objective gradient cosine is at least `0.10` above control and at least `0.25`;
2. candidate positive active-ratio fraction is at least `0.05` above control;
3. candidate unclipped surrogate gain is positive and at least 50% of the positive control gain;
4. candidate finite ratios and gradients are nonzero, and rollback is active on at least 5% but no
   more than 80% of candidate actor loss elements across the ten-epoch update.

Stop without online reward training if either batch fails any gate. Do not alter the coefficient,
seed, audit set, epochs, or gates after observing results. If all gates pass, integrate the same
default-off loss into a matched five-update control/candidate screen, then evaluate one balanced
episode from each of 20 environments under shared zero and random-source seeds. A larger
confirmation remains forbidden until that reward screen passes its separately committed gates.
