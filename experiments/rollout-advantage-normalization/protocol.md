# Protocol: Rollout-Level Advantage Normalization

## Hypothesis

FPO++ currently recomputes advantage mean and standard deviation independently inside every
minibatch and training epoch. With terminal binary rewards, this changes the relative scale of the
same successful trajectory as minibatches are reshuffled. Computing one fixed set of advantage
moments over the complete rollout should reduce update-weight drift and improve early on-policy Can
success.

## Method

Compare the official `minibatch` normalization against opt-in `rollout` normalization. In the
candidate, compute mean and population variance once from all valid rollout advantages and reuse
them for every minibatch and epoch. Aggregate exact sum, squared sum, and count across DDP ranks.
Keep signed advantages, the FPO++ clipped per-sample ratio, source distribution, reward, mask
behavior, and every other hyperparameter unchanged.

## Locked Screen

- Initialization: `95j3noe4_step_6000`, EMA weights
- Conditions: official `minibatch` control and `rollout` candidate
- Training seed: 20260812 for both conditions
- Budget: five 48k-environment-step updates (240k steps per condition)
- Can environments: 30; collection steps: 1600; CFM samples: 8
- Final evaluation: 50 zero-source and 50 Gaussian-random episodes at iteration 5
- Run control first, candidate second; no restart or seed retry

## Gates

Primary: pool Gaussian-source collection episodes from actor-update iterations 2-5. Candidate must
exceed the matched control success rate by at least 3 percentage points.

Secondary safety gates: all losses and gradient norms must remain finite; every candidate update
must log finite rollout advantage moments with a positive sample count; final zero-source success
must be at least 90%; and final random-source success may not be more than 5 percentage points
below control.

Stop if any gate fails. If all gates pass, run one matched new-seed short screen before transferring
the method to the official sparse `step_1000` initialization. Do not tune the normalization epsilon,
moment estimator, learning rate, clipping, source distribution, or mask behavior in this screen.
