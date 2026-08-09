# H74: Official Spot Cross-Task Screen

## Question

Does equal-total-NFE symmetric endpoint projection transfer from independently
trained Go2 policies to the official Spot locomotion task?

## Fixed training and evaluation

- Task: `Isaac-Velocity-Flat-Spot-v0`
- Training seed 42, 4,096 environments, 1,500 official iterations
- Preserve the official Spot FPO++ config, including 32 CFM samples, 32 learning
  epochs, and value-loss coefficient 0.5.
- Disable only the post-training all-checkpoint evaluation sweep.
- Select the fixed final `model_1499.pt`; no reward-based checkpoint selection.
- Evaluate 256 environments per method with evaluation seed `20261640`, primary
  source seed `20261641`, secondary source seed `20261642`, and one episode per
  environment.
- Methods:
  - `zero64`: zero source, Euler-64, 64 NFE/action
  - `zero32`: zero source, Euler-32, 32 NFE/action
  - `random64`: one Gaussian source, Euler-64, 64 NFE/action
  - `iid_pair32`: two IID Euler-32 endpoints, 64 total NFE/action
  - `antithetic32`: one `z/-z` Euler-32 pair, 64 total NFE/action
- Bootstrap seed `20261643`, 20,000 paired resamples.

All methods must share initial-observation hashes. The three stochastic methods
must share the complete primary source stream. `iid_pair32` and `antithetic32`
must use the same secondary-source seed, although only IID consumes it.

## Decisions

The Spot transfer hypothesis passes if:

1. the fixed policy is healthy (`zero64` mean return at least 250), all actions
   and returns are finite, and all pairing hashes match;
2. `antithetic32 - random64` has a positive point estimate and strictly positive
   paired-bootstrap 95% lower bound; and
3. `antithetic32 - iid_pair32` has a positive point estimate and strictly
   positive paired-bootstrap 95% lower bound.

`Antithetic32 - zero32` is a separate deployment gate. A failed primary gate
stops Spot without another seed, checkpoint, step count, or source scale and
blocks H1/G1 expansion. A pass authorizes one official H1 screen before any
multi-seed cross-task confirmation.
