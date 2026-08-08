# Protocol: H3b Matched WarmPrior BC Continuation on Can

## Question

Does training against the WP-Past source, rather than grafting it onto a Gaussian-trained model only
at inference time, improve exploratory Can success after a small and matched adaptation budget?

## Training Comparison

- Starting point: released `95j3noe4_step_1000` checkpoint, including its optimizer and EMA state.
- Conditions: Gaussian source versus previous-action source with residual `sigma=0.5`.
- Budget: 100 optimizer updates per condition, ending at step 1100.
- Seed: 20260808.
- Batch size: 64; one GPU per run; all other loaded policy and optimizer settings unchanged.
- WP-Past input: preceding executed-length eight-action chunk for the first eight source positions;
  remaining source positions stay Gaussian. Samples crossing an episode boundary fall back to Gaussian.

The absolute CFM losses are not compared across source distributions. Training is valid only if losses
and gradient norms are finite, both runs save step 1100, and WP-Past reports a nonzero valid-history
fraction.

## Screening Evaluation

Evaluate each EMA checkpoint with its matching source distribution, ten Euler steps, seed 20260808,
ten environments, and 20 episodes under each sampling mode:

- Gaussian-trained checkpoint with Gaussian source: zero and random sampling.
- WP-Past-trained checkpoint with previous-action source: zero and random sampling.

Primary metric: random-sampling success count. Secondary metric: zero-sampling success count.

## Gate

Continue H3 only if WP-Past has more random-sampling successes than the matched Gaussian continuation
and its zero-sampling success rate is no more than 15 percentage points below Gaussian and remains at
least 50%. A tie or failure of either preservation criterion stops H3 at this budget and moves the main
effort to one-stage conditional reflow. A positive screen warrants a second seed or a longer, still
moderate continuation before any large experiment.

