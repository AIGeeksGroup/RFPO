# Protocol: Antithetic CFM Gradient-Estimator Audit

## Hypothesis

Joint antithetic pairs `(t, epsilon)` and `(1-t, -epsilon)` reduce the Monte Carlo variance of the
MC8 CFM gradient estimator used by FPO++ relative to eight independent samples, at identical model
evaluation cost and without changing either marginal sampling distribution.

## Locked Audit

- Checkpoint: `95j3noe4_step_6000`, EMA weights, Huber CFM loss as in FPO++
- Dataset: `ankile/robomimic-mh-can-image`
- Data: two deterministic batches of 16 pairs (32 fixed pairs total)
- Audit seed: 20260813
- Estimator: mean of 8 CFM samples per batch
- Repeats: 8 independent MC8 estimators per mode and batch
- Control: 8 i.i.d. uniform times and Gaussian noises
- Candidate: 4 i.i.d. draws plus their joint `(1-t, -epsilon)` counterparts
- Parameters: all non-vision actor parameters
- No online environment interaction or policy update

## Gates

For each of both dataset batches, candidate must:

1. reduce normalized gradient MSE to its mode-average gradient by at least 15%;
2. not reduce mean gradient-to-average cosine;
3. reduce the CV of the scalar MC8 loss estimator.

Also require finite, nonzero gradients and no material shift in the mode-average gradient direction
(cross-mode cosine at least 0.99). Stop without online training if any gate fails. Do not test
epsilon-only pairing, time-only pairing, other sample counts, or quasi-Monte Carlo variants after a
failure.
