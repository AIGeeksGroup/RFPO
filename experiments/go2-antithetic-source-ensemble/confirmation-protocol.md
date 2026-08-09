# H66 Independent 4,096-Environment Confirmation Protocol

## Authorization

The locked H66 screen passed with a paired random-return gain of `+1.36131` and bootstrap 95% interval
`[1.02934, 1.81896]`. Its fixed-state audit also established exact zero-path identity, exact
antithetic pairing and averaging, active candidate actions, and unchanged actor parameters. This
authorizes the independent confirmation specified in the original H66 protocol.

## Frozen Evaluation

- Checkpoint and candidate rule remain exactly those in `protocol.md`.
- Conditions: official single Gaussian endpoint and one-pair `z/-z` endpoint average.
- Environments and episodes per method: 4,096, exactly one completed episode per environment.
- Evaluation seed: `20261103`.
- Gaussian source seed: `20261104`.
- Bootstrap seed: `20261105`; resamples: 20,000.
- Solver: Euler-64 per endpoint. Control uses 64 NFE and candidate uses 128 NFE per action.
- Run each method in a fresh process. Require exact initial-observation and complete pre-generated
  Gaussian source-stream hashes across methods.
- Zero mode is not rerun: the candidate dispatches the unmodified single-endpoint zero path, whose
  bitwise identity passed Stage A.

## Confirmation Gates

All gates must pass:

1. all 8,192 episodes complete with finite actions and returns;
2. initial-observation hashes, source-stream hashes, source counts, checkpoint, task, and all frozen
   evaluation metadata match;
3. candidate-minus-control paired mean return is at least `+0.20`;
4. the deterministic paired-bootstrap 95% lower bound is strictly positive.

A full pass supports H66 as an official-scale random-mode improvement with unchanged zero-mode
behavior. Any failure refutes stable improvement and stops the method without pair-count, scale,
weight, solver, checkpoint, or seed variants. The method's twofold Gaussian inference compute must
remain explicit in every claim.
