# Protocol Correction: Deterministic H46 Evaluation Environments

## Invalidation

After the first H46 reward screen passed, code tracing showed that `eval_checkpoint.py` seeded Python,
NumPy, and Torch in the parent process but did not pass per-environment seeds to
`create_vectorized_env`. Async workers use spawn, and `RobosuiteGymWrapper` therefore passed
`seed=None` to `robosuite.make`. The 5/20 control and 8/20 candidate cells are individually balanced
samples, but they do not implement the protocol's intended evaluation seed and cannot authorize a
confirmation. The candidate's curvature-selection audit remains valid because it does not depend on
matched environment placement.

## Locked Correction

- Pass deterministic seeds `[cfg.seed + env_id for env_id in range(cfg.eval_num_envs)]` from
  `eval_checkpoint.py` to `create_vectorized_env`; preserve `[None]` behavior when `cfg.seed` is unset.
- Add a regression test that inspects the evaluation entry point and requires this exact seed mapping.
- Keep checkpoint, seed `20260922`, 20 environments, one episode per environment, ordinary-random
  control, curvature-best-of-two candidate, Euler-10, 16 actions, no EMA, OSMesa, evaluation order,
  and the +3/20 gate unchanged.
- Write corrected runs to fresh names ending in `_seeded`; do not overwrite or discard the invalid
  artifacts.

If the corrected candidate gains fewer than 3/20 successes, refute H46 without confirmation,
another seed, or method variants. If it gains at least 3/20, separately lock the 50-episode
confirmation. The correction is infrastructure validation, not permission to tune the method.
