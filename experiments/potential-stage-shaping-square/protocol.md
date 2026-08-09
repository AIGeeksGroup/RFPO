# Protocol: H51 Potential-Based Square Stage Shaping

## Hypothesis

Potential-based reach/grasp/lift/hover shaping supplies signed local progress to Square GAE while
preserving the sparse-success objective, improving Gaussian-source success without degrading the
released policy's deterministic behavior.

## Locked Reward

- Sparse reward `r` and success termination remain exactly those returned by RoboSuite Square.
- `Phi(s)` is `max(reach, grasp, lift, hover)` from RoboSuite's existing `staged_rewards()` method.
- Candidate training reward is `r' = r + 0.995 * Phi(s_next) - Phi(s)` with coefficient exactly one.
- `Phi(s_next)=0` on success or timeout. The initial potential is measured after environment reset.
- Control training uses the unchanged sparse reward. Evaluation always uses sparse reward only.
- No stage thresholds, learned reward model, clipping, normalization, coefficient sweep, or additional
  intrinsic reward is allowed.

## Stage A: Frozen-Policy Mechanism Audit

- Actor: released Square `trc7rbt0_step_110000` EMA checkpoint, bitwise frozen.
- Behavior: Gaussian source, Euler-10, 16 executed actions per plan.
- Collection: 16 complete OSMesa episodes with consecutive environment seeds `20260958..20260973`,
  one episode per seed. Actions do not depend on the shaped reward.
- Record each primitive transition's sparse reward, current and next potential, shaping term, candidate
  reward, terminal flag, and episode identifier.
- Verify for every episode that the discounted candidate return minus discounted sparse return equals
  `-Phi(s_0)` within absolute tolerance `1e-5`.
- Among nonterminal transitions from failed episodes, require at least 20% to have absolute shaping
  term above `1e-4`, with at least 5% positive and at least 5% negative terms.
- All rewards and potentials must be finite; potentials must lie in `[0, 0.7]`, and every seed must
  contribute exactly one complete episode.

Failure of any Stage A condition stops H51 without changing seeds, thresholds, potential definition,
discount, or collection size.

## Stage B: Paired Short Training

Stage B is authorized only if Stage A passes.

- Initialization: released Square `trc7rbt0_step_110000` checkpoint for both control and candidate.
- Pairing: identical seed `20260974`, environment seeds, checkpoint, Gaussian source, Euler-10,
  16-action execution, observations, optimizer, schedulers, CFM variables, and FPO++ objective.
- Budget: five iterations of 320 primitive steps across 16 environments per method, or 25,600
  environment interactions each. The first iteration remains critic-only as in the released setup.
- Hyperparameters: official Square FPO++ settings used by H41/H43-H45; the only candidate difference
  is the locked reward transformation above.
- Validity: initial action and sparse-reward trajectories must match exactly until parameter updates
  can affect behavior; all model, reward, loss, and ratio values must remain finite.

## Locked Evaluation and Gate

- Evaluate final EMA actors on sparse Square success at seed `20260975`.
- Use 20 deterministic environments contributing exactly one episode each in zero-source mode and
  another 20 in Gaussian-source mode, Euler-10 and 16 executed actions.
- H51 passes only if candidate minus control is at least `+2/20` Gaussian successes, no worse than
  `-1/20` zero-source successes, and at least `+3/40` pooled successes.
- A pass authorizes a separately committed independent confirmation. Failure stops without another
  seed, reward coefficient, potential variant, training duration, checkpoint choice, or reward gate.

This is a paired OSMesa method screen, not the official EGL-renderer benchmark and not a claim of
real-robot performance.

