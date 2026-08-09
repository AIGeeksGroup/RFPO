# Protocol: H50 Asymmetric Square Critic Audit

## Hypothesis

A critic trained on simulator state ranks complete-episode discounted Square success more accurately
than the official critic trained on the frozen actor's visual conditioning. Better value information
is a prerequisite for a useful FPO++ advantage or an RLDT transport field.

## Locked Collection

- Actor: released Square `trc7rbt0_step_110000` EMA checkpoint, bitwise frozen.
- Policy input and behavior: unchanged RGB plus proprioception, Gaussian source, Euler-10, and 16
  executed actions per plan.
- Reward: original sparse binary Square reward; no shaping or stage relabeling.
- Episodes: 32 complete episodes, collected as two 16-environment blocks from consecutive seeds
  `20260926` through `20260957`, exactly one episode per seed.
- Renderer: paired OSMesa screen. This is a mechanism audit, not an official-renderer benchmark.
- Records: frozen visual conditioning, simulator privileged state, per-plan discounted reward,
  duration, terminal flag, episode identifier, and final success.
- Privileged state: robot end-effector position/quaternion and gripper position, square-nut world
  position/quaternion, nut-to-end-effector relative position/quaternion, and target-peg position.
- Labels: `0.995 ** remaining_environment_steps` for successful episodes and zero for failures,
  evaluated at every recorded plan boundary.

Collection is valid only if all records are finite, every seed contributes one terminal episode, and
the 32 episodes contain at least eight successes and eight failures. Invalid collection stops without
changing seeds or source mode.

## Locked Cross-Fit

Sort successful and failed episodes separately by seed and assign alternating episodes to folds. For
each direction, train on one fold and evaluate on the other; no plan from one episode may cross the
split.

- Control input: the frozen actor conditioning used by the official scalar critic.
- Candidate input: the fixed privileged-state vector only.
- Both models: official critic widths `256, 128, 64`, ReLU, scalar output, independently seeded but
  otherwise identical initialization procedure.
- Normalize each input using training-fold moments only.
- Objective: MSE to complete-episode discounted-success labels.
- Optimizer: AdamW, learning rate `1e-4`, epsilon `1e-5`, weight decay `1e-6`.
- Budget: ten epochs, eight deterministic seed-shuffled minibatches per epoch.
- Metrics: held-out plan-level MSE and Spearman rank correlation, reported separately for both folds.

## Gate and Stop Rule

H50 passes only if, in both held-out folds, the asymmetric critic reduces MSE by at least 10%, raises
Spearman correlation by at least 0.10, and has positive Spearman correlation. A pass authorizes a
separately committed actor-gradient audit before any online reward training. Failure closes H50
without changing the state vector, network, optimizer, loss, split, seed, collection size, or gate.

This experiment is not RLDT. It audits a prerequisite value-learning assumption before paying for
RLDT's replay, double-Q, SVGD particles, and actor updates.

