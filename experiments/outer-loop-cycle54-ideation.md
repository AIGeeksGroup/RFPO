# Outer Loop Cycle 54: Reward-Relevant Dense Go2 Inference

## Failure Boundary

H65 moved the actor by 3.51% relative L2 but changed paired zero/random return by only
`-0.0080/+0.0017`. Late checkpoint variation is therefore not complementary reward structure, and
checkpoint-window or soup tuning would select noise. H60 also shows that reducing integration error
does not improve return by itself. The next candidate needs a direct mechanism tied to a measured
reward deficit rather than parameter smoothing or path fidelity.

The reproduced final Go2 actor has a repeatable source-mode gap: zero-source return is about 41.5,
while Gaussian-source return is about 40.5. H26's Can antithetic average did not beat zero-source
behavior, but across its two locked seeds it tied zero exactly at 52/70 successes. That negative
result is evidence that symmetric source averaging may recover a central action, even though it
cannot exceed the central policy on sparse Can.

## Diverged Candidates

1. One-pair antithetic source endpoint averaging in Gaussian Go2 inference.
2. Two independent Gaussian endpoint averaging.
3. A fixed reduced Gaussian source temperature.
4. A learned global deterministic source optimized by low-dimensional ES.
5. Mirrored actor-parameter ES using dense episodic return.
6. Low-rank actor adapters optimized by ES.
7. Short actor-only FPO++ continuation with a reduced learning rate.
8. Critic reset followed by short FPO++ continuation.
9. Post-RL reflow distillation of the final actor.
10. Reward-weighted post-RL reflow distillation.
11. Output ensembling of the five late checkpoints.
12. Temporally correlated Gaussian sources during locomotion inference.

## Convergence

Select candidate 1 as H66. It has one coefficient-free choice, directly targets the measured random-
source penalty, and leaves zero-source inference mathematically unchanged. A fixed-state audit can
first verify exact antithetic sources, exact endpoint averaging, activity relative to zero, finite
outputs, and unchanged parameters. Only then are two paired 256-environment random-source rollouts
needed. Candidate 2 lacks the odd-component cancellation guarantee; candidate 3 reopens closed
source-scale tuning; candidates 4-8 need a new optimization pipeline; candidates 9-10 inherit the
lack of a supported reflow-to-reward mechanism; candidate 11 follows H65's neutral late-actor
geometry; and candidate 12 already produced source-mode tradeoffs in manipulation.

H66 does not reopen H26's claim that an antithetic average beats zero-source Can behavior. It tests a
different, dense-return prediction: symmetric averaging should reduce the official Gaussian-mode
deficit toward an exactly preserved zero-source policy. Its cost is approximately two actor solves,
which must be reported if the reward screen succeeds.

Two-sentence pitch: stochastic FPO++ inference loses about one Go2 return point relative to the same
actor's central source, despite being useful for exploration during training. Symmetric flow
transport from `z` and `-z` cancels source-odd endpoint variation and may recover that point without
changing the checkpoint or deterministic behavior.
