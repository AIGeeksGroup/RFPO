# Continue or Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon Execution

- Authors: Weichen Xu et al.
- Year: 2026
- Identifier: arXiv:2608.03483v1
- URL: https://arxiv.org/abs/2608.03483
- Project page: https://fleetfootwork.github.io/BCP/
- Retrieved: 2026-08-09 via `https://export.arxiv.org/api/query?id_list=2608.03483`
- Full text inspected: 14-page arXiv PDF, version 1

## Relevant Mechanism

BCP freezes the base chunked policy and learns a lightweight head that selects how much of each
predicted action chunk to execute. Candidate horizons are ordered. The head represents each longer
horizon as a sequence of Bernoulli continue decisions, giving an exact likelihood for GRPO rather
than requiring the implicit flow-policy ratio. It conditions on visual-language tokens, denoised
actions, and final-step action-velocity features.

Training uses groups of seven adaptive trajectories plus one fixed-horizon reference. A trajectory
success reward is augmented by a bounded reference-relative efficiency term based on the number of
base-policy calls. The paper uses a two-layer Transformer continuation head, group size 8, 256
parallel environments, 300 GRPO steps, and eight A100 GPUs. This is not a small first experiment.

## Prerequisite Evidence

Before training BCP, the paper performs a phase-shift experiment: only the first chunk's executed
prefix changes, after which the same fixed execution horizon is restored. Across 50 RoboTwin tasks,
the per-task best and worst phases produce an 11.30-point success gap. This isolates replanning-boundary
alignment from globally shorter horizons and motivates learning a state-dependent schedule.

## Relevance to FPO++

FPO predicts 16 actions and executes eight. H27 showed that executing four actions at every replan is
not better, but did not test whether the eight-step schedule is phase-misaligned with grasp or place
events. A common-seed phase audit is therefore a cheap prerequisite before implementing a continuation
head. If Can outcomes do not change across first-chunk phases, BCP has no demonstrated headroom on
this checkpoint and its training infrastructure should not be built.
