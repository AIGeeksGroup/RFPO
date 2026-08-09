# Group-Relative Episodic Policy Optimization

## Sources

### DeepSeekMath

- Title: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
- Authors: Shao et al.
- Year: 2024
- Identifier: arXiv `2402.03300v3`
- URL: https://arxiv.org/abs/2402.03300

DeepSeekMath introduces Group Relative Policy Optimization (GRPO), which replaces a learned value
baseline with rewards normalized among multiple outputs for the same prompt. The transferable
principle is common-context sampling: conditional on one context, other samples provide a baseline
that does not depend on the current sample's action.

### Bernoulli-Continuation Policy Learning

- Title: *Continue or Replan? Bernoulli-Continuation Policy Learning for Adaptive Horizon Execution*
- Authors: Xu et al.
- Year: 2026
- Identifier: arXiv `2608.03483v1`
- URL: https://arxiv.org/abs/2608.03483

BCP applies trajectory-level GRPO to a robot chunk-policy continuation head. Each training group
contains seven adaptive trajectories and one fixed-horizon reference. This establishes a direct
robot-control precedent, but its 256-environment, eight-A100 training recipe is too large for an
unvalidated FPO++ modification.

## FPO++ Opportunity

Square's released random-source policy succeeds on about 35% of the locked balanced screen. Four
independent policy trajectories from the same deterministic simulator initialization therefore have
approximately

`1 - 0.35^4 - 0.65^4 = 0.806`

probability of containing both a success and a failure if outcomes are conditionally independent.
For replica `i`, use the leave-one-out advantage

`A_i = R_i - mean(R_j for j != i)`.

The baseline depends on the common initial scene and the other replicas, but not on replica `i`'s
actions. It is therefore a valid episodic REINFORCE control variate. Unlike H62's local vines, this
uses the full policy trajectory and should avoid mostly tied suffix outcomes. Unlike BCP, it does not
add a continuation head or require an explicit policy likelihood: the existing FPO++ CFM ratio can
be weighted by the episodic advantage at every valid chunk.

## Retrieval Provenance

- arXiv API: `GET https://export.arxiv.org/api/query`, `id_list=2402.03300,2608.03483`,
  `max_results=2`, accessed 2026-08-09. Both identifiers and titles matched.
- OpenAlex exact DOI lookup was also attempted. It resolved BCP correctly as work `W7172487171`,
  but attached an incorrect title to arXiv `2402.03300`; the arXiv record is authoritative here.

