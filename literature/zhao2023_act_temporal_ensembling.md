# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

- Authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
- Year: 2023
- Venue: Robotics: Science and Systems XIX
- DOI: 10.15607/RSS.2023.XIX.016
- OpenAlex: W4385430674
- Project code: https://github.com/tonyzhaozh/act
- Code revision audited: `742c753c0d4a5d87076c8f69e5628c79a8cc5488`
- Retrieved: 2026-08-09

## Relevant Mechanism

ACT predicts overlapping action chunks and temporally ensembles all predictions that cover the
current time. The released evaluator queries every step when temporal aggregation is enabled and
weights populated predictions by normalized `exp(-0.01 * i)`, where `i` follows stored prediction
order. With exactly two predictions, the older and newer predictions receive weights
`1 / (1 + exp(-0.01)) = 0.50249998` and `exp(-0.01) / (1 + exp(-0.01)) = 0.49750002`.

## Relevance to FPO++

The released FPO policy predicts 16 actions but executes only the first eight before replanning.
The unused eight-action continuation and the next observation-conditioned eight-action prefix cover
the same environment times. Their fixed weighted average can reduce action discontinuity without
changing the checkpoint, source distribution, Euler solver, eight-step replanning cadence, or number
of network calls.

## Scope and Risk

H54 is ACT-inspired rather than an exact ACT reproduction because it retains one query every eight
steps instead of querying every step. An old continuation can be smooth but stale after contact or
object motion, so a mechanism gate must show both reduced boundary jump and a material contribution
from the new observation-conditioned prediction before reward evaluation.
