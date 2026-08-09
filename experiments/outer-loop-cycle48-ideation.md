# Outer Loop Cycle 48: Replace Learned Credit with Replay-Exact Branch Returns

## Failure Boundary

H61 closes direct current-CFM anchoring after H2 had already closed offline reflow as a gradient-stability mechanism. H19, H21, and H50 show that a more elaborate critic is not yet justified, while H53 shows that complete-episode parameter perturbations produce irreproducible binary-return rankings. The remaining actionable question is whether real returns become informative when alternatives are compared much later in a trajectory from the exact same intermediate simulator state.

## Divergent Candidates

| Rank | Candidate | Fast falsification | Decision |
|---:|---|---|---|
| 1 | Replay-exact branch-return FPO | Two independent continuation blocks from common Square prefixes | Select H62 |
| 2 | Direct VINE value-gradient training | Action-gradient audit plus new twin-Q/replay infrastructure | Park; too many coupled changes and critic prerequisite is unsupported |
| 3 | Polychromic FPO with semantic diversity | Vine return audit, then task-specific diversity definition | Park diversity term; first test whether vine returns rank actions at all |
| 4 | Initial-state GRPO over complete episodes | Group-return reward screen | Reject as insufficiently localized; H15 and H53 already expose trajectory-level bias/noise |
| 5 | Learned latent-source success proposal | Short source-head training | Reject; H8-H10 and H48-H49 close nearby source-transfer and latent-modulation routes |
| 6 | More flow anchors, solver variants, or coefficient tuning | Fixed-batch geometry audit | Reject by H1/H2/H60/H61 |

## Selected Hypothesis

H62 uses a problem-first decomposition: FPO++ needs action-specific sparse-reward credit, but its short-budget critic is unreliable. Exact prefix replay lets two policy action chunks be compared from the same physical and visual state, while common continuation source sequences reduce future-policy variance. Two disjoint continuation blocks test whether the comparison survives new future noise.

The strongest objection is that Square's binary return remains too sparse for branch rankings to reproduce even with common states and common random numbers. Therefore H62 is a no-update prerequisite. It must show cross-block correlation, sign agreement, and held-out top-quartile gain before any branch-return actor update is implemented.
