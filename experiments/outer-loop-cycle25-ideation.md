# Outer-Loop Cycle 25: Numerical Sampling Before Heavier Optimization

## Reflection

H24, H36, and H37 make SVRG or outer-optimizer integration a high-cost bet: neither full-batch
updates, ratio rollback, nor robust mean estimation improved the required direction consistently.
The supported reflow result instead shows that endpoint integration error is measurable and task
performance can be preserved at low NFE. A numerical-solver intervention can test a direct,
training-free path to higher Can success before adding another optimizer mechanism.

## Candidates

| Rank | Candidate | Cost | Decision |
|---:|---|---:|---|
| 1 | Explicit midpoint, 5 steps | 10 NFE | Select H38 |
| 2 | Heun, 5 steps | 10 NFE | Park; do not solver-sweep |
| 3 | RK4, 2 steps plus endpoint correction | 9-10 NFE | Reject; irregular schedule and implementation surface |
| 4 | Euler-20 | 20 NFE | Reject; unequal compute and weak methodological claim |
| 5 | SVRG FPO++ updates | multiple backward passes | Park; H24 weakens its motivating mechanism |
| 6 | Outer-PPO momentum | multi-update training | Park; H25 already rejects one-update interpolation |

## Selection

Explicit midpoint is the simplest second-order solver with an exact equal-NFE comparison to the
released Euler sampler. The fixed endpoint audit can reject it within minutes; only a replicated
error reduction under both source modes permits a small environment screen.

