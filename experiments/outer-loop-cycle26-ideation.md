# Outer-Loop Cycle 26: Policy-Divergence Control

## Reflection

H38 established a large equal-NFE endpoint-fidelity gain, but H38b showed that numerical fidelity
alone does not improve pooled Can reward. H18 and H23 provide the more reward-relevant clue: the
held-out FPO++ gradient rotates during optimization, while a globally smaller fixed learning rate
sacrifices progress without preserving direction. The next candidate should therefore react to
measured policy movement rather than reduce every update uniformly.

Code inspection also rejected an initially plausible action-suffix hypothesis without an
experiment. The policy predicts 16 actions, `select_action` returns only the configured eight
executed actions, rollout storage contains those eight actions, and `forward_fpo` consequently uses
`T=8`. There is no gradient through an unexecuted eight-action suffix in the released manipulation
loop.

## Candidates

| Rank | Candidate | Distinct mechanism | Decision |
|---:|---|---|---|
| 1 | FPO++ `x1_pred` KL-adaptive actor LR | react to measured flow-policy divergence | Select H39 |
| 2 | Kozachenko-Leonenko action entropy | preserve non-parametric exploration support | Park behind H39; introduces a coefficient and neighbor choice |
| 3 | Small action perturbation during rollout | direct entropy proxy used by locomotion | Park; changes benchmark source distribution |
| 4 | KL penalty in the actor objective | constrain the same `x1_pred` proxy continuously | Reject for now; adds an untuned coefficient |
| 5 | KL-triggered epoch stopping | stop after divergence is observed | Reject; repeats refuted H18 stopping logic |
| 6 | Held-out Armijo line search | accept only surrogate-improving steps | Reject; trains against the audit sensor and adds optimizer-state ambiguity |
| 7 | Per-layer adaptive clipping | limit outlier parameter blocks | Park; no paper-supported threshold |
| 8 | Natural-gradient approximation | precondition policy movement | Reject at this stage; high implementation and compute cost |
| 9 | Extragradient FPO++ | anticipate the post-update gradient | Park; H24 weakens update-path-only explanations |
| 10 | SVRG across update epochs | reduce repeated-sample gradient variance | Park; H16 found only modest stored/held-out ratio differences |
| 11 | Unexecuted action-suffix masking | remove gradients for actions not sent to the environment | Reject by code inspection; the suffix is already absent |
| 12 | Reopen midpoint/Heun variants | improve integration fidelity | Reject; H38b closed solver variants after the locked reward failure |

## Selection

Appendix D.2 of the FPO++ paper says entropy regularization and KL-adaptive learning rates produced
some improvement. More importantly, the released motion-tracking implementation supplies the
otherwise omitted controller: cache behavior-policy `x1_pred`, measure the mean squared change,
divide or multiply LR by 1.5 outside a two-sided target band, and clamp the LR. H39 transfers that
defined controller to manipulation and first asks whether it improves the already-established
held-out direction failure. No reward training is permitted unless the mechanism audit passes.

