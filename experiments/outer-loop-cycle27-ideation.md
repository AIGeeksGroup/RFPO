# Outer-Loop Cycle 27: Move Method Screening to a Non-Saturated Benchmark

## Reflection

Can step 6000 made optimizer audits information-rich but reward screens unstable or saturated, while
step 1000 made short online training too sparse. H39 also closes adaptive step-size control: it
improved active ratios by behaving like a fixed low LR and discarded most progress. The released
repository already contains every manipulation checkpoint, so the next scientific decision is to
identify a paper benchmark with enough headroom and reward signal for inexpensive paired screens.

Appendix D.5 warns that extra entropy and ASPO can be detrimental for pretrained manipulation
policies. That evidence rejects a reflexive entropy experiment despite Appendix D.2's motion-
tracking observation.

## Candidates

| Rank | Candidate | Rationale | Decision |
|---:|---|---|---|
| 1 | Square base-policy regime | paper reports about 28.6% base success and meaningful FPO++ gain | Select H40 regime screen |
| 2 | Threading base-policy regime | precision task and FPO++ ablation benchmark | Park behind Square; two-arm simulation is costlier |
| 3 | Box Clearance regime | available released checkpoint | Park; reward structure and two-arm cost complicate first screen |
| 4 | Tray Lifting regime | available released checkpoint | Park; long horizon and two-arm cost |
| 5 | Can step-1000 method screen | matches main benchmark exactly | Reject for now; prior short training is reward-starved |
| 6 | Can step-6000 method screen | dense standard-Gaussian success | Reject for next cycle; several reward controls approach saturation |
| 7 | kNN entropy regularization | repository-defined non-parametric entropy | Reject for manipulation; Appendix D.5 predicts undesirable behavior |
| 8 | Action perturbation entropy | locomotion implementation already provides it | Reject; changes exploration and conflicts with D.5 |
| 9 | Exact CNF likelihood ratios | removes the variational ratio proxy | Park; second-order divergence gradients are too costly before a viable regime |
| 10 | Natural-gradient preconditioning | directly controls policy geometry | Park; substantial infrastructure for an unvalidated premise |
| 11 | Reuse zero-endpoint PCGrad on Square | same supported mechanism in a harder task | Park until H40 establishes a valid screen |
| 12 | Reopen adaptive LR target/bounds | may avoid lower-bound collapse | Reject by H39's locked no-tuning rule |

## Selection

Evaluate the exact released Square base checkpoint under balanced zero and Gaussian-source episodes.
This is not an improvement claim. It is a cheap gate for whether Square can replace Can step 6000 as
the small-scale method-selection regime before any new training protocol is registered.

