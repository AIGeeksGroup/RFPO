# Mania et al. (2018): Augmented Random Search

- Title: Simple random search provides a competitive approach to reinforcement learning
- Authors: Horia Mania, Aurelia Guy, Benjamin Recht
- arXiv: 1803.07055
- DOI: 10.48550/arXiv.1803.07055
- OpenAlex: W2789525339
- Year: 2018
- Accessed: 2026-08-09

The paper challenges the assumption that parameter-space exploration must be less sample efficient
than action-space policy gradients on continuous-control tasks. Its use of mirrored directions and
careful empirical variance analysis motivates H53's common-seed paired audit. The paper's strongest
results use linear policies and large hyperparameter studies, so H53 deliberately tests signal
identifiability before transferring an update rule to a visual flow actor.

Provenance: targeted OpenAlex single-work lookup at
`/works/https://doi.org/10.48550/arXiv.1803.07055`, selecting title, year, authors, citations, and
abstract metadata.
