# Salimans et al. (2017): Evolution Strategies for RL

- Title: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
- Authors: Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever
- arXiv: 1703.03864
- DOI: 10.48550/arXiv.1703.03864
- OpenAlex: W2596367596
- Year: 2017
- Accessed: 2026-08-09

The paper treats policy optimization as black-box parameter search and estimates gradients through
mirrored Gaussian perturbations and scalar episodic returns. Its relevant advantages for H53 are
invariance to action frequency and delayed rewards, and the absence of temporal discounting or a
value function. H53 does not claim that the original large-worker ES configuration transfers
directly; it first tests whether a fixed 16-direction subspace exposes a reproducible Square success
signal at a small budget.

Provenance: targeted OpenAlex single-work lookup at
`/works/https://doi.org/10.48550/arXiv.1703.03864`, selecting title, year, authors, citations, and
abstract metadata.
