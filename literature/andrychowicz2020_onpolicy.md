# What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study

- Authors: Marcin Andrychowicz et al.
- Year: 2020
- arXiv: 2006.05990
- DOI: 10.48550/arXiv.2006.05990
- OpenAlex: W3035700320
- URL: https://arxiv.org/abs/2006.05990

## Relevance

The paper evaluates more than 50 low- and high-level implementation choices across over 250,000
on-policy continuous-control agents. It supports treating minibatch construction and normalization
as consequential algorithmic choices rather than incidental code details. It does not directly test
advantage-sign-stratified minibatches, so H45 remains a new empirical hypothesis rather than a
literature-established improvement.

## Retrieval Provenance

- Targeted lookup accessed 2026-08-09.
- OpenAlex direct DOI endpoint
  `GET https://api.openalex.org/works/doi:10.48550/arxiv.2006.05990` returned the title, authors,
  year, abstract, citation count, and work ID `W3035700320`.
- Two bounded OpenAlex topic searches for stratified/balanced policy-gradient minibatches returned
  49 and 654 indexed works. Their leading relevant results concerned off-policy replay or general
  Monte Carlo variance reduction; no direct on-policy advantage-sign minibatch precedent was found.
