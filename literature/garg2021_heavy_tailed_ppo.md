# On Proximal Policy Optimization's Heavy-tailed Gradients

- Authors: Saurabh Garg, Joshua Zhanson, Emilio Parisotto, Adarsh Prasad, J. Zico Kolter,
  Zachary C. Lipton, Sivaraman Balakrishnan, Ruslan Salakhutdinov, and Pradeep Ravikumar
- Year: 2021
- arXiv: https://arxiv.org/abs/2102.10264
- DOI: https://doi.org/10.48550/arXiv.2102.10264

## Main Result

The paper measures heavy-tailed PPO actor gradients and attributes much of the effect to likelihood
ratios and advantages as the policy moves away from the behavior policy. It proposes geometric
median-of-means (GMOM), a high-dimensional robust gradient estimator, and reports performance
competitive with clipped PPO on continuous-control tasks with fewer clipping heuristics.

## Relevance to FPO++

FPO++ inherits both implicated factors: advantage-weighted likelihood-ratio gradients and repeated
epochs away from a fixed behavior policy. H34's coordinate-wise median reduced outcome alignment,
suggesting that independent coordinate filtering destroys useful joint structure. GMOM is a distinct
test because it selects a robust center in the complete gradient-vector geometry.

## Retrieval Provenance

- Accessed: 2026-08-09
- arXiv API: `https://export.arxiv.org/api/query`
- Query: `all:"proximal policy optimization" AND (all:"gradient" OR all:"trust region")`
- OpenAlex entity lookup: `https://api.openalex.org/works/doi:10.48550/arxiv.2102.10264`
- OpenAlex fields: title, year, citation count, authors, abstract, open-access metadata

