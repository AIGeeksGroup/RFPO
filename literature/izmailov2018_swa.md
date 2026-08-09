# Averaging Weights Leads to Wider Optima and Better Generalization

- Authors: Pavel Izmailov, Dmitry Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
- Year: 2018
- arXiv: 1803.05407
- DOI: 10.48550/arXiv.1803.05407
- OpenAlex: W2792287754
- URL: https://arxiv.org/abs/1803.05407

## Relevance

Stochastic Weight Averaging (SWA) averages multiple parameter points along one optimization
trajectory to obtain a single model. The paper reports wider solutions and improved generalization
at almost no inference overhead. For FPO++, this motivates averaging all actor-update checkpoints
from one fixed short online trajectory rather than selecting a checkpoint or interpolating the BC
anchor with one final policy.

## Retrieval Provenance

- Targeted lookup accessed 2026-08-09.
- arXiv API query `id_list=1803.05407&max_results=1` returned rate-limit errors twice.
- Semantic Scholar paper lookup `ARXIV:1803.05407` returned HTTP 429.
- OpenAlex exact-title search resolved work `W2792287754`; a direct
  `GET https://api.openalex.org/works/W2792287754` supplied title, authors, year, DOI, and abstract.
