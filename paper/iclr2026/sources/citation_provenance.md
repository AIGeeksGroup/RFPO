# Citation provenance

Access date: 2026-08-09.

The arXiv entries in `references.bib` were checked against targeted calls to
`https://export.arxiv.org/api/query` with these identifiers:

- `2602.02481` (Yi et al., Flow Policy Gradients for Robot Control)
- `2507.21053` (McAllister et al., Flow Matching Policy Gradients)
- `2506.06185` (Jia et al., Antithetic Noise in Diffusion Models)
- `2209.03003` (Liu et al., Rectified Flow)
- `2210.02747` (Lipman et al., Flow Matching)
- `1707.06347` (Schulman et al., PPO)
- `2303.04137` (Chi et al., Diffusion Policy)
- `1703.03864` (Salimans et al., evolution strategies)
- `2605.04732` (Yadav et al., rollout common random numbers)

For each entry, author order, title, first-submission year, identifier, and
primary category were matched to the Atom response. The SIGIR entry was checked
through Crossref endpoint
`https://api.crossref.org/works/10.1145%2F3209978.3210068`; its title, authors,
venue, pages, year, publisher, and DOI match the returned work record.

The local novelty search notes under `literature/` record the query scope and
known coverage limits. In particular, unauthenticated Semantic Scholar requests
returned HTTP 429, so the no-direct-precedent statement is deliberately phrased
as a bounded search result rather than an exhaustive novelty proof.
