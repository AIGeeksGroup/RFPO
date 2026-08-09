# Polychromic Objectives for Reinforcement Learning

- Authors: Jubayer Ibn Hamid, Ifdita Hasan Orney, Ellen Xu, Chelsea Finn, Dorsa Sadigh
- Year: 2025, revised 2026
- arXiv: `2509.25424v6`
- URL: https://arxiv.org/abs/2509.25424
- Retrieved: 2026-08-09 through the arXiv API and PDF endpoint

## Method

Polychromic PPO optimizes a set-valued reward and diversity objective. Its data collection uses vine sampling: collect seed trajectories, choose intermediate rollout states, reset the environment to each state, and generate `N` independent continuations. The paper uses `N=8` vines, set size `n=4`, and two rollout states per seed trajectory. Monte Carlo set returns replace GAE at vine states; ordinary GAE is retained elsewhere.

## Relevance to FPO++

The transferable prerequisite is not the paper's task-specific semantic diversity score. It is the ability to obtain repeatable action rankings from multiple on-policy continuations of the same simulator state without learning a critic. For Square, exact low-level prefix replay can construct those common states without relying on incomplete MuJoCo state snapshots. H62 tests whether binary branch returns contain reproducible local action information before adding a branch-return FPO objective.

## Retrieval Provenance

- Metadata query: `GET https://export.arxiv.org/api/query` with `(all:"vine sampling" OR all:"branching rollouts" OR all:"rollout from intermediate states") AND all:"policy gradient"`, `start=0`, `max_results=20`, sorted by relevance.
- Full text: `https://arxiv.org/pdf/2509.25424`.
- The targeted arXiv query returned one record.
