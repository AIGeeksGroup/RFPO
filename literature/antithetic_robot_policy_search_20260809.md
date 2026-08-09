# Antithetic Robot-Policy Prior-Art Search (2026-08-09)

## Scope

This targeted search asks whether prior work applies symmetric Gaussian sources
`z` and `-z` to a diffusion/flow robot policy, averages the two generated action
endpoints, and evaluates the result under equal total neural-function-evaluation
(NFE) compute. It is a reproducible prior-art screen, not an exhaustive proof
that no such work exists.

## Databases and queries

The arXiv API endpoint was `https://export.arxiv.org/api/query`, with
`start=0`, relevance sorting, and the stated maximum result count. The OpenAlex
endpoint was `https://api.openalex.org/works`, with first-page retrieval and the
query in the `search` parameter.

| Database | Exact query | Returned / retrieved | Relevant outcome |
|---|---|---:|---|
| arXiv | `all:antithetic AND (all:"diffusion policy" OR all:"flow policy" OR all:robot OR all:control)` | 31 / 31 | No endpoint-averaged generative robot policy. Results were predominantly biochemical antithetic control, Monte Carlo estimators, and evolution-strategy work. |
| arXiv | `all:"antithetic noise" AND (all:"diffusion policy" OR all:"flow policy" OR all:robot OR all:control)` | 0 / 0 | No hit. |
| arXiv | `(all:"symmetric latent" OR all:"paired noise" OR all:"antithetic variates") AND (all:"diffusion policy" OR all:"generative policy" OR all:robot OR all:"robot control")` | 0 / 0 | No hit. |
| arXiv | `(all:antithetic OR all:"paired noise" OR all:"symmetric noise") AND (all:diffusion OR all:flow) AND (all:policy OR all:robot OR all:control OR all:action)` | 5 / 5 | One neighboring flow-matching paper, *Reward Transport*, but no robot-policy endpoint averaging. |
| OpenAlex | `"antithetic noise" "diffusion policy"` | 0 / 0 | No hit. |
| OpenAlex | Earlier query `antithetic diffusion policy` | 1 / 1 | The retrieved item was *Constraint-Aware Diffusion Priors for High-Fidelity and Versatile Quadruped Locomotion* (arXiv:2605.08804). Its “symmetric” component is command conditioning, not `z/-z` endpoint averaging. |

The broader OpenAlex Boolean search combining `"symmetric latent"`,
`"paired noise"`, or `"antithetic variates"` with robot/policy terms returned
69 items, but relevance was poor and manual inspection of the complete first
page found no direct generative-policy endpoint-average method. This result is
treated only as supporting coverage, not as a count of relevant papers.

## Direct and neighboring precedent

- Jia et al., *Antithetic Noise in Diffusion Models*, arXiv:2506.06185, is the
  direct precedent for pairing `z/-z` and exploiting negatively correlated
  diffusion or flow endpoints. Therefore this project must not claim invention
  of antithetic latent pairing or endpoint averaging in generative models.
- *Reward Transport: Property Control in Flow Matching via Noise-Space
  Alignment*, arXiv:2607.08781, is neighboring work on modifying flow-matching
  noise to control generated properties. It does not report the same symmetric
  two-endpoint robot-policy inference estimator or equal-total-NFE control
  benchmark.
- *Constraint-Aware Diffusion Priors for High-Fidelity and Versatile Quadruped
  Locomotion*, arXiv:2605.08804, uses diffusion in quadruped locomotion, but its
  symmetric augmented command conditioning addresses tracking symmetry rather
  than paired latent endpoints.

An OpenAlex citation-neighborhood query for Jia et al. used work ID
`W4417097706` and `filter=cites:W4417097706` with `per_page=100`. OpenAlex
reported zero citing works and supplied zero indexed references for the work.
Because the source is a recent preprint, this empty neighborhood is treated as
an indexing limitation rather than evidence that no follow-up or cited prior
work exists.

## Defensible novelty boundary

The contribution can be framed around control-specific evidence: closed-loop
robot-policy behavior, comparison against an equal-NFE IID two-endpoint
estimator, equal-total-NFE comparison against one random endpoint, independent
training seeds and official tasks, and source-cancellation/affine-antisymmetry
mechanism measurements. It cannot be framed as the invention of antithetic
noise or `z/-z` averaging.

## Coverage limitations

The search covers arXiv and OpenAlex as indexed on 2026-08-09 and uses English
keyword queries. Search ranking, incomplete abstracts, terminology mismatch,
unindexed workshop papers, and papers published after the access date can hide
related work. A final submission should repeat the queries and add citation- and
reference-neighborhood checks around arXiv:2506.06185 and any newly found direct
robot-policy paper.
