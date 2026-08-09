# Antithetic Noise in Diffusion Models

- Authors: Jing Jia, Sifan Liu, Bowen Song, Wei Yuan, Liyue Shen, Guanyang Wang
- Year: 2025; arXiv version updated 2026-01-30
- Identifier: arXiv:2506.06185v2
- URL: https://arxiv.org/abs/2506.06185

The paper studies initial-noise pairs `z` and `-z` in diffusion models and finds
strong negative endpoint correlation across datasets, architectures, conditional
and unconditional generation, VAEs, and normalizing flows. It proposes that the
learned score is approximately affine antisymmetric and uses the resulting
correlation primarily for uncertainty quantification, reporting confidence
interval reductions up to 90%, plus image editing and diversity applications.

This is the closest known precedent for H66-H69 and prevents claiming
antithetic latent pairing itself as novel. The current project differs in
averaging paired conditional flow-policy endpoints into one closed-loop robot
action, measuring task return, comparing against IID pairing at equal NFE, and
testing deterministic zero-source deployment. A publishable contribution must
therefore center on control-specific behavior, generalization, and efficiency or
training integration rather than on the classical pairing operation.

## Retrieval provenance

- Accessed: 2026-08-09
- arXiv endpoint: `https://export.arxiv.org/api/query`
- Query: `all:antithetic AND (all:diffusion OR all:"flow matching")`
- Parameters: `start=0`, `max_results=20`, `sortBy=relevance`
- arXiv reported 19 matches; this paper was the only direct learned-generative-
  model antithetic-noise result near the top.
- A second arXiv query for `all:antithetic` combined with robot, diffusion
  policy, flow policy, or robot control terms reported two unrelated matches.
- OpenAlex searches were also run for antithetic sampling with diffusion/flow
  matching and for diffusion/flow policies with ensemble/symmetry terms.
- Semantic Scholar search was attempted twice without an API key and returned
  HTTP 429 both times; its coverage is not represented in this provisional
  summary.

