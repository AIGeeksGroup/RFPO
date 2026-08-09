# Lei et al. (2026): Reinforcement Learning for Flow-Matching Policies with Density Transport

- Authors: Boshu Lei, Kostas Daniilidis, Antonio Loquercio
- arXiv: `2606.08602v1`
- URL: https://arxiv.org/abs/2606.08602
- Accessed: 2026-08-09

## Method

RLDT treats policy improvement as action-density transport. For each state it samples `K=8` flow
actions and uses an RBF-kernel SVGD field containing both critic-gradient attraction and particle
repulsion. An expected-target approximation maps every intermediate flow state to the action
manifold, allowing the transport field to supervise the velocity network without backpropagating
through the complete ODE. The actor loss combines transport alignment, straight-flow consistency,
and a Fisher-divergence constraint to a reference policy.

## Robomimic Setting

- Square action chunk: 4; flow steps: 8.
- Actor update: learning rate `2e-5`, 32 gradient steps per iteration.
- Critic: trainable ViT encoder plus a double-Q MLP ensemble, learning rate `1e-3`.
- Training: 200 iterations, 64 parallel environments, 200 steps per iteration, 512 Q updates per
  iteration, and two critic-only warmup iterations.
- Sparse rewards use balanced sampling and Q targets clipped to `[0, 1]`.
- Reported cost: about 30 GPU hours on one NVIDIA A40.

## Relevance and Availability Audit

RLDT directly addresses the value-aware exploration gap left by the failed curvature and residual
modulation experiments. It also requires a substantially stronger critic pipeline than released
FPO++, so critic validity is a necessary mechanism gate before a local reimplementation.

As of the access date, the project URL declared in the paper,
https://rpfey.github.io/rldt/, returns HTTP 404. The author's public GitHub repository list contains
`fpo-control` but no RLDT or density-transport repository, and an exact-title GitHub repository
search returned zero results. The arXiv record contains one version and no code URL. Therefore no
official implementation is currently available for a faithful low-cost port.

## Reproducible Retrieval

- arXiv endpoint: `https://export.arxiv.org/api/query?id_list=2606.08602`
- GitHub user endpoint: `https://api.github.com/users/RPFey/repos?per_page=100`
- GitHub repository query: exact paper title, total count zero
- Project-page response: HTTP 404 on 2026-08-09

