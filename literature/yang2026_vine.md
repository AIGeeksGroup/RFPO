# VINE: Taming Generative Control Policies for Reinforcement Learning

- Authors: Rushuai Yang, Zhuo Han, Houlin Li, Hecheng Wang, Zhichao Wu, Rui Zhang, Zhaowei Zhang, Zihong Chen, Xiaohan Yan, Chiming Liu, Yi Chen, Wei Shan, Maoqing Yao
- Year: 2026
- arXiv: `2607.10369v1`
- URL: https://arxiv.org/abs/2607.10369
- Retrieved: 2026-08-09 through the arXiv API and PDF endpoint

## Method

VINE is a value-gradient actor-critic method, not a value-free credit-assignment method. At denoising step `k`, it reconstructs a fresh interpolation state around the current endpoint estimate,

`x_hat_k = t_k * a_hat_k + (1 - t_k) * z_k`,

then predicts the next endpoint as

`a_hat_{k+1} = x_hat_k + (1 - t_k) * v_theta(x_hat_k, t_k; s)`.

Fresh Gaussian noise is injected at every step, and the critic action gradient is backpropagated through the complete iterative sampler. The paper trains a twin action-value critic with replay and a behavior-regularized actor loss. It reports strong OGBench and real-world insertion results and attributes the gain to smaller, more stable BPTT gradients than vanilla Euler sampling.

## Relevance to FPO++

The sampler formula is directly compatible with a flow-matching velocity field, but the optimization method is not a local replacement for FPO++: the released FPO++ manipulation code has a state-value critic rather than VINE's twin action-value critic, and prior H19/H21/H50 audits found weak short-budget critic and action-effect evidence. Direct VINE training would simultaneously add a new critic, replay system, behavior regularizer, stochastic sampler, and BPTT actor objective, preventing a clean small-scale test. Park direct implementation until an action-value prerequisite passes.

## Retrieval Provenance

- Metadata query: `GET https://export.arxiv.org/api/query` with `search_query=all:"VINE" AND (all:"credit assignment" OR all:"reinforcement learning")`, `start=0`, `max_results=10`, sorted by relevance.
- Full text: `https://arxiv.org/pdf/2607.10369`.
- The query returned five records; the title and identifier above uniquely matched generative robot-control policy optimization.
