# Self-Imitation Learning

Oh, Guo, Singh, and Lee (2018), *Self-Imitation Learning*, arXiv:1806.05635.

The paper adds an off-policy actor-critic objective that reproduces an agent's past high-return
decisions and reports improvements for A2C on sparse-exploration Atari tasks and PPO on MuJoCo.
It motivates reward-conditioned reuse in principle, but it does not justify an H47-style auxiliary
here: the completed H15 audit found cosine 0.04877 between success-only replay and same-rollout
positive-advantage FPO++ gradients, indicating severe selection bias. A self-imitation CFM loss is
therefore rejected rather than promoted as the next experiment.

Provenance: OpenAlex work `W2804380964`, exact-title search via `/works` with
`search.exact=Self-Imitation Learning`, accessed 2026-08-09. OpenAlex reports arXiv DOI
`10.48550/arXiv.1806.05635`.
