# Su et al. (2026): Residual Flow Steering

- Title: *RFS: Reinforcement Learning with Residual Flow Steering for Dexterous Manipulation*
- Authors: Entong Su, Tyler Westenbroek, Anusha Nagabandi, Abhishek Gupta
- arXiv: `2602.01789v3`
- Project page: <https://weirdlabuw.github.io/rfs/>
- Accessed: 2026-08-09

## Method

RFS freezes a pretrained flow-matching policy and trains a separate modulation policy with PPO.
The modulation policy jointly outputs the initial flow latent `a0` and an additive residual action
`ar`. The latent branch provides global mode movement while the residual branch provides local
off-manifold correction. The executed action is `Des(s, a0, v_theta) + ar`.

The simulation policy and value networks use three hidden layers of widths 256, 128, and 64 with
ReLU activations. Reported PPO settings are discount 0.99, GAE lambda 0.95, policy LR `3e-4`,
value LR `1e-3`, clip 0.2, value coefficient 0.5, gradient norm 1.0, and minibatch size 1024.

## Relevance to FPO++

This is materially different from the closed H1-H47 directions: it does not update the flow field
and does not use a CFM surrogate probability ratio. PPO acts on the explicit joint distribution over
source latents and residuals. This directly tests whether restricting online adaptation to a compact
modulation layer preserves the pretrained flow policy while improving reward-relevant exploration.

The paper does not provide an RFS repository link. Its project-page code link resolves to the generic
GitHub homepage as of the access date. H48 therefore implements the paper-level method locally and
labels task/environment differences explicitly rather than claiming exact RFS reproduction.

## Retrieval Provenance

- arXiv identifier lookup: `https://export.arxiv.org/api/query?id_list=2602.01789`
- Full text: `https://arxiv.org/pdf/2602.01789`
- Project page: `https://weirdlabuw.github.io/rfs/`
