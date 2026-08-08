# Behavior-Preserving Online RL Fine-Tuning

## Scope

This note records the literature used to choose the post-H27 direction. The observed FPO++ failure
is rapid rotation of an initially useful policy gradient during repeated updates, while simple
learning-rate reduction and post-hoc weight interpolation have already failed.

## Relevant Methods

### Gradient Surgery for Multi-Task Learning

Yu et al. (2020), *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020,
arXiv:2001.06782. PCGrad detects a negative dot product between two task gradients and removes the
component of one gradient that conflicts with the other. The useful property here is the
parameter-free geometric rule: preserve the RL direction when it does not oppose a behavior
objective and intervene only on measured conflict.

Source provenance: OpenAlex work `W2997359900`, queried 2026-08-09; arXiv PDF listed by OpenAlex.

### Kickstarting Deep Reinforcement Learning

Schmitt et al. (2018), *Kickstarting Deep Reinforcement Learning*, arXiv:1803.03835. Kickstarting
uses an auxiliary distillation loss from a fixed teacher while a student learns from RL. It supports
the use of a pretrained policy as a behavior anchor, but its scalar auxiliary weight introduces a
tuning burden. H28 borrows the fixed-teacher objective but replaces weighted loss addition with
conflict-triggered projection.

Source provenance: arXiv title query and paper metadata checked 2026-08-09.

### Adaptive Behavior Cloning Regularization

Zhao et al. (2022), *Adaptive Behavior Cloning Regularization for Stable Offline-to-Online
Reinforcement Learning*, ESANN 2022, DOI:10.14428/esann/2022.es2022-110. This work directly supports
behavior cloning regularization as a way to reduce destructive policy drift during online
fine-tuning. Its actor objective differs from an implicit flow policy, so it motivates the problem
rather than supplying a drop-in algorithm.

Source provenance: OpenAlex title search, queried 2026-08-09.

### Rectified Flow

Liu, Gong, and Liu (2022), *Flow Straight and Fast: Learning to Generate and Transfer Data with
Rectified Flow*, arXiv:2209.03003. The user-proposed paper learns straight ODE paths and recursively
rectifies a coupling. In this project, conditional reflow reduced path curvature by about 22%, but
did not improve official-scale Can reward. It remains a supported sampling-efficiency result, not the
next reward-improvement mechanism.

Source provenance: arXiv export API entry for 2209.03003, queried 2026-08-09.

## Decision

Use a fixed initial-policy velocity-field target on rollout states as the behavior objective. Project
the FPO++ gradient only when its dot product with the behavior-anchor gradient is negative. Audit the
two gradients and one-step effects before integrating the method into reward training. This is
distinct from H23 because it changes direction rather than only step size, and from H25 because the
constraint acts during optimization in function space rather than interpolating final parameters.

