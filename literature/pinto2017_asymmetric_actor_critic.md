# Pinto et al. (2017): Asymmetric Actor Critic for Image-Based Robot Learning

- Authors: Lerrel Pinto, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, Pieter Abbeel
- arXiv: `1710.06542v1`
- URL: https://arxiv.org/abs/1710.06542
- Accessed: 2026-08-09

## Key Idea

The actor receives only the partial observation available at deployment, while the critic receives
the simulator's full state during training. The paper reports that this asymmetry improves visual
robot-policy learning without requiring privileged inputs at evaluation. This targets representation
and credit-assignment difficulty without changing the environment reward.

## Relevance

Released FPO++ freezes the Square policy's visual encoder and trains its scalar critic on the same
high-dimensional encoded visual conditioning. H19 showed that changing the critic target alone
improved calibration but degraded return ranking. A low-dimensional simulator-state critic changes
the information available to value learning while preserving the original sparse reward and the
RGB-plus-proprioception actor contract. It is also a prerequisite audit for heavier Q-gradient or
RLDT methods, whose policy updates are only as trustworthy as their learned critic.

## Reproducible Retrieval

- arXiv endpoint: `https://export.arxiv.org/api/query?id_list=1710.06542`
- PDF: `https://arxiv.org/pdf/1710.06542v1`

