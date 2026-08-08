# Truly Proximal Policy Optimization

- Authors: Yuhui Wang, Hao He, Chao Wen, and Xiaoyang Tan
- Year: 2020 (arXiv v2; originally posted 2019)
- arXiv: https://arxiv.org/abs/1903.07940
- DOI: https://doi.org/10.48550/arXiv.1903.07940
- OpenAlex work: `W2921474818`
- Retrieved: 2026-08-09 through OpenAlex and the arXiv PDF

## Mechanism

Ordinary PPO flattens a sample's clipped objective after an improving likelihood ratio crosses its
clip boundary. Other samples and optimizer momentum can still push that ratio farther outward. PPO
with rollback (PPO-RB) replaces the flat branch with

```text
F_RB(r; epsilon, alpha) =
  -alpha * r + (1 + alpha) * (1 - epsilon),  r <= 1 - epsilon
  r,                                             otherwise
  -alpha * r + (1 + alpha) * (1 + epsilon),  r >= 1 + epsilon
```

inside the same lower-bound objective `min(r*A, F_RB(r)*A)`. The slope on an improving, out-of-range
sample is therefore reversed instead of set to zero. The paper uses `alpha=0.3` for PPO-RB on all
continuous-control tasks except Humanoid (`0.02`).

The complete Truly PPO method instead triggers a KL rollback outside a trust region. That version is
not selected here because FPO++ has an implicit flow policy and its released objective exposes a
Monte Carlo CFM likelihood ratio, not a tractable full action-distribution KL.

## Relevance To FPO++

FPO++ already applies PPO clipping to its CFM-derived ratios. Our H18 audit showed that after ten
optimizer epochs the independent positive-ratio active fraction falls to `0.734`, while held-out
gradient cosine to the pre-update direction falls to `0.209` and `0.080` in two batches. PPO-RB is a
direct update-path intervention: it preserves the official on-policy gradient at ratio one, then adds
inward pressure only after an improving sample crosses the existing FPO++ bound. It is distinct from
H18 early stopping because its correction acts within the first epoch, and from H33 temporal
clipping because it changes the out-of-bound gradient rather than the ratio granularity.

## Evidence Boundary

The paper's reported control gains do not establish gains for CFM pseudo-likelihood ratios or robot
manipulation. H36 therefore uses the paper-default coefficient without a sweep and requires a fixed
mechanism audit before any reward run.
