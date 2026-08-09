# Analysis: H50 Asymmetric Square Critic Audit

## Validity

The locked collection completed 32 terminal Square episodes from seeds `20260926` through
`20260957`, with exactly one episode per seed. It produced 704 finite plan records, 8 successes, and
24 failures, exactly meeting the minimum outcome-support gate. The actor remained frozen and used
the original RGB-plus-proprioception inputs, Gaussian source, Euler-10 sampler, 16-action execution,
and sparse binary reward.

The raw `records.pt` artifact remains on the server at
`~/workspace/outputs/fpo-control/results/square_h50_asymmetric_critic_osmesa_seed20260926/records.pt`
(29 MB, SHA-256 `4cdd2e6677f1a7999ee8e0e43c1e209512f8427b5dcbacb82beff50d98310157`).
The compact result, config, and complete run log are archived locally under `results/`.

## Cross-Fit Result

| Held-out fold | Visual MSE | Privileged MSE | Relative change | Visual Spearman | Privileged Spearman | Gain |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.03703 | 0.04206 | +13.58% | 0.3770 | 0.4077 | +0.0307 |
| 1 | 0.04139 | 0.04569 | +10.39% | 0.2674 | 0.2334 | -0.0340 |

The candidate fails every improvement gate. Its held-out MSE is worse in both folds rather than at
least 10% better. Rank correlation improves only slightly in fold 0 and worsens in fold 1, rather
than gaining at least 0.10 in both.

## Decision

H50 is refuted. Do not integrate the privileged critic into FPO++, run an actor-gradient audit, or
change its state vector, network, optimizer, loss, split, seed, collection size, or gate. The result
does not refute RLDT itself; it rejects the proposed cheap prerequisite that simulator state alone
makes value ranking reliably better under the official short critic budget. A full RLDT port remains
parked while no official implementation is public.

