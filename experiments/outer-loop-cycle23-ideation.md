# Outer Loop Cycle 23: Multi-Epoch Update-Path Stabilization

## Constraint From Prior Results

H21, H22, and H35 show that the official initial GAE signal is already informative. H18 shows severe
held-out gradient rotation after one epoch, while H23 and H24 show that lower learning rate and full-
batch accumulation do not fix it. The next method must modify the repeated update path while leaving
the on-policy gradient, reward, source distribution, and advantage weights unchanged.

## Shortlist

| Rank | Candidate | Update-path mechanism | Decision |
|---:|---|---|---|
| 1 | PPO-RB ratio rollback | reverses improving out-of-bound ratio gradients | Select H36 |
| 2 | Truly PPO KL rollback | penalizes policy KL beyond a trust region | Park; full flow-policy KL is not tractable |
| 3 | KL early stopping | stops epochs after a divergence threshold | Reject; repeats H18's too-late sensor |
| 4 | Mirror-descent proximal update | constrains repeated policy displacement | Park; larger implementation and tuning surface |
| 5 | Extragradient update | anticipates next-gradient rotation | Park; doubles actor passes and adds step-size choices |

## Selection

H36 implements the ratio-rollback component of Truly PPO directly on FPO++'s existing CFM ratios.
It is exactly equal to official clipping at the behavior policy and differs only on improving ratios
outside the existing `epsilon=0.01` bound. The coefficient `alpha=0.3` is fixed from the paper's
continuous-control experiments. A paired step-6000 audit can reject the mechanism before online
reward evaluation if rollback merely suppresses surrogate progress or fails to preserve the held-out
gradient direction.
