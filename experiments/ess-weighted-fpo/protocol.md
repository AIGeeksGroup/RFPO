# Protocol: ESS-Weighted Mirror FPO++

## Hypothesis

Can's few successful standard-Gaussian trajectories are diluted by signed, minibatch-normalized
advantages. Replacing the signed advantage in FPO++ with normalized exponential mirror-descent
weights will concentrate the same clipped per-sample ratio update on high-advantage on-policy action
chunks and improve early full-noise success without requiring guided-source rollouts.

## Method

Add an opt-in `ess_softmax` advantage-weighting mode. After the existing distributed advantage
normalization, compute detached weights `softmax(A / tau) * N`. Select `tau` by deterministic
bisection so the weights have a target effective sample size of 50% of the minibatch. Use these
positive weights in the existing per-sample PPO-clipped ratio objective. Defaults must preserve the
official signed-advantage behavior.

Run one Can seed-0 pilot from the released step-1000 EMA checkpoint for five 48k-step updates. Keep
the official standard-Gaussian source, PPO trust region, 8 CFM samples, lambda 0.99, cumulative-mask
behavior, learning rates, and all other parameters unchanged.

## Gates

The primary metric is pooled collection success over actor-update iterations 2-5. It must reach at
least 10.77%, two percentage points above the same-code official control's 53/604 (8.77%). Final
50-episode zero success may not be more than five points below 84%, final random success must be at
least 10%, all losses and gradient norms must remain finite, and observed weight ESS must remain
within two percentage points of the 50% target.

Stop after this seed if any gate fails. If all gates pass, run one matched new-seed control and
candidate before any longer training. Do not sweep ESS fraction, temperature, clipping, source
distribution, or auxiliary losses in this pilot.
