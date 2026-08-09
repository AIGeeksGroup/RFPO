# H73 Result: Independent Equal-Total-NFE Confirmation

## Outcome

H73 passed every primary gate. Six new 512-environment cells completed on the
fixed H69 checkpoints, adding `antithetic32` and `zero32` to archived paired
`random64` and `zero64` results.

| Training seed | Random64 | Antithetic32 | Equal-NFE gain | Zero32 | Anti32 - zero32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 40.557 | 41.459 | +0.903 | 41.424 | +0.035 |
| 44 | 40.301 | 41.308 | +1.008 | 41.354 | -0.046 |
| 45 | 40.718 | 41.605 | +0.888 | 41.672 | -0.067 |

The stratified `antithetic32 - random64` estimate was +0.9326 with bootstrap
95% CI [+0.7851, +1.0850]. Every training-seed point estimate was positive,
and random64 and antithetic32 each used 64 total NFE/action. Initial-observation
and primary-source hashes matched exactly within each seed.

The separate method-specific deployment gate failed. `Antithetic32 - zero32`
pooled to -0.0258 with CI [-0.1597, +0.1096]. `Zero32 - zero64` was -0.0467
with CI [-0.1079, +0.0002], providing no evidence that 32 deterministic steps
materially improve or degrade the 64-step control across these policies.

## Decision

Support an independent-policy, equal-total-NFE improvement over the official
Gaussian-random evaluation mode. Preserve zero32 as the strongest deployment
control and do not claim antithetic-specific latency or deterministic-policy
superiority. Proceed to the preregistered medium Spot cross-task screen.
