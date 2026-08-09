# H70 Result: Equal-Total-NFE Antithetic Projection

## Outcome

H70 is supported. All four methods completed 256 finite paired episodes with
identical initial observations. The stochastic methods used the same primary
Gaussian source stream, and `random64` and `antithetic32` each used exactly 64
velocity-field evaluations per action.

| Method | Total NFE/action | Mean return |
| --- | ---: | ---: |
| zero64 | 64 | 41.5765 |
| random64 | 64 | 40.5906 |
| antithetic32 | 64 | 41.5720 |
| antithetic64 | 128 | 41.5694 |

At equal total compute, `antithetic32 - random64` was +0.9813 return (paired
SEM 0.0140; bootstrap 95% CI [+0.9542, +1.0085]). Halving each endpoint solve
from 64 to 32 steps retained the full antithetic result:
`antithetic32 - antithetic64` was +0.0026, with 95% CI [-0.0023, +0.0080].

The separate deployment gate did not pass. `Antithetic32 - zero64` was
-0.0045, with 95% CI [-0.0264, +0.0204]. The supported claim is therefore that
antithetic projection recovers the zero-source return while preserving the
Gaussian source marginal and using the same total NFE as one official random
sample. It does not improve over zero-source deployment on this checkpoint.

## Decision

Confirm equal-total-NFE antithetic projection on the independently trained H69
policies if H69 first passes its equal-compute antithetic-versus-IID gate. Do
not screen additional step counts or source scales on seed 42.
