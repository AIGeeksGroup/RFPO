# H68 Analysis: Cross-Checkpoint Attribution

H68 passed all preregistered gates. At checkpoint 500, antithetic pairing beat
the equal-128-NFE IID pair by `+0.4371`; at checkpoint 1000 the gain was
`+0.7412`. Pooling the 256 new paired differences gave `+0.5891` with bootstrap
95% interval `[+0.0223, +1.0377]`. H67's archived checkpoint-1499 gain was
`+0.7793 [0.4913, 1.2095]`. The antithetic-over-IID direction is therefore
positive at all three tested stages of the seed-42 trajectory.

The deterministic comparison changes with policy maturity. At checkpoint 500,
zero/antithetic mean returns were `39.3909/37.1937`, so antithetic remained
`-2.1972` below zero. At checkpoint 1000 they were `41.4011/41.3534`, a much
smaller but precisely measured `-0.0477`. At checkpoint 1499, H67 found a
suggestive `+0.2547` antithetic point estimate whose interval crossed zero.
This supports a source-symmetry mechanism that increasingly recovers central
policy behavior during training; it does not yet prove a universal improvement
over deterministic deployment.

All 768 new episodes completed with exact initial-observation pairing, exact
primary-source pairing between IID and antithetic methods, finite actions, and
128 NFE/action for both candidates. Local replay reproduced the remote analysis
exactly. Independent official training seeds are now authorized.

