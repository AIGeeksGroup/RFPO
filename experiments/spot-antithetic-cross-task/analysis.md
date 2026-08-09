# H74 Result: Official Spot Cross-Task Screen

H74 failed its preregistered transfer gate on the official seed-42 Spot policy.
The fixed `model_1499.pt` was healthy: `zero64` averaged 331.814, above the
locked 250-return threshold, and all 1,280 episodes, action values, initial-
observation hashes, source-stream hashes, seeds, and NFE checks were valid.

| Method | Mean return | Total NFE/action |
|---|---:|---:|
| `zero64` | 331.814 | 64 |
| `zero32` | 331.704 | 32 |
| `random64` | 314.002 | 64 |
| `iid_pair32` | 325.112 | 64 |
| `antithetic32` | 327.783 | 64 |

At equal total NFE, antithetic projection improved the official random-source
baseline by +13.780 return, with paired-bootstrap 95% interval [5.576, 22.373].
The point estimate also exceeded an IID two-endpoint average by +2.671, but its
interval [-5.717, 10.664] crossed zero. The required antithetic-specific
cross-task attribution therefore did not pass.

The stronger deployment comparison was negative. `Antithetic32 - zero32` was
-3.921 with interval [-8.600, -0.423], while `zero32 - zero64` was -0.110 with
interval [-3.730, 2.813]. A deterministic half-depth solve remains the stronger
Spot deployment baseline.

The locked protocol stops Spot and blocks H1/G1 expansion after this result. It
supports transfer of stochastic-source deficit recovery from Go2 to Spot, but
does not establish transfer of the symmetry-specific advantage over generic
two-sample averaging. The remote and local corrected analyses are byte-
identical with SHA-256
`5cff55dd788cfee1086785118967e92ddcba7332661a9b25de2a77bcb7329a48`.
