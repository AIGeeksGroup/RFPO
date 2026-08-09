# H77 Result: Mirrored Closed-Loop Rollout Variance

H77 failed its preregistered closed-loop variance gates on the fixed official
seed-42 Go2 policy. All 1,024 episodes completed, actions and returns were
finite, all initial-observation and primary base-source hashes matched, and the
negative source was an exact sign reversal.

Mean return was 41.447 for zero32, 40.546 for the primary `z` trajectory,
40.451 for its mirrored `-z` trajectory, and 40.452 for the independent `w`
trajectory. The matched source-induced return deviations remained positively
correlated after feedback: correlation was 0.479 for `(z,-z)` and 0.450 for
`(z,w)`. Mirrored covariance was slightly lower (6.255 versus 6.791), passing
only the weakest point-estimate gate.

The zero-residualized pair-return variance was 9.676 for the mirrored pair and
10.967 for the IID pair, a ratio of 0.882. This 11.8% point reduction missed the
locked 20% threshold, and its paired-bootstrap 95% interval [0.414, 1.475]
crossed one widely. Immediate action antisymmetry therefore does not provide a
stable closed-loop return control variate at this scale.

Per protocol, mirrored-rollout gradient audits and training are stopped without
source scaling, another seed, checkpoint, step count, task, or gate tuning. The
remote and local analyses are byte-identical with SHA-256
`67bbb3e65c0c7c06dd1469ae52bdceca5dd3e64809b0e95e9504f12b959bebd1`.
