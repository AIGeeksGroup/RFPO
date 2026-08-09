# Analysis: Group-Relative Complete-Episode Signal Audit

## Outcome

H63 is refuted. The formal audit was valid, but common-scene leave-one-out terminal success did not
meet three of four locked signal gates. No actor integration, group-size change, seed change, or
additional rollout block is authorized.

All 32 episodes completed within the 400-step horizon. Repeated replicas had exact initial processed
observation and privileged-state hashes within every group. The aggregate Gaussian source had mean
`0.001054` and standard deviation `0.996772`; first action chunks had median within-group pairwise
normalized RMS `0.426432`. Leave-one-out advantages summed to zero within `1.12e-16`, and all policy
parameters remained bitwise unchanged.

## Locked Gates

| Gate | Observed | Requirement | Result |
|---|---:|---:|---|
| Success support | 14 / 32 | at least 6 | Pass |
| Failure support | 18 / 32 | at least 6 | Pass |
| Mixed groups | 4 / 8 | at least 5 | Fail |
| Nonzero advantages | 16 / 32 | at least 20 | Fail |
| Absolute-weight ESS | 12.0 | at least 15 | Fail |

The smoke separately passed every infrastructure gate on two groups of two replicas. One smoke group
was mixed, producing the expected `[+1, -1]` advantages; formal signal gates correctly did not apply
to the smoke population.

## Interpretation

Full trajectories are more outcome-sensitive than H62's local vines: 14 successes and 18 failures
provide ample global class support. Common-scene grouping reveals a different limitation. Seeds
`20261080`, `20261081`, and `20261085` failed in every replica, while seed `20261083` succeeded in
every replica. Their 16 trajectories receive zero group-relative advantage despite containing four
successful episodes. Only four scene groups were mixed, leaving 16 nonzero trajectories and ESS 12.

Thus, initial-scene difficulty is real, but conditioning the baseline on an exact scene discards too
much of this small Square rollout budget. Increasing group size or adding groups could manufacture
more effective samples at substantially higher cost, contrary to the locked prerequisite. H63 does
not justify replacing official GAE or implementing a group-relative FPO++ update.

## Artifacts

- `results/audit/config.json`
- `results/audit/records.json`
- `results/audit/results.json`
- `logs/audit.log`
- `results/smoke/config.json`
- `results/smoke/records.json`
- `results/smoke/results.json`
- `logs/smoke.log`

Local and remote SHA-256 checksums match for all eight artifacts. Physical GPU 3 returned to 4 MiB
after the audit.

