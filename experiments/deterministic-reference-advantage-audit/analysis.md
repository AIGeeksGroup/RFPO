# Analysis: Deterministic-Reference Episodic Advantage Audit

## Outcome

H64 is refuted. The formal audit passed every validity gate, but the deterministic-reference signal
failed three of four advantage-support gates. No fixed-batch actor-gradient audit, actor integration,
reference variant, or reward training is authorized.

All eight zero-source episodes completed within 400 steps and contained four successes and four
failures. Their initial processed-observation and privileged-state hashes exactly matched all 32
archived H63 Gaussian trajectories. All 18,032 supplied source values were bitwise zero, first action
chunks were finite, the archived record checksum was exact, and policy parameters remained bitwise
unchanged.

## Locked Gates

| Gate | Observed | Requirement | Result |
|---|---:|---:|---|
| Zero-reference successes | 4 / 8 | at least 2 | Pass |
| Zero-reference failures | 4 / 8 | at least 2 | Pass |
| Positive advantages | 3 / 32 | at least 4 | Fail |
| Negative advantages | 5 / 32 | at least 4 | Pass |
| Nonzero advantages | 8 / 32 | at least 20 | Fail |
| Absolute-weight ESS | 8.0 | at least 15 | Fail |

The two-scene smoke separately passed every infrastructure and validity gate. Both zero references
and all eight archived Gaussian trajectories failed, so smoke advantages were correctly all zero;
formal signal gates did not apply.

## Interpretation

The deterministic reference has balanced marginal outcomes but closely tracks the modal Gaussian
outcome within each scene. Seeds `20261080`, `20261081`, `20261083`, and `20261085` agree across the
reference and every Gaussian trajectory. Seeds `20261082` and `20261084` each expose only one
Gaussian regression, seed `20261086` exposes three Gaussian rescues, and seed `20261087` exposes
three regressions. Consequently, reference subtraction leaves only eight nonzero trajectories and
ESS 8, reducing rather than improving H63's 16 nonzero trajectories and ESS 12.

The result strengthens the scene-difficulty finding from H63: a deterministic policy outcome is a
strong scene-level predictor, but that makes it an overly aggressive baseline for sparse binary
success. H64 does not provide enough signed support for a stable FPO++ actor update. Per protocol,
additional references, smoothing, partial pooling, new seeds, and training are closed.

## Artifacts

- `results/audit/config.json`
- `results/audit/records.json`
- `results/audit/results.json`
- `logs/audit.log`
- `results/smoke/config.json`
- `results/smoke/records.json`
- `results/smoke/results.json`
- `logs/smoke.log`

Local and remote SHA-256 checksums match for all eight generated artifacts. Physical GPU 3 returned
to 4 MiB after the audit.
