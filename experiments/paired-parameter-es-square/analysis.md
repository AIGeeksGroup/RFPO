# H53 Paired Parameter-Space ES Signal Audit

## Validity

The formal audit loaded the released Square `trc7rbt0_step_110000` EMA actor and completed all 256
locked episodes: 16 Gaussian parameter directions, both mirrored signs, and eight constructor-seeded
Square environments per sign. The four locked `model.mlp.{0,2,4,6}.weight` displacements matched the
1% per-tensor norm within tolerance in every rollout. All unlisted parameters remained bitwise
unchanged during inference, every anchor restore was bitwise exact, and all recorded values were
finite.

Because this repository's RoboSuite wrapper ignores `reset(seed=...)`, each sign used a newly created
vector environment with the same constructor seeds. Initial image-and-state observation SHA-256
hashes matched exactly between signs for every direction and seed. The median first normalized
action-chunk half paired RMS was 0.02024, with direction values from 0.01667 to 0.03675, inside the
locked `[0.01, 0.10]` behavioral-activity interval.

## Signal Results

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| nonzero directions, replica A | 11/16 | at least 6 | pass |
| nonzero directions, replica B | 9/16 | at least 6 | pass |
| A/B Pearson correlation | -0.4996 | at least 0.35 | fail |
| common nonzero directions | 6 | at least 5 | pass |
| common-nonzero sign agreement | 1/6 (16.7%) | at least 70% | fail |
| A top-4 mean on B | -0.125 | at least 0.125 | fail |
| B top-4 mean on A | -0.250 | at least 0.125 | fail |

Replica success-rate differences (`positive - negative`) were:

- A: `[-.25, .25, .25, .25, 0, -.50, 0, .25, 0, 0, .25, 0, -.25, -.25, -.50, .25]`
- B: `[0, -.25, 0, -.25, 0, 0, -.25, 0, 0, .25, 0, -.25, .25, .50, .50, .25]`

The negative cross-replica correlation and reversed top-direction transfer show that complete-episode
binary success did produce finite differences, but those differences did not reproducibly rank this
fixed parameter subspace. Aggregate successes were 81/128 under negative perturbations and 83/128
under positive perturbations; this pooled two-success difference is not a directional learning
result.

## Decision

H53 is refuted. Do not implement an ES update and do not tune the perturbation scale, layers, basis,
direction count, seeds, checkpoint, or reward. This closes fixed-subspace mirrored parameter ES at
the preregistered budget; it does not claim that arbitrary large-scale black-box optimization is
impossible.

This was an OSMesa paired method audit, not an official EGL benchmark or real-robot result. Raw
records are in `raw/audit_results.json`; `raw/audit.log` contains the complete remote execution log.
