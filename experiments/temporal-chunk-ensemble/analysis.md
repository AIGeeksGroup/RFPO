# H54 ACT-Inspired Temporal Chunk Ensemble Analysis

## Outcome

H54 is refuted as a Can reward-improvement method. The locked temporal ensemble was active, finite,
and substantially reduced chunk-boundary discontinuity, but the 20-episode reward screen tied the
official control exactly at 16/20 successes. The `+2/20` gate failed, so no independent confirmation,
weight variant, cadence variant, random-source evaluation, or official EGL benchmark is authorized.

## Stage A: Mechanism Audit

The two-environment Can smoke at seed `20261001` produced 76 replan records, 73 of which blended an
old continuation with a new prefix. Every first chunk was bitwise unmodified, all values were finite,
the maximum weighted-average reconstruction error was 0, and the median feedback fraction was
`0.49750003`. Median boundary jump fell from the same-trajectory raw-new counterfactual of
`0.14365196` to `0.08741090`, a 39.15% reduction. All locked mechanism gates passed.

## Stage B: Reward Screen

Both conditions used the official Can step-1000 EMA checkpoint, zero source, 10 Euler steps, four
environments, five completed episodes per environment, and deterministic seeds `20261001 + env_id`.

| Condition | Successes | Episodes | Success rate | FPS |
|---|---:|---:|---:|---:|
| Official control | 16 | 20 | 80% | 38.1 |
| Temporal ensemble | 16 | 20 | 80% | 36.8 |

The complete candidate screen recorded 603 replans, including 578 blended replans. Every invariant
remained valid. Median boundary jump decreased from `0.18113513` to `0.11987275`, a 33.82% reduction,
while feedback fraction remained `0.49750003`. Therefore the reward tie is not an inactive-method or
numerical-validity artifact.

## Interpretation

Reusing the discarded continuation provides a clean continuity mechanism without extra network
calls, but smoother chunk boundaries do not improve released Can deterministic success at this
budget. This agrees with H31 and H27: changing temporal execution can materially alter action paths,
yet continuity or more frequent feedback alone is not the limiting factor. H54 closes fixed overlap
averaging and nearby weights; a learned gate would require a separate predictive signal and is not
justified by this result.

## Artifacts

- `raw/smoke/temporal_chunk_ensemble.json`: Stage A records and aggregate audit
- `raw/control/`: official 20-episode summary and log
- `raw/candidate/temporal_chunk_ensemble.json`: 603-record full candidate audit
- `raw/candidate/`: candidate summary and log

These are OSMesa paired method screens, not official EGL benchmark or real-robot results.
