# Analysis: Rank-Preserving Advantage Weight Audit

## Result

H22 is refuted. The formal audit had 124 eligible fresh-rollout chunks, finite nonzero reference
gradients, centered candidate weights with both signs, and exact GAE order preservation. The rank
transform nevertheless reduced gradient alignment in both locked batches.

| Batch | Control cosine | Rank cosine | Gain | Gate |
|---:|---:|---:|---:|---|
| 0 | 0.85488 | 0.81527 | -0.03961 | fail |
| 1 | 0.88728 | 0.61801 | -0.26927 | fail |

The control-to-candidate Spearman correlation was numerically 1.0. Candidate weights had mean below
`1e-8` in magnitude and standard deviation 0.58203. Reference gradient norms were 4.04304 and
2.63091. Iteration-2 collection succeeded on 3/8 episodes, so outcome labels were non-degenerate.

## Interpretation

GAE magnitude information is useful on this rollout even when its absolute calibration is
imperfect. Replacing magnitudes with empirical percentiles preserves ordering but discards enough
information to rotate the FPO++ actor gradient away from observed outcomes. The positive 16-chunk
smoke did not survive the preregistered 64-chunk seed.

Stop without online rank weighting, another seed, or tuning the rank formula or scope. Combined with
H21, this result supports retaining the official GAE weights. The next intervention should constrain
the parameter displacement after the already well-aligned pre-update gradient rather than change
the advantage estimator.

## Artifacts

- `results/rank_advantage_audit.json`
- `~/workspace/outputs/fpo-control/results/can_step6000_rankadv_audit_osmesa_formal_seed20260820/rank_advantage_audit.json`
- `~/workspace/outputs/fpo-control/logs/can_step6000_rankadv_audit_osmesa_formal_seed20260820.log`
