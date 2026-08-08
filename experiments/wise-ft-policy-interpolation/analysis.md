# H25 BC/FPO++ Weight-Space Interpolation Screen

## Result

H25 is refuted. The fixed `alpha=0.5` midpoint partially recovered deterministic performance but
substantially degraded stochastic-source success.

| Policy | Zero success | Random success | Mean |
|---|---:|---:|---:|
| Step-6000 actor anchor | 20/20 (100%) | 15/20 (75%) | 87.5% |
| Four-update FPO++ endpoint | 18/20 (90%) | 15/20 (75%) | 82.5% |
| WiSE-FT midpoint | 19/20 (95%) | 12/20 (60%) | 77.5% |

The midpoint improved zero-sampling by 5 points over the finetuned endpoint, below the locked
10-point gate. Random success was 15 points below both endpoints, failing the retention and
anchor-improvement gates, and its mean score was 5 points below the finetuned endpoint. No
100-episode confirmation or interpolation-coefficient sweep was run.

The paired endpoints also show that these four FPO++ actor updates did not improve random success on
seed 20260823: both scored 15/20, while zero success fell by 10 points. There is no demonstrated
online gain for interpolation to preserve in this run.

## Checkpoint Integrity

All 35 floating tensors were interpolated. The anchor-to-endpoint L2 distance was `0.5473601`; the
anchor-to-midpoint distance was `0.2736800`, or `0.5000000` relative distance. Full provenance is in
`interpolation_manifest.json`.

## Artifacts

- `interpolation_manifest.json`
- `anchor_{zero,random}_seed20260823.txt`
- `finetuned_{zero,random}_seed20260823.txt`
- `midpoint_{zero,random}_seed20260823.txt`
- `screen_seed20260823.log`

The remote evaluation completed normally. Physical GPU 1 returned to 0% utilization and 5 MiB.
