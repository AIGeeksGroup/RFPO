# Analysis: Mixed-Source Rollout Distillation for Can

## Result

The 20% zero-source mixture generated substantially more reward information, but did not improve the
preregistered Gaussian-source collection metric. Over actor-update iterations 2 through 5, the
Gaussian subset achieved 34/480 (7.08%), below the matched control's 53/604 (8.77%).

| Iteration | Zero-source success | Gaussian-source success | Overall success |
|---:|---:|---:|---:|
| 1 (critic only) | 31/42 (73.81%) | 13/121 (10.74%) | 44/163 (26.99%) |
| 2 | 34/44 (77.27%) | 6/120 (5.00%) | 40/164 (24.39%) |
| 3 | 39/45 (86.67%) | 7/120 (5.83%) | 46/165 (27.88%) |
| 4 | 40/46 (86.96%) | 7/120 (5.83%) | 47/166 (28.31%) |
| 5 | 20/37 (54.05%) | 14/120 (11.67%) | 34/157 (21.66%) |
| Iterations 2-5 pooled | 133/172 (77.33%) | 34/480 (7.08%) | 167/652 (25.61%) |

The mechanism gate passed: the mixture collected 167 successes over iterations 2 through 5, 3.15
times the control's 53. Final evaluation was 42/50 (84%) zero and 6/50 (12%) random, versus 42/50
(84%) and 5/50 (10%) for the control. Losses remained finite. However, overall mixed-source success
was diagnostic only; the Gaussian-subset primary gate failed by 3.69 points relative to its required
10.77% threshold.

## Interpretation

Successful zero-source trajectories can be incorporated without collapsing either final evaluation
mode, but a delta source at zero has too little overlap with the full Gaussian behavior distribution
to improve its online collection rate in this pilot. Do not run another seed or sweep the zero-source
fraction. A future source intervention must retain continuous overlap with the Gaussian source and be
evaluated separately on the official full-noise distribution.

## Artifacts

- `~/workspace/outputs/fpo-control/results/can_mixedsource20_smoke_seed0`
- `~/workspace/outputs/fpo-control/results/can_mixedsource20_5iter_seed0`
- `~/workspace/outputs/fpo-control/logs/can_mixedsource20_smoke_seed0.log`
- `~/workspace/outputs/fpo-control/logs/can_mixedsource20_5iter_seed0.log`
