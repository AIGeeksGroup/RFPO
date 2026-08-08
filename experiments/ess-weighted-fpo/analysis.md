# Analysis: ESS-Weighted Mirror FPO++

## Result

H11 is refuted. The implementation behaved as preregistered: every logged minibatch reached exactly
50.00% effective sample size, with automatically selected temperatures between 1.75 and 4.30.
Losses and gradient updates remained finite.

| Iteration | Full-noise success | Valid CFM | Logged temperature | Logged ESS |
|---:|---:|---:|---:|---:|
| 1 | 16/152 (10.53%) | 98.01% | 2.9158 | 50.00% |
| 2 | 8/151 (5.30%) | 97.36% | 3.0000 | 50.00% |
| 3 | 18/152 (11.84%) | 96.58% | 4.1680 | 50.00% |
| 4 | 18/153 (11.76%) | 95.91% | 4.3040 | 50.00% |
| 5 | 8/151 (5.30%) | 95.65% | 1.7516 | 50.00% |

The confirmatory iterations 2-5 pooled to 52/607 (8.57%), below the same-code official control's
53/604 (8.77%; two-sided Fisher `p=0.919`) and far below the required 10.77% gate. Final 50-episode
evaluation was 41/50 (82%) for zero sampling and 8/50 (16%) for random sampling. Both evaluation
non-degradation gates passed, but they cannot override failure of the locked primary metric.

## Interpretation

Replacing signed advantages with positive mirror-descent weights does not make rare on-policy Can
successes more useful over this short budget. The objective remained stable and retained random
evaluation behavior, but the large iteration-to-iteration collection variance was centered at the
same success rate as the official update. Reward weighting alone is therefore not the missing
mechanism identified by H8-H10.

Stop without a second seed, longer training, or an ESS/temperature/clipping sweep. Retain the
opt-in implementation for reproducibility, with the official signed objective as the default.
