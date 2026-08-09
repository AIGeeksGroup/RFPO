# Outer Loop Cycle 53: Dense-Reward Post-FPO Refinement

## Failure Boundary

H62-H64 show that exact replay, common-scene groups, and deterministic references cannot make sparse
Square terminal success into a sufficiently dense action-learning signal. The failure is not missing
class balance: H64's zero references split 4/4 success/failure, yet agreed with the modal Gaussian
outcome in every scene and left only eight nonzero weights. Another binary-baseline variant would
spend rollouts without changing this boundary.

Go2 provides a distinct, already reproduced regime: dense return, 31 checkpoints along one official
FPO++ run, and 4,096-environment evaluation. The best zero-source checkpoint is iteration 1450 while
the best random-source checkpoint is the final iteration, so late-training checkpoint variance is a
measured optimization target rather than a speculative source of gain.

## Diverged Candidates

1. Uniform actor averaging over the final five official Go2 checkpoints.
2. Uniform averaging of only checkpoints 1450 and 1499.
3. Return-selected soup over the best zero and random checkpoints.
4. Exponential moving average reconstructed across all 31 saved checkpoints.
5. Action-output ensembling of the final five actors.
6. Post-RL reflow distillation of the final Go2 flow policy.
7. Fixed-subspace mirrored parameter search using dense Go2 return.
8. Low-rank actor adapter optimized by evolution strategies.
9. Midpoint-sampler Go2 finetuning after H60's equal-NFE inference result.
10. Actor-only continuation from checkpoint 1450 with a lower learning rate.
11. Critic reset and short continuation from checkpoint 1450.
12. Observation-normalizer averaging across late checkpoints.

## Convergence

Select candidate 1 as H65. It is coefficient-free, uses a window fixed by checkpoint cadence rather
than reward selection, adds no training or environment interaction to construct the candidate, and
can be rejected with one paired 256-environment screen. Izmailov et al. (2018) motivate averaging
nearby points on one optimization trajectory. A pre-protocol read-only geometry check found actor
cosines of `0.9978-0.9997` to the final checkpoint, a `3.51%` averaged relative displacement, and
bitwise-identical observation-normalizer statistics across the window.

H44 does not settle this test: it averaged only four updates from a short Square run and was judged by
sparse binary success. H65 asks whether late official-scale FPO++ iterates in a dense-return regime
occupy a useful common basin. Candidate 2 is a reward-informed window choice, candidate 3 directly
selects on evaluation reward, candidate 4 introduces a decay/window convention, and candidate 5
multiplies inference cost. Reflow distillation is retained as an efficiency direction but lacks a
reward mechanism after H60. Parameter search and new training are parked behind the zero-training
average; normalizer averaging is inactive because the five stored normalizers are identical.

Two-sentence pitch: late FPO++ Go2 checkpoints trade a small amount of deterministic and stochastic
return despite lying in a tight EMA-actor neighborhood. Uniformly averaging a fixed tail may suppress
optimizer-path variance and improve both-mode mean return without changing training, sampling, or
inference cost.
