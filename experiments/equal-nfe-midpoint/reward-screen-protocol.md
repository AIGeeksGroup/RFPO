# H38b Balanced Can Reward Screen

## Hypothesis

The endpoint-fidelity gain from midpoint-5 improves closed-loop Can success relative to official
Euler-10 at the same ten velocity-network evaluations, without trading deterministic and stochastic
source performance against each other.

## Locked Comparison

- Checkpoint: released Can `95j3noe4_step_1000`, EMA weights; no training or checkpoint changes.
- Control: Euler integration, 10 steps, 10 NFE.
- Candidate: explicit midpoint integration, 5 steps, 10 NFE.
- Source modes: `zero` and `random`, evaluated separately.
- Environment seed and policy/source seed: `20260910` for every condition.
- Accounting: 20 parallel environments, exactly one completed episode from each environment in each
  condition (`20` episodes per method and source mode; `80` episodes total).
- Rendering: paired isolated OSMesa environment used by the recent small Can screens. This is a
  screening backend; any larger confirmation requires healthy EGL.
- No videos, W&B logging, source-prior changes, action-step changes, or policy retraining.

Run the Euler control first, followed by midpoint, with zero then random source mode inside each
method. Preserve each raw log and output directory.

## Gates

The screen is valid only if the Euler control scores between `10/20` and `18/20` under zero sampling
and between `0/20` and `6/20` under random sampling. These wide sanity bounds cover the reproduced
step-1000 regime without selecting a favorable baseline.

The candidate passes only if all conditions hold:

1. Midpoint gains at least three successes across the pooled zero/random total (`+3/40`).
2. Midpoint loses at most one success in either individual source mode.
3. All 20 environments contribute exactly one completed episode in every condition.
4. Evaluation completes without non-finite actions or runtime errors.

If any gate fails, stop H38b without another seed, solver, step count, schedule, ensemble, or training
run. If all gates pass, separately preregister an independent healthy-EGL confirmation before running
it; the screen alone is not benchmark-improvement evidence.
