# Protocol: H35 Terminal-Consistency GAE Filter

## Hypothesis H35

Official GAE is strongly but imperfectly aligned with observed Can outcomes. For chunks whose rollout
contains an observed terminal, zeroing only normalized GAE weights whose sign disagrees with the
centered discounted terminal return should remove misleading first-step contributions while
retaining GAE magnitudes, correctly signed failures, and correctly signed successes.

The candidate has no threshold or mixing coefficient. In a later online implementation, chunks
without an observed terminal retain their official GAE weight; this audit uses labeled chunks only so
that its claimed mechanism can be judged directly.

## Fixed Audit

- Checkpoint: released Can `95j3noe4_step_6000` EMA actor under OSMesa.
- Seed: 20260905.
- Collection: 16 environments, 320 steps, one critic-only warmup plus a fresh iteration-2 Gaussian
  rollout using official settings.
- Select 128 seeded, fully valid chunks with observed terminal labels and split them into two
  independent 64-chunk replicas.
- Within each replica, normalize GAE using the official non-DDP formula.
- Control weights: normalized GAE.
- Outcome reference weights: centered discounted returns to the observed terminal.
- Candidate weights: control weights where control and reference have the same nonzero sign, zero
  otherwise. Do not renormalize after filtering.
- Compute control, candidate, and reference FPO++ gradients with identical stored MC8 variables.

## Gates

All gates must pass in both replicas:

1. Control, candidate, and reference gradients are finite and nonzero, and the candidate retains at
   least eight positive and eight negative weights.
2. Candidate cosine to the outcome-reference gradient is at least 0.75 and at least 0.10 higher than
   control.
3. Candidate cosine to the control gradient is at least 0.75.
4. Candidate norm is between 50% and 120% of control norm.
5. The filter retains between 40% and 90% of chunks, proving that it is active without collapsing to
   a small selected subset.

Stop without online integration, return thresholds, soft masks, recentering, or mixing coefficients
if any gate fails. A full pass authorizes a separately committed matched short-training protocol.
