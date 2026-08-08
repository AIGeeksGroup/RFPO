# Outer Loop Cycle 20: First-Step Signal Candidates

## Constraint From H31-H33

The next method must change the first actor-step learning signal. Temporal source coherence harmed
reward, advantage-sign projection was inactive, and finer clipping was indistinguishable from control
before ratios reached a boundary.

## Shortlist

| Rank | Candidate | First-step effect | Distinct from closed work | Fast audit | Decision |
|---:|---|---:|---:|---:|---|
| 1 | Coordinate-wise median microbatch gradient | direct | yes | outcome-gradient cosine | Select H34 |
| 2 | Coordinate sign consensus mask | direct | yes | outcome-gradient cosine | Park; threshold required |
| 3 | Leave-environment-out baseline | direct | partly | variance audit | Park; weak state matching |
| 4 | Extragradient actor update | direct | yes | virtual two-step audit | Park; doubles forward/backward cost |
| 5 | State-uncertainty trust weight | direct | no | critic calibration | Reject; returns to failed critic route |
| 6 | Larger clipping-aware virtual step | delayed | no | H33 variant | Reject by H33 stopping rule |

## Selection

H34 is problem-first: H24 rejects ordinary mean accumulation, but does not test whether a minority of
microbatches contributes coordinate outliers that dominate that mean. Coordinate median aggregation
is coefficient-free, acts before parameter drift, and can be judged against observed terminal returns
without an online run. Its main risk is destructive coordinate mixing, addressed by explicit
candidate/control direction and norm-retention gates.
