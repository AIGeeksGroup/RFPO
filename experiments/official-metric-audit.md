# Official Manipulation Metric Audit

## Question

Does the released manipulation benchmark optimize only deterministic zero-source success, making
the random-source non-degradation gates in earlier screens unnecessarily strict?

## Evidence

- Figure 4 in the paper explicitly has two rows: zero-sampling success in the first row and standard
  Gaussian random-sampling success in the second row.
- `manipulation_experiments/docs/reproduce.md` describes the main-benchmark plot as all five tasks and
  four methods with zero and random sampling side by side.
- `launch_main_benchmark.sh` passes `--zero_sampling=True`, but the flag is deprecated for evaluation.
  `eval_all_ranks` always runs both source modes and logs
  `eval/success_rate_zero_sampling` and `eval/success_rate_random_sampling`.
- Every reported checkpoint evaluation uses 200 episodes per source mode in the official command.
- The convenience `best` checkpoint is selected only by zero-source success. Figure 4, however, plots
  both complete metric histories rather than reporting only that selected checkpoint.

## Decision

Random-source success is part of the official manipulation benchmark, not merely an appendix
diagnostic. Earlier paired gates that reject a deterministic gain accompanied by a stochastic loss
remain aligned with the paper's main comparison. H43, H44, and H25 therefore stay closed; their
zero-source gains cannot be reclassified as benchmark improvements.

