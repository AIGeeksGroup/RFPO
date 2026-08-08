# Protocol: Can Step-6000 Initialization Audit

## Claim

The `95j3noe4_step_6000` Can checkpoint is the high-quality base policy described in Appendix D.4,
with substantially denser Gaussian-source successes than the official main-benchmark
`95j3noe4_step_1000` initialization.

This experiment characterizes initialization quality only. Any difference from `step_1000` is not
an FPO++ algorithm improvement.

## Locked Evaluation

- Server GPU: physical GPU 1
- Checkpoint: local `downloaded_checkpoints/95j3noe4_step_6000`
- Weights: EMA
- Environment: Can, 10 Euler sampling steps, 10 parallel environments
- Evaluation seed: 20260811
- Primary cell: 50 Gaussian-random-source episodes
- Diagnostic cell: 50 zero-source episodes
- No training, checkpoint selection, or evaluation-seed retry

## Primary Gate

Treat the initialization hypothesis as supported if random-source evaluation obtains at least
20/50 successes (40%). This is deliberately below the paper's 64.06% point estimate but clearly
separates the checkpoint from the reproduced `step_1000` rate of 47/400 (11.75%). If the gate
fails, do not fine-tune from this checkpoint or schedule a larger evaluation.

If the gate passes, use `step_6000` only as a higher-signal screening environment for paired,
short-budget method comparisons. The official `step_1000` benchmark remains the final target for
any claimed algorithmic gain.
