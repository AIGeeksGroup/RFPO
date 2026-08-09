# Can FPO++ First-Iteration Smoke

Date: 2026-08-08

Configuration: official Can FPO++ hyperparameters, 30 environments, 1600 collection steps, seed 0,
one 48,000-step iteration, and the local released checkpoint loaded with EMA weights.

| Metric | Value |
|---|---:|
| Collection successes | 13/151 (8.61%) |
| Collection throughput | 459.2 local SPS |
| Policy loss | approximately 0.0000 (actor not trained during critic-only warmup) |
| Value loss | 0.0014 |
| Post-update zero evaluation | 20/20 |
| Post-update random evaluation | 4/20 |

The run produced step-48000, latest, and best checkpoints. The released loop makes iteration 1 a
critic-only warmup, so this run did not perform an actor update. A retrospective tensor audit on
2026-08-09 loaded the exact released step-1000 EMA actor and compared all 35 saved state tensors with
`step_48000`; every tensor was bitwise identical (L2 distance and maximum absolute difference both
zero). The 20/20 zero-source score is therefore evaluation variance, not a finetuning gain. These
20-episode evaluations verify execution only.
