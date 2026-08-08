# Can FPO++ One-Update Smoke

Date: 2026-08-08

Configuration: official Can FPO++ hyperparameters, 30 environments, 1600 collection steps, seed 0, one 48,000-step update, local released checkpoint.

| Metric | Value |
|---|---:|
| Collection successes | 13/151 (8.61%) |
| Collection throughput | 459.2 local SPS |
| Policy loss | approximately 0.0000 |
| Value loss | 0.0014 |
| Post-update zero evaluation | 20/20 |
| Post-update random evaluation | 4/20 |

The run produced step-48000, latest, and best checkpoints. The 20-episode evaluations verify execution only and are not used as evidence of an improvement.
