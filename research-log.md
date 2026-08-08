# Research Log

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-08 | bootstrap | Selected the official amazon-far/fpo-control release as the reproduction authority. Locked Go2 return and Can zero-sampling success as primary representative metrics before any method change. |
| 2 | 2026-08-08 | bootstrap | Identified conditional reflow as the first improvement hypothesis. It must first reduce measured path curvature and preserve low-step return before testing online regularization. |
| 3 | 2026-08-08 | infrastructure | Manipulation environment completed on server 30061. Verified PyTorch 2.7.1+cu126, CUDA availability, LeRobot and Tyro imports, and `eval_checkpoint.py --help`. Tyro accepts both underscore and kebab-case options used by the official examples. |
| 4 | 2026-08-08 | incident | Isaac setup was interrupted while downloading a 2.45 GB wheel because the login-shell proxy at 127.0.0.1:7890 temporarily stopped listening. Git and pip had no persistent proxy setting. Restarted the idempotent installer after connectivity recovered; pip reused the completed wheel cache. |
