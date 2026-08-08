# Research Log

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-08 | bootstrap | Selected the official amazon-far/fpo-control release as the reproduction authority. Locked Go2 return and Can zero-sampling success as primary representative metrics before any method change. |
| 2 | 2026-08-08 | bootstrap | Identified conditional reflow as the first improvement hypothesis. It must first reduce measured path curvature and preserve low-step return before testing online regularization. |
| 3 | 2026-08-08 | infrastructure | Manipulation environment completed on server 30061. Verified PyTorch 2.7.1+cu126, CUDA availability, LeRobot and Tyro imports, and `eval_checkpoint.py --help`. Tyro accepts both underscore and kebab-case options used by the official examples. |
| 4 | 2026-08-08 | incident | Isaac setup was interrupted while downloading a 2.45 GB wheel because the login-shell proxy at 127.0.0.1:7890 temporarily stopped listening. Git and pip had no persistent proxy setting. Restarted the idempotent installer after connectivity recovered; pip reused the completed wheel cache. |
| 5 | 2026-08-08 | incident | The vendored robosuite 1.5.1 tree omitted all 30 upstream texture files, preventing PickPlaceCan construction. Added a pinned asset fetch to setup and verified the vendored tree differs from upstream only in intentional robot additions and two robot XML files. |
| 6 | 2026-08-08 | incident | `eval_checkpoint.py --save-video False` crashed by popping an empty frame list and did not count episode steps. Moved step counting outside the video branch and guarded frame removal; the 2+2 episode smoke then passed. |
| 7 | 2026-08-08 | result | Can checkpoint validation over 20 episodes per mode produced 80% zero-sampling success and 10% random-sampling success. This agrees with the reported base-policy regime (73.76% zero sampling and approximately 10% random sampling), but is not the final baseline estimate. |
