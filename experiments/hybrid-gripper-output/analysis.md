# H58 Random Arm / Deterministic Gripper Output Analysis

## Outcome

H58 is refuted. The hybrid candidate scored 1/20, tying the locked full-Gaussian control and missing
the required 4/20. It rescued one control failure but lost the control's only success, so deterministic
gripper substitution exchanges failure modes rather than improving random-source success. No
confirmation, averaging, agreement selection, or separate gripper head is authorized.

## Stage A: Paired Smoke

The two-seed candidate completed finite episodes and matched the archived H57 control's initial
normalized observations and 76 common Gaussian arm-source hashes exactly. Across all 76 candidate
replans, the six hybrid arm coordinates were bitwise equal to the Gaussian prediction and the hybrid
gripper was bitwise equal to the zero-source prediction. All records were active, and the maximum
first-action change was 0.0962. Every implementation gate passed.

## Stage B: Reward Screen

The candidate reused H57's checkpoint, seeds `20261031..20261050`, stateless source seed `20261101`,
10-step Euler solver, and 16/8 horizons. It performed two flow solves per replan.

| Condition | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| Archived full-Gaussian control | 1 | 20 | 5% |
| Gaussian arm / deterministic gripper output | 1 | 20 | 5% |

All 750 common active source keys had exact arm hashes. The candidate recorded 760 active replans;
every arm and gripper coordinate invariant was exact. Control succeeded only on seed `20261041`;
candidate failed there and instead succeeded only on `20261046`. Therefore the tie is not an inactive
intervention artifact.

## Interpretation

H57's two sub-threshold rescues cannot be explained by a better deterministic gripper decision alone.
Replacing the gripper trajectory changes which sparse case succeeds but does not preserve the useful
Gaussian trajectory. Together with H56, this closes fixed gripper output projection and substitution.
Further work should target action-candidate quality rather than gripper-specific post-processing.

## Artifacts

- `raw/smoke/audit_results.json` and `audit.log`: two-seed candidate smoke
- `raw/audit/audit_results.json` and `audit.log`: 20-episode candidate screen
- H57 raw audit: archived matched control and stateless source records

These are OSMesa method screens, not official EGL benchmark or real-robot results.
