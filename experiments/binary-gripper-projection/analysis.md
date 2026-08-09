# H56 Binary Gripper-Support Projection Analysis

## Outcome

H56 is refuted. Projecting every executed gripper command to `{-1, +1}` was valid and active, but the
candidate scored 15/20, exactly tying the locked H55 official control. All 20 per-seed outcomes were
identical between conditions. The 17/20 screening gate failed, so no independent confirmation,
threshold, hysteresis, hold-time, source, or training variant is authorized.

## Stage A: Implementation Smoke

The two-seed smoke completed at 2/2 success. Its initial normalized-observation hashes exactly matched
the archived control, the six arm coordinates were preserved bitwise, and every projected gripper
value was exactly `-1` or `+1`. Of 301 executed actions, 195 (64.78%) changed the gripper command by
more than 0.05. All locked implementation gates passed.

## Stage B: Common-Seed Reward Screen

Both conditions used the official Can step-1000 EMA checkpoint, zero source, 10 Euler steps, 16/8
prediction/execution horizons, and environment seeds `20261010..20261029`.

| Condition | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| Official control | 15 | 20 | 75% |
| Binary gripper projection | 15 | 20 | 75% |

The candidate completed 4,136 active executed actions. It materially changed 2,794 gripper commands
(67.55%), with mean absolute change 0.1841 and maximum change 0.9983. Raw gripper outputs ranged from
-1.4092 to +1.4613. The arm coordinates remained bitwise exact, projected support was exact, and all
initial hashes matched control. Nevertheless, every seed retained its original success or failure.

## Interpretation

The continuous gripper endpoint visibly differs from the nominal two-state command support, but that
difference is behaviorally irrelevant to binary Can success for these common seeds. The simulator
controller and task tolerate the intermediate command magnitudes well enough that hard projection
does not address the limiting failures. H56 closes fixed binary projection and its nearby
threshold/hysteresis variants; future candidates should target a different failure mechanism rather
than add complexity to gripper post-processing.

## Artifacts

- `raw/smoke/audit_results.json` and `audit.log`: two-seed implementation smoke
- `raw/audit/audit_results.json` and `audit.log`: complete 20-seed candidate screen
- H55 `raw/audit_results.json`: locked official control and common-seed hashes

These are OSMesa method screens, not official EGL benchmark or real-robot results.
