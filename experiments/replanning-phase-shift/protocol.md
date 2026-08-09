# H55 Protocol: Can Replanning Phase-Shift Prerequisite Audit

## Hypothesis

Released Can step-1000 deterministic success depends on where the official eight-step replanning
boundaries land relative to task progress. Changing only the first executed prefix should alter
outcomes for the same initial states, creating measurable headroom for an adaptive BCP-style
continuation policy.

## Locked Intervention

- Checkpoint: official released Can step-1000 EMA policy.
- Source and solver: zero source, 10 Euler integration steps.
- Prediction horizon: 16 actions.
- Conditions: first prediction executes `phi` actions for every integer `phi` in `[1, 8]`; all later
  predictions execute the official eight actions.
- Control: `phi=8`.
- No action averaging, source change, extra prediction, altered checkpoint, or training.
- The first-prefix option is evaluation-only, disabled by default, and reset independently at every
  episode boundary.
- All eight phases are part of one prerequisite audit, not a phase-tuning result. The per-seed best
  phase is a hindsight oracle and cannot be reported as a deployable policy.

## Common-Seed Audit

- renderer: OSMesa mechanism screen, not official EGL or real robot
- base seed: `20261010`
- environment seeds: `20261010` through `20261029`
- 20 environments, one completed episode per environment and phase
- total budget: 160 completed episodes
- phase order: `8, 1, 2, 3, 4, 5, 6, 7`, fixing the official control first
- video and W&B: disabled

The implementation must save the initial normalized observation SHA-256 and binary outcome for every
`(phase, seed)` cell. All eight initial hashes for a seed must match exactly, every environment must
contribute one completed finite episode, and the control must reproduce at least 12/20 successes.

H55 passes the prerequisite only if:

1. At least 4/20 seeds have nonconstant success across the eight phases.
2. The hindsight phase oracle succeeds on at least three more seeds than `phi=8`.
3. No single noncontrol phase is declared a method improvement; only the oracle gap authorizes the
   next research stage.

If any validity or signal gate fails, refute H55 and do not implement BCP, try more seeds, change the
phase set, or add a heuristic trigger. If all gates pass, archive the matrix and separately freeze a
small continuation-head training protocol before adding any optimizer or rollout code.
