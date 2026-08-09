# Outer Loop Cycle 40: Is Replanning Timing Reward-Relevant?

## Failure Boundary

H54 proves that the old and new overlapping chunks contain different actions and that their average
reduces boundary jump, but continuity does not improve Can reward. H27 separately shows that globally
doubling visual replanning frequency is harmful. These results do not establish that every fixed
eight-step boundary is well placed: a fresh prediction immediately before gripper closure or object
release may help even when uniformly shorter chunks do not.

## Diverged Candidates

1. BCP-style learned Bernoulli continuation head over candidate execution horizons.
2. Phase-shift audit of the fixed eight-step schedule before training any head.
3. Action-disagreement threshold for early replanning.
4. Gripper-action threshold for early replanning.
5. Simulator-contact trigger for early replanning.
6. Learned task-phase classifier with supervised stage labels.
7. GRPO categorical head over unordered horizons.
8. Per-environment bandit over fixed execution phases.
9. Replan only before predicted gripper sign changes.
10. Train a low-level corrective residual only near contact.

## Convergence

Select H55, the phase-shift prerequisite audit from BCP. It changes only how many actions are executed
from the first prediction, then returns to the official eight-step cadence. Evaluating all phases on
the same environment seeds gives a per-seed hindsight oracle and directly asks whether replanning
timing, rather than global frequency or smoothing, changes task outcome. This is necessary before
building the BCP continuation-head pipeline, whose published recipe uses 256 environments and eight
A100 GPUs.

Reject direct BCP training until the oracle gap exists. Reject hand-designed disagreement, gripper,
or contact thresholds because they introduce uncalibrated signals and, for simulator contact, a
privileged deployment input. Reject supervised phases because no ground-truth stopping label exists.
A categorical or bandit policy discards BCP's ordered prefix structure, while a contact residual
returns to the explicit modulation route closed by H48-H49.

Two-sentence pitch: fixed action horizons can fail because a replan lands after, rather than before,
a precision-critical event. H55 measures this latent headroom with common-random phase shifts before
spending compute on a learned continuation policy.
