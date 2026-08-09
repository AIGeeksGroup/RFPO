# Protocol: H46 Curvature-Selected Gaussian Best-of-Two

## Hypothesis

For two Gaussian action chunks sampled by the same Square flow policy at the same observation, the
chunk whose conditional ODE path is straighter is a more self-consistent policy sample and improves
closed-loop random-source success. Selecting within an observation avoids globally changing policy
weights, the ODE solver, the source marginal used to generate candidates, or zero-source behavior.

## Locked Method

- Checkpoint: frozen H43 official-control Square checkpoint
  `square_h43_control_osmesa_seed20260916/checkpoints/step_25600`.
- Control: one standard-Gaussian source and the unchanged Euler-10 sampler.
- Candidate: exactly two independent standard-Gaussian sources per environment whenever a new chunk
  is required; integrate both with the unchanged Euler-10 sampler.
- Score: existing normalized `straightness_error`, computed separately per candidate over the 16
  actions actually executed. The source is the normalized Gaussian initial state and the path is the
  ten stored post-Euler states, including the endpoint.
- Selection: per environment, execute the complete candidate with lower score; ties select the first
  candidate. No action averaging, source scaling, threshold, reward model, or state carried between
  replans.
- Zero-source inference must use the existing path without drawing or ranking candidates.

## Pipeline Smoke

Before reward evaluation, use seed `20260921`, eight Square environments, and enough steps to record
at least 32 replanning decisions. Record both scores and selected branch for every decision.

The smoke passes only if:

1. all candidate actions, paths, and scores are finite;
2. every selected score equals the per-environment minimum of the two candidate scores;
3. both candidate branches are selected at least once;
4. exactly two candidates and ten Euler velocity evaluations per candidate are used at each replan;
5. the existing zero-source and ordinary random paths remain unchanged when the new mode is disabled.

Smoke reward is diagnostic and cannot change the formal seed, checkpoint, method, or gate.

## Balanced Reward Screen

- Seed: `20260922`, not used by prior Square evaluations.
- Conditions: ordinary one-source random control, then curvature-selected best-of-two candidate.
- Each condition uses 20 environments, one completed episode per environment, 16 executed actions,
  Euler-10, no EMA, OSMesa, and the same frozen checkpoint.
- Primary metric: successful episodes out of 20.

H46 passes only if the candidate exceeds control random-source success by at least 3/20. Zero-source
success is analytically identical because the mode is disabled on that path; therefore a +3 random
gain is also a +3 pooled gain against the same zero cell. A pass authorizes a separately locked
independent confirmation, not an official benchmark claim.

If the smoke or reward gate fails, stop without another seed, best-of-three or larger candidate sets,
path-length or endpoint-residual scoring, score thresholds, action averaging, checkpoint selection,
training, or confirmation. OSMesa remains a paired screen only.

