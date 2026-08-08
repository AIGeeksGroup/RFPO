# Protocol: H3a Exploratory WP-Past Warm Start on Can

## Status and Scope

This is an exploratory compatibility test, not a faithful WarmPrior reproduction. The released Can
checkpoint was pretrained with a standard Gaussian source, whereas WarmPrior trains with the warm
source from the beginning.

## Intervention

Use the released `95j3noe4_step_1000` EMA checkpoint. At each chunk boundary, replace the first
eight normalized source actions with the previously executed eight-action chunk plus Gaussian noise
of scale `sigma=0.5`. Keep the remaining eight prediction positions Gaussian. The first chunk of
each episode falls back to the standard source.

## Comparison

- Task: Robomimic Can
- Checkpoint: released step 1000, EMA weights
- Euler steps: 10
- Seed: 20260808
- Environments: 10
- Episodes: 20 per condition
- Conditions: Gaussian vs. previous-action source, each under zero and random sampling

## Prediction and Gate

The warm source may increase random-sampling success by keeping exploration near a temporally
plausible action sequence. Continue to a 200-episode check only if random success improves over the
matched Gaussian run and deterministic success remains at least 50%. Stop this no-retraining branch
if either criterion fails.

## Metrics

Primary: pooled random-sampling success rate. Secondary: pooled zero-sampling success rate and
rollout throughput. Raw summaries and logs are retained under the remote runtime directory.
