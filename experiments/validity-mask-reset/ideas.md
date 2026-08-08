# Reward-Learning Candidate Screen

## Observed Bottleneck

Can random-sampling success is only about 10%, and increasing CFM Monte Carlo samples did not help.
The next intervention should preserve or improve the scarce successful on-policy signal instead of
spending more compute on the same samples.

## Candidates

| Candidate | Mechanism | Decision |
|---|---|---|
| Reset the CFM invalid-step mask before each rollout | Prevent stale episode-boundary positions from permanently removing unrelated future samples | Test first: concrete implementation defect, one-variable fix |
| Preserve `done` across action-chunk boundaries | Prevent GAE leakage when termination occurs on a chunk's final action | Defer: requires separating GAE and within-chunk validity semantics |
| Full-return GAE (`gamma=lambda=1`) | Propagate terminal success through the full episode | Defer: higher variance and changes two coupled coefficients |
| Global rather than minibatch advantage normalization | Keep sparse advantages on a consistent scale | Defer until validity corruption is removed |
| Stratified successful-trajectory minibatches | Ensure every minibatch sees positive examples | Defer: more implementation surface and sampling bias risk |
| Positive-advantage replay | Reuse successful chunks more often | Reject for now: no longer strictly on-policy without correction |
| Potential-based dense reward | Improve manipulation credit assignment | Defer: task-specific reward engineering changes benchmark semantics |
| Success-conditioned BC auxiliary loss | Anchor updates to successful sampled actions | Defer: requires a new loss weight and careful ratio interaction |
| Critic pretraining beyond one iteration | Improve early advantage estimates | Defer: delays actor learning and does not create reward information |
| Larger collection batch | Observe more successful episodes per update | Reject for now: increases environment cost without fixing discarded data |

The mask-reset candidate passes the simplicity test: it changes no reward, policy, optimizer, or
evaluation behavior and is directly predicted to keep the usable CFM fraction stationary across
iterations.

