# Outer-Loop Cycle 28: Reward-Relevant Square Behavior

## Reflection

H41 repeats H30b's endpoint-anchor pattern: deterministic behavior is preserved, but random-source
success rises by only one episode. The intervention acts on optimizer conflict yet does not alter how
the closed-loop controller gathers new evidence. The next candidate should directly affect
reward-relevant execution or exploration, remain coefficient-light, and be rejectable without a
large training run.

The key newly exposed boundary is task-specific action execution. The released Square base
checkpoint executes 8 actions, while the official FPO++ command overrides this to 16. Can's prior
8-to-4 screen failed, but restoring Square from 16 to its pretrained 8-step execution horizon is a
different, task-grounded intervention: it doubles visual feedback while remaining inside the action
chunk on which the policy was pretrained.

## Candidates

| Rank | Candidate | Lens and rationale | Decision |
|---:|---|---|---|
| 1 | Restore Square execution horizon 16 to 8 | boundary probe; direct checkpoint-native feedback change, no training | Select H42 |
| 2 | Critic-guided best-of-N action chunks | composition; directly rerank exploration, but current critic is state-only | Park pending an action-value audit |
| 3 | Outcome-predictor action reranking | decomposition; learn reward relevance explicitly, but censored short rollouts complicate labels | Park |
| 4 | Exact CNF likelihood ratios | method boundary; improve ratio fidelity, but second-order divergence cost is high | Park |
| 5 | Layerwise trust-region clipping | optimizer boundary; targets localized drift, but requires a new threshold | Park |
| 6 | Extragradient FPO++ | control analogy; anticipates post-step rotation, but H24 weakens update-path-only premises | Park |
| 7 | State-conditional source scale | adaptive exploration; directly changes behavior but reopens the closed source-scaling family | Reject |
| 8 | Zero/Gaussian source curriculum | composition; H8-H10 already reject guided-to-Gaussian transfer | Reject |
| 9 | Successful self-imitation | replay analogy; H15 found stale success gradients clipped and strongly biased | Reject |
| 10 | kNN action entropy | diversity tension; Appendix D.5 reports entropy preservation hurts pretrained manipulation | Reject |
| 11 | Endpoint-PCGrad coefficient or layer variants | local refinement; forbidden by H41 and lacks a larger signal | Reject |
| 12 | Longer H41 training or checkpoint selection | scale probe; spends more compute after the primary gate failed | Reject |

## Selection

Use the frozen H41 control checkpoint to compare official 16-step execution against an 8-step
override under balanced zero and Gaussian-source evaluation. This isolates feedback frequency from
training and model weights. A positive screen would justify confirmation and later retraining with
the restored horizon; a negative result closes the Square horizon shortcut without a training run.
