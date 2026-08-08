# Post-Reflow Candidate Selection

Pure reflow reliably straightened the flow but improved official-scale Can zero success by only 1.5
percentage points. Candidate generation used failure analysis, composition/decomposition, and the
simplicity filter.

| Candidate | Mechanism | Decision |
|---|---|---|
| Mixed dataset/reflow endpoints | Restore real-action grounding while retaining straighter teacher couplings | Test first |
| EMA/L2-SP anchoring | Limit drift from the released checkpoint | Reject: preserves behavior but has no clear source of reward gain |
| Second-stage recursive reflow | Further reduce curvature | Reject: geometry is no longer the reward bottleneck |
| Lower-timestep loss weighting | Favor low-NFE integration accuracy | Reject: endpoint error is already improved without reward transfer |
| Test-time endpoint averaging | Reduce source variance | Reject: zero sampling is deterministic and already the primary score |
| Best-of-N action selection | Select higher-value flow samples | Park: requires a calibrated critic not present in the BC checkpoint |
| Reward-filtered teacher endpoints | Distill only successful actions | Park: needs new rollout labeling and risks selection bias |
| Success-classifier guidance | Guide endpoints toward task success | Park: adds a new model and a heavier data protocol |
| Online reflow auxiliary loss | Couple straightening to reward optimization | Stop: official-scale pure-reflow reward gate failed |
| WarmPrior plus reflow | Combine temporal source and straight endpoints | Reject: WarmPrior failed decisively in both prior screens |

Winner: a single 50/50 mixed-endpoint pilot. It adds one scalar and uses the existing teacher, dataset,
optimizer, and evaluation pipeline. The strongest objection is that conflicting couplings may remove
the straightness benefit; the geometry gate tests that before any rollout.

