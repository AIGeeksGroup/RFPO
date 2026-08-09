# Research Findings

## Research Question

Can the official FPO++ robot-control results be reproduced, and can conditional reflow make the learned policy flow straighter enough to improve return, robustness, or sampling efficiency?

## Current Understanding

FPO++ stabilizes likelihood-free policy gradients through per-CFM-sample ratios and an asymmetric trust region for from-scratch locomotion. Its policy still uses an Euler-integrated conditional flow with up to 64 steps. Rectified Flow suggests that regenerating paired endpoints from an existing flow and fitting those pairs can straighten trajectories, but it does not imply a reward improvement by itself. The first defensible target is therefore lower curvature and preserved task return at fewer sampling steps; online return improvements are a second-stage hypothesis.

## Key Results

The released Can step-1000 checkpoint reproduced its reported behavior. Two 200-episode runs gave zero-sampling success rates of 70.5% and 71.0%; both Wilson intervals include the paper's 73.76% result. Random sampling produced 9.5% and 14.0%, or 11.75% pooled over 400 episodes, consistent with the approximately 10% low-success exploration regime highlighted by the paper. The official parallelism setting is 50 environments; pooled episode counts are retained as the authoritative metric.

The official Go2 configuration reproduced successfully with 4096 environments, 1500 iterations, and seed 42. Final training return was 40.620. All 31 saved checkpoints completed post-training evaluation; the final zero-sampling return was 41.529 and random-sampling return was 40.523, matching the paper's approximately 40-return regime. Separately, one official-budget Can FPO++ iteration completed from the released checkpoint, including collection, critic optimization, checkpointing, and both evaluation modes. A retrospective audit established that iteration 1 is critic-only and that all 35 saved actor tensors are bitwise identical to the released step-1000 EMA actor. Its exploratory 20/20 zero-source score is therefore evaluation variance and only pipeline evidence, not a training gain.

An exploratory inference-only WP-Past intervention failed decisively on Can. Under a matched seed,
the Gaussian source achieved 17/20 zero-sampling and 2/20 random-sampling successes, while replacing
the source prefix with the previous action chunk produced 0/20 in both modes. This is expected input-
distribution shift for a model trained exclusively on Gaussian sources. It rules out the inexpensive
checkpoint-only intervention, not the published method, which changes the source during BC training.

A faithful short training-time WP-Past adaptation also failed its screening gate. Matched 100-update
continuations from the same released checkpoint gave 14/20 zero and 0/20 random successes for the
Gaussian control, versus 0/20 in both modes for WP-Past. The warm run used valid previous-action
histories in 95-98% of logged batches, so the failure is not explained by universal Gaussian fallback.
This refutes H3 at the allocated screening budget and shifts the main effort to conditional reflow.

One-stage conditional reflow passed both pilot gates. On 128 fixed observation/source pairs, it
reduced normalized straightness error by 22.9%, reduced four-step endpoint MSE by 27.8%, and retained
the control's action diversity. In a matched 20-episode-per-cell Can screen, reflow improved zero-
sampling success from 11/20 to 14/20 at 10 Euler steps and from 14/20 to 15/20 at four steps. Random
success pooled across the two step counts was 3/40 for both methods. These results support the claimed
straightening and provide an initial task-level improvement signal, but the rollout sample is not yet
large enough to establish a stable benchmark gain.

An independent environment-seed evaluation narrowed this conclusion. Reflow again gained three
10-step zero-sampling successes (17/20 versus 14/20), yielding 31/40 versus 25/40 pooled over two
seeds. At four steps, however, the second seed reversed the one-success pilot advantage; the pooled
result was tied at 29/40. Random success remained comparable at 8/80 versus 7/80. Conditional reflow
therefore supports lower integration error and four-step performance preservation, while its apparent
official-step reward gain still needs a second training seed.

The independent training seed repeated both parts of the result. Reflow reduced normalized
straightness error by 22.4% and four-step endpoint MSE by 24.5%, with unchanged diversity. Its matched
rollout gained one 10-step zero success and three four-step zero successes. Across three comparisons
from two training seeds, reflow/control zero success was 47/60 versus 40/60 at 10 steps and 46/60
versus 43/60 at four steps; random success was tied at 13/120. This is sufficient evidence to move
from screening to one official-scale confirmation, but not yet to claim final benchmark improvement.

The official-scale confirmation separated the efficiency result from the reward claim. At 10 Euler
steps and 200 episodes, reflow/control zero success was 150/200 versus 147/200 (+1.5 percentage
points), far below the preregistered +5-point gate and statistically indistinguishable (Fisher
p=0.819). Random success was 23/200 versus 24/200. Pure one-stage reflow therefore preserves Can
performance while reliably straightening the flow, but it does not establish a benchmark reward
improvement and should not be integrated into online FPO++ on that premise.

A 50/50 mixture of dataset and teacher endpoints tested whether action grounding was the missing
ingredient. It retained an 11.0% straightness reduction and improved zero success in both small
screens, but random success changed from 5/20 versus 1/20 on the first seed to 0/20 versus 3/20 on
the confirmation seed. The deterministic signal is interesting, but unstable exploration violates
the online-policy requirement. The mixed-endpoint direction is stopped without a probability sweep.

Increasing FPO++ Monte Carlo compute did not solve the sparse-reward problem. In matched five-update
Can runs, 16 CFM samples per action produced 48/604 (7.95%) pooled collection success over actor
updates, compared with 55/603 (9.12%) for the official 8-sample control. Final 50-episode zero/random
evaluation was also slightly lower at 78%/12% versus 80%/14%. The candidate failed both the primary
+2-point improvement gate and random-sampling non-degradation gate, so larger sample counts are not
being pursued.

The rollout implementation also accumulates its `cfm_value_invalid_stored` mask across independent
iterations. Resetting the mask held valid CFM actions near 98%, whereas the released behavior fell
from 98.11% to 95.81% over five iterations. This is a real bookkeeping defect, but fixing it increased
pooled actor-update success only from 53/604 (8.77%) to 56/604 (9.27%, Fisher `p=0.841`). Final
zero/random evaluation was 82%/10% versus 84%/10%. Preserving these additional samples is correct,
but it does not establish a useful short-budget reward gain.

Increasing GAE lambda from 0.99 to 1.0 also failed to improve sparse-reward learning. Actor-update
collection success fell from 53/604 (8.77%) to 40/602 (6.64%, Fisher `p=0.195`). Final zero/random
evaluation was 80%/12% versus 84%/10%, and losses remained finite. Full-lambda GAE therefore did not
cause collapse, but its higher-variance Monte Carlo credit did not compensate for the scarcity of
successful random-source trajectories. Estimator-only changes are not the next priority.

A fixed 20% zero-source rollout subset tested whether the strong deterministic policy could supply
reward information to the weak Gaussian policy. It increased actor-update successes from 53 to 167,
while final zero/random evaluation remained 84%/12% versus 84%/10%. However, the separately measured
Gaussian subset fell to 34/480 (7.08%) from the control's 53/604 (8.77%). Delta-zero exploitation can
provide successful trajectories without collapse, but those trajectories did not improve the
full-noise behavior distribution under the preregistered short-budget metric.

Replacing the delta source with a scale-0.5 Gaussian established that continuous source overlap is
helpful but insufficient. The official scale-1 subset rose from H8's 7.08% to 44/484 (9.09%), only
0.32 points above the control and statistically indistinguishable (`p=0.915`). Final zero success tied
the control at 84%, while random success fell to 6%. The scale-0.5 subset itself succeeded on 86/147
episodes, confirming the reward-supply mechanism. Its extra episode boundaries drove cumulative valid
CFM actions down to 93.91%, making mask contamination a stronger interaction than in the baseline.

Resetting the validity mask removed that interaction but did not unlock reward transfer. Valid CFM
actions stayed near 98% in every iteration, yet official full-noise collection was 41/481 (8.52%),
slightly below the control's 53/604 (8.77%) and the no-reset mixture's 44/484 (9.09%). Final
zero/random evaluation was 86%/6%. The bookkeeping defect therefore explains lost training samples,
but not why abundant tempered-source successes fail to improve the Gaussian-source policy.

Normalized exponential advantage weights also failed to improve the standard-Gaussian update.
An ESS-controlled mirror-weighted FPO++ objective held weight ESS at exactly 50% and trained with
finite losses, but pooled actor-update success was 52/607 (8.57%) versus 53/604 (8.77%) for the
official signed-advantage control. Final zero/random evaluation was 82%/16%. The random point
estimate is non-degraded, but the preregistered collection metric establishes no learning gain.

The pending reflow variance mechanism also failed a paired audit. Across two independently trained
control/reflow pairs, mean CFM gradient-to-average cosine changed from 0.5027 to 0.5066 and from
0.4973 to 0.5125, far below the required +0.05. Gradient-norm CV improved only 2.9% in one pair and
worsened 12.3% in the other. Straighter flows slightly reduced scalar CFM-loss variation, but did
not materially align the gradients that drive FPO++ updates.

The paper's higher-quality Can initialization was identified as the same `95j3noe4` base-policy
run at step 6000. A locked 50-episode audit reproduced Figure A.8: Gaussian-random success was
34/50 (68%) and zero-source success was 48/50 (96%), compared with the figure's 64.36% and 96.11%.
This checkpoint removes the early sparse-reward bottleneck and is suitable for inexpensive method
screening, but its gain over step 1000 is attributable to behavior-cloning initialization quality.

Changing advantage normalization from per-minibatch moments to fixed rollout-level moments did not
improve the high-signal step-6000 screen. Actor-update collection was 426/734 (58.04%) versus
425/725 (58.62%) for control, and final random evaluation was 64% versus 72%. The hypothesis that
minibatch normalization drift materially destabilizes the update is therefore unsupported.

Joint antithetic CFM sampling produced a real variance reduction but failed its no-shift gate. On
two fixed Can batches, normalized MC8 gradient MSE fell by 35.8% and 23.8%, scalar-loss CV fell, and
within-mode gradient cosine improved. However, IID-versus-antithetic average-gradient cosine was
only 0.964 and 0.955, below the preregistered 0.99 requirement in both batches. This does not justify
an online update, and paired or quasi-Monte Carlo sampling variants are stopped without tuning.

Successful standard-Gaussian replay also failed before integration. A formal audit found 1,608
success-linked valid chunks, but after one official actor update only 40.82% of their positive ratios
remained below the PPO upper clip. Ratio ESS stayed high at 99.56%, so weight degeneracy was not the
problem. Pre/post replay-gradient cosine was 0.614, and the success-only gradient was nearly
orthogonal to the same-rollout fresh positive-advantage gradient (cosine 0.049). Selecting success
episodes introduces a conflicting update rather than simply reusing more on-policy information.

Fresh held-out CFM draws did not rescue the update-generalization hypothesis. In the locked H16
audit, held-out draws retained 54.69% active positive ratios versus 48.24% for stored draws, a gain
of only 6.45 points against the 15-point gate. Their median absolute log-ratio was 77.8% of the stored
value and ESS was 99.63%, but the held-out pre/post gradient cosine was only 0.743. Fixed-draw
overfitting is measurable but not dominant, so behavior-loss resampling per PPO epoch is stopped.

Advantage-stratified CFM sampling reduced normalized gradient MSE at fixed compute by 35.5% and
32.0% in two locked batches, while improving mean cosine to an independent MC64 reference. However,
the stratified and uniform repeat-average gradients had cosine only 0.977 and 0.974, below the 0.99
direction-preservation gate. Like antithetic sampling, this is genuine variance reduction without
sufficient evidence that the online update remains invariant, so fixed-budget sampling allocation is
closed without training or post-hoc tuning.

Held-out active-ratio early stopping did not rescue the unstable actor update. The locked 80% rule
selected epoch 1 because pooled positive active ratios fell from 85.5% to 72.9% at epoch 2, but the
two epoch-1 held-out gradient cosines were already only 0.754 and 0.263. One batch's surrogate moved
in the wrong direction, and the other retained only 5.4% of the epoch-10 gain. Repeated epochs worsen
drift, but the first actor step is already unreliable; threshold tuning or a fixed smaller epoch count
cannot address the underlying learning-signal problem.

The bounded discounted-success critic improved value calibration but not the learning signal needed
by the actor. On 1,837 fresh iteration-2 targets, BCE training reduced MSE by 12.18%, from 0.11796 to
0.10359, but Spearman correlation fell from 0.344 to 0.211. Gradient cosine to a Monte Carlo outcome
reference worsened by 0.073 in one 32-chunk batch and improved by 0.168 in the other. Because rank
quality and per-batch gradient improvement were preregistered requirements, H19 is stopped without
online integration or post-hoc tuning.

Continuous-action Direct Advantage Estimation failed more decisively. On 129 fully valid fresh
chunks, official GAE already achieved 0.817 Spearman correlation with Monte Carlo returns, while the
four-sample centered DAE head achieved -0.041. Its actor-gradient cosine was negative in both locked
32-chunk batches (-0.134 and -0.486), compared with 0.690 and 0.718 for control. Side-training loss
fell only 2.96% against the 20% gate. Learned replacement advantages are therefore stopped; the
useful signal to preserve is GAE ordering, not a new action-effect predictor.

A centered empirical-rank transform then preserved GAE ordering exactly but still reduced actor
gradient alignment. On two formal 32-chunk batches, control cosine was 0.855 and 0.887; rank-weight
cosine fell to 0.815 and 0.618. The transform's positive 16-chunk smoke did not survive the locked
seed and sample size. GAE magnitudes therefore contain useful information alongside their strong
ordering, and advantage replacement or nonlinear reweighting is no longer the next route.

A fixed fourfold lower actor learning rate also failed to preserve the useful pre-update direction.
At epoch 10, pooled active positive ratios were 0.537 for the `1e-5` control and 0.545 for the
`2.5e-6` candidate. Candidate gradient cosine changed by only -0.023 and +0.029 in the two locked
batches, missing both the 0.60 absolute and +0.20 relative gates. Its surrogate gains were positive
but retained only about 30% of control. Simple global step-size reduction therefore sacrifices
progress without resolving update-direction drift.

Accumulating all eight minibatch gradients before one Adam step per epoch did not resolve the drift
either. It retained 78.1% and 50.1% of the control surrogate gains, but epoch-10 gradient cosine
improved only in one batch (`0.291` to `0.409`) and worsened in the other (`0.280` to `0.212`). The
pooled active positive ratio fell from 0.537 to 0.424. Minibatch update order contributes at most
part of the instability; simply reducing optimizer-step frequency is not a viable improvement.

WiSE-FT-style interpolation between an exact step-6000 actor anchor and its four-update FPO++
endpoint also failed. On a paired 20-episode screen, anchor/endpoint/midpoint scored 100/90/95% with
zero sampling and 75/75/60% with random sampling. The midpoint recovered only half of the endpoint's
deterministic loss while introducing a new 15-point stochastic deficit. Moreover, the FPO++ endpoint
did not beat its anchor under random sampling on this seed, leaving no demonstrated online gain for
weight interpolation to retain.

Zero-endpoint PCGrad passed its mechanism audit but H30 did not produce a valid reward comparison.
The candidate projection was active on 36/320 actor optimizer steps (11.25%) with 99.954% mean
gradient-norm retention, satisfying the locked training-frequency gate. However, its 20-episode
screen used 50 asynchronous environments and retained the first 20 completions. Because successful
Can episodes terminate earlier than failures, the apparent 20/20 control scores are informatively
censored and invalid. Candidate evaluation was never run. H30 thus supports only the endpoint-
projection mechanism, not a reward conclusion.

Marginal-preserving temporal Gaussian sources failed despite an unusually strong mechanism result.
With `rho=0.9`, the measured source correlation was 0.8999, predicted action-chunk change fell by
68.03%, and 75.25% of IID action diversity remained. Yet the corrected balanced Can screen scored
0/20 for AR(1) versus 13/20 for IID, and all correlated-source episodes reached the 299-step horizon.
The intervention preserves every single-time `N(0,I)` marginal but changes the episode-level joint
source distribution; the resulting persistence suppresses useful replanning and adaptation. H31 is
closed without a confirmation run or correlation sweep.

Advantage-sign conflict projection also failed before online integration. On two fixed 32-chunk
batches, positive- versus negative-GAE gradient cosine was -0.0052 and +0.0549. Projection was active
only in the first batch, where it changed outcome-reference cosine from 0.94806 to 0.94797; in the
second batch the candidate was identical to control at 0.73389. The negative branch does not
materially cancel the positive branch, so sign-level PCGrad cannot address the observed actor drift.

Temporal per-timestep ratio clipping preserved the official on-policy gradient exactly but was
inactive at the first actor-step scale. Both fixed batches retained 100% positive active ratios under
the official chunk ratio and the temporal candidate, while post-step gradient-cosine gains were only
0.0000010 and 0.0000013. Finer clipping may differ after larger accumulated drift, but it cannot fix
the first-step direction problem identified by H18 and is stopped under the locked audit.

Coordinate-wise median aggregation also failed before optimizer integration. It stayed close to the
ordinary mean gradient in direction and norm, but reduced alignment to observed terminal outcomes
from 0.833 to 0.696 and from 0.688 to 0.619 in two independent replicas. Microbatch-specific
coordinates therefore contain useful joint structure rather than removable scalar outliers; robust
coordinate aggregation is closed without microbatch-count, trimming, or sign-voting variants.

Terminal-consistency filtering found that normalized GAE signs already matched centered discounted
terminal outcomes on 93.8-95.3% of labeled chunks. Removing the few contradictions left gradient
norms nearly unchanged and slightly reduced outcome alignment in both replicas. The residual error in
the official GAE gradient is therefore not a useful sign-misclassification problem.

Corrected balanced evaluation of zero-endpoint PCGrad produced a small same-direction gain but missed
its primary screen gate. Control scored 19/20 zero and 9/20 random, while the candidate scored 20/20
and 10/20. The candidate improved pooled success by 2/40 without deterministic degradation, but its
random gain was only 1/20 rather than the locked 2/20 required for confirmation. This is suggestive,
not benchmark-improvement evidence.

Paper-default PPO-RB ratio rollback also failed before reward integration. The corrected paired audit
matched all pre-update held-out statistics exactly and used 85 positive chunks. Rollback increased
epoch-10 active positive ratios from 25.8%/24.6% to 44.1%/39.8%, but gradient cosine changed from
0.454/0.463 to 0.431/0.571, failing the required gain in one batch. More decisively, surrogate gains
retained only 36.0% and 29.1% of control, below the locked 50% gate in both batches. A 66.42% rollback
activity rate confirms that the mechanism was active rather than vacuous; it exerted too much inward
pressure to preserve useful progress.

Full-vector geometric median-of-means aggregation did not rescue robust gradient aggregation. On two
fixed 96-chunk replicas, the geometric median converged in 12 and 11 iterations and retained the
arithmetic mean's direction and norm almost exactly. Nevertheless, outcome-reference cosine changed
from 0.741 to 0.736 and from 0.688 to 0.678. H34 and H37 together distinguish two failure modes:
coordinate-wise median damages joint structure, while geometric median preserves that structure but
finds no superior robust center in the four FPO++ block gradients.

Equal-NFE explicit midpoint integration passed a strong fixed-endpoint audit. Against source-matched
Euler-64 actions on 128 Can observations, midpoint-5 reduced endpoint MSE by 98.54% for Gaussian
sources and 98.88% for zero sources relative to Euler-10. Mean element standard deviation and mean
pairwise action distance were retained at 100.35% and 100.25%, all outputs were finite, and forward
hooks measured exactly ten velocity evaluations for both methods. This establishes a large numerical
fidelity gain at unchanged network-evaluation cost, but closed-loop reward remains untested.

That fidelity gain did not produce a reward-screen gain. In a balanced step-1000 Can comparison,
midpoint-5 improved zero-source success from 14/20 to 16/20 but reduced Gaussian-source success from
4/20 to 2/20. Pooled success tied at 18/40, failing both the locked +3-success gate and the per-mode
non-degradation gate. A closer approximation to the high-NFE flow endpoint is therefore not
intrinsically a better closed-loop action for the released policy.

Dense-reward Go2 confirmed the same separation more precisely. On 256 formal H200 observations,
midpoint-32 at exactly 64 NFE reduced zero/random endpoint MSE against Euler-256 to 12.52%/13.68% of
Euler-64 while retaining random diversity within 1.33%. In the paired 50-episode-per-mode screen,
zero return changed by only +0.0049 and random return fell by 0.0627; average gain was -0.0289 and the
pooled paired bootstrap interval was [-0.0400, -0.0187]. Higher endpoint fidelity therefore does not
translate into reward even when sparse binary success is removed as a confounder.

Uniformly averaging all four actor-update checkpoints from the H43 Square control trajectory also
failed. The finite 35-tensor average improved zero-source success from 7/20 to 8/20 but reduced
Gaussian-source success from 5/20 to 3/20, leaving pooled success down from 12/40 to 11/40. Together
with BC-to-final interpolation and rollout-local Adam, this shows that smoothing the short online
trajectory repeatedly favors modal deterministic behavior rather than preserving useful source
diversity.

Advantage-sign-stratified actor minibatches produced the strongest recent Gaussian-source direction
but missed the joint screen. All batching invariants and exact pre-intervention pairing passed.
Candidate random success rose from 4/20 to 7/20 while zero success fell from 8/20 to 7/20, so pooled
success improved from 12/40 to 14/40, one below the locked +3 gate. Stabilizing the sign mix seen by
each sequential Adam step can affect stochastic-source reward, but this screen does not establish a
stable aggregate gain.

Within-observation curvature selection passed its mechanism audit but failed its corrected Square
reward screen. The initial unseeded comparison appeared to improve success from 5/20 to 8/20 and was
invalidated before confirmation. With deterministic per-environment seeds, control scored 7/20 while
best-of-two scored 4/20. The candidate selected both branches 266/238 times over 504 replans and
lowered curvature by 12.62%, so the mechanism was active but its ranking signal was anti-correlated
with the aggregate reward outcome in this screen.

Full-horizon Square collection also passed its intended mechanism but failed the reward screen.
Changing the equal-interaction schedule from five 320-step rollouts to four 400-step rollouts exposed
17 first-iteration terminals, including 14 failures censored by the shorter control. The final policy
then tied Gaussian success at 6/20, reduced zero success from 10/20 to 8/20, and reduced pooled
success from 16/40 to 14/40. Complete failure labels do not compensate for one fewer fresh actor
rollout and are not the dominant bottleneck under this budget.

An asymmetric simulator-state critic also failed as a prerequisite for value-guided transport. On
32 complete Square episodes with 704 plan records, it increased held-out MSE by 13.58% and 10.39%
relative to the official frozen-visual conditioning critic. Spearman changed by only +0.0307 and
-0.0340 across the two episode-disjoint folds. Full state is therefore not automatically a better
short-budget value representation; RLDT's substantially larger replay and trainable double-Q system
cannot be replaced by this cheap critic substitution.

## Patterns and Insights

- FPO++ already uses the linear conditional flow-matching objective, so simply renaming it rectified flow is not a contribution.
- The transferable mechanism from arXiv:2209.03003 is recursive reflow using model-induced endpoint coupling.
- A reflow method that only improves inference speed is still useful, but it must not be presented as an RL performance gain.
- WarmPrior (arXiv:2605.13959) offers a more direct reward-improvement hypothesis for chunked manipulation: center the flow source on recent actions and retain residual Gaussian noise for exploration.
- WarmPrior cannot be grafted onto a Gaussian-trained FPO checkpoint at inference time; training-time source adaptation is necessary on Can.
- A short 100-update training-time adaptation is also insufficient: it preserves neither deterministic success nor random exploration.
- Conditional reflow's lower integration error transferred to a positive low-step rollout screen; this is the first improvement candidate to pass both mechanism and task gates.
- Small 20-episode screens overestimated pure reflow's reward effect; the 200-episode result retained only +1.5 points while confirming non-degradation.
- Mixing real and reflow endpoints can retain partial straightening and deterministic gains, but does not reliably preserve random-source exploration.
- Doubling CFM samples does not create more reward information; in this pilot it added compute while pooled early success fell by 1.17 points.
- CFM invalid-step state must be reset between independent rollouts, but the resulting 2.3-point increase in valid samples by iteration 5 was not enough to improve Can reward materially.
- Propagating the same sparse terminal rewards farther with lambda 1.0 reduced rather than improved early collection success; the bottleneck is reward-information acquisition, not lambda decay alone.
- Zero-source collection tripled the number of successes without final-policy collapse, but the disjoint delta source did not transfer into better Gaussian-source collection; source overlap is now the key constraint.
- Tempered Gaussian overlap recovered the delta-source deficit but not a useful gain; high-success source mixtures also amplify the cumulative invalid-mask defect by creating more episode boundaries.
- Correcting that amplified mask defect restored about 98% valid CFM actions but left full-noise success unchanged; failed transfer is an update/objective problem rather than a sample-retention problem.
- Soft positive mirror weights preserved stability and random evaluation but did not outperform signed FPO++ advantages; simple reward reweighting is not enough to improve early Can learning.
- Reflow's 22% curvature reduction does not translate into lower CFM gradient disagreement, so an online reflow auxiliary lacks its proposed stabilization mechanism.
- Can step 6000 faithfully reproduces the high-quality 96% zero / roughly 64% random base-policy regime in Appendix D.4, enabling more informative short-budget optimizer experiments without conflating initialization and algorithm effects.
- Freezing advantage moments over a complete rollout does not improve FPO++; simple rescaling changes are now closed alongside reward-weighting changes.
- Joint antithetic time/noise pairs reduce finite-MC gradient dispersion, but the locked audit could not establish sufficiently invariant average update direction.
- One-update-old successful chunks retain high ratio ESS but are mostly positive-clipped, and success-only selection produces a strongly biased gradient direction.
- Independent MC8 draws are modestly less shifted than stored draws, but do not retain enough active positive terms or gradient alignment to justify epoch-level behavior-loss resampling.
- Absolute-advantage 12/4 allocation improves fixed-budget estimator accuracy, but both it and antithetic sampling fail the locked cross-estimator direction gate; CFM sampling tricks are not the next reward-improvement route.
- Positive active-ratio fraction does not reliably diagnose gradient quality: it remained 85.5% after one epoch while one held-out gradient had rotated to cosine 0.263. Actor early stopping is closed without threshold tuning.
- Lower value MSE does not imply a better policy gradient: the discounted-success critic improved calibration while degrading value ranking and one batch's outcome-reference gradient alignment.
- Continuous-action Monte Carlo centering does not transfer DAE's discrete-action benefit here; it reversed an already strong GAE outcome ranking and both audited gradient directions.
- Exact GAE rank preservation is insufficient: discarding advantage magnitudes reduced outcome-gradient cosine in both formal batches.
- A fourfold actor learning-rate reduction scales down surrogate improvement but does not materially preserve held-out gradient direction; global Adam step size is not the dominant failure mechanism.
- Full-batch gradient accumulation retains surrogate progress but does not consistently preserve held-out direction, so sequential minibatch Adam steps are not the sole source of rotation.
- Midpoint weight interpolation can partially recover deterministic BC behavior, but here it degraded random-source success and cannot manufacture an online gain absent at the finetuned endpoint.
- Explicit Gaussian-bridge distillation is closed before implementation because H8 already resamples Gaussian CFM variables for zero-source rollout actions, and H15 rejects success-only gradient selection as a faithful proxy for positive GAE.
- Antithetic action-chunk averaging produced a 15-point small-screen gain but a 6-point loss on the independent confirmation; its pooled 52/70 success exactly ties zero-source inference, so source symmetry is not a stable inference improvement.
- Doubling visual replanning frequency by executing four rather than eight actions reduced the locked Can screen from 70% to 60%; the released checkpoint does not benefit from a shorter execution horizon without retraining.
- A mechanism-positive intervention is still not reward evidence: H29 passed its endpoint-drift audit, while H30's reward protocol was invalidated by first-completion censoring before candidate evaluation.
- Smooth action chunks are not intrinsically better control: H31 reduced adjacent chunk change by 68% but collapsed balanced random-source success from 65% to 0%.
- Matching each source marginal is insufficient for inference compatibility; the policy also depends on the episode-level joint source process created by repeated replanning.
- Small asynchronous screens must use equal per-environment episode quotas because successful Can episodes terminate earlier than failures.
- Positive- and negative-GAE FPO++ gradients are nearly orthogonal or weakly aligned on the audited step-6000 batches; their direct cancellation is not the source of unstable policy updates.
- Finer trust-region clipping cannot improve an update before any ratios reach the clipping boundary; at the official first-step scale, both chunk and temporal ratios remained fully active.
- Coordinate-wise robust aggregation can preserve global gradient scale while deleting task-relevant cross-coordinate structure; closeness to the mean is not evidence of improved outcome alignment.
- At the step-6000 checkpoint, official GAE gets the broad outcome sign right on more than 93% of labeled chunks; further sign filtering is inactive and does not improve gradient alignment.
- Balanced per-environment accounting rescued H30 from invalid censoring, but zero-endpoint PCGrad's +1/20 random and +2/40 pooled gains remain below the preregistered confirmation threshold.
- PPO-RB demonstrates that retaining more unclipped positive ratios does not by itself preserve the useful update: its strong boundary activity traded away roughly two thirds of surrogate progress and improved held-out direction in only one batch.
- Robust microbatch aggregation is closed at this scale: coordinate median damages useful cross-coordinate structure, while full-vector GMOM stays near the ordinary mean and slightly worsens outcome alignment in both replicas.
- A second-order sampler can exploit the existing flow field without retraining: midpoint-5 nearly matches the Euler-64 endpoint at the same NFE as Euler-10 and preserves Gaussian-source diversity. Reward transfer must still be established independently.
- Higher endpoint fidelity is not a sufficient reward objective: midpoint-5's deterministic +2/20 was exactly offset by a Gaussian-source -2/20 in balanced Can evaluation.
- Dense-reward Go2 closes the remaining endpoint-fidelity ambiguity: equal-NFE midpoint reduced integration error by about 86-87% but left zero return unchanged and caused a small paired random-return loss. Do not tune solver order, step count, or training around numerical fidelity alone.
- The released `x1_pred` KL-adaptive LR controller does preserve more active ratios, but in manipulation it immediately collapses to the transferred lower LR bound, retains only 27-32% of control surrogate progress, and fails to improve held-out direction consistently. Adaptive step-size reduction is closed alongside fixed low LR.
- Balanced Square evaluation gives both zero and random inference meaningful success and headroom (45% and 35%), making it a better short reward screen than sparse step-1000 or saturated step-6000 Can.
- Zero-endpoint PCGrad repeats the same sub-threshold pattern across Can and Square: it preserves zero-source success and adds only one random-source success in each balanced 20-episode screen. Endpoint conflict protection is therefore closed as a reward-improvement route without confirmation or tuning.
- Restoring a short-finetuned Square checkpoint from 16 to 8 executed actions improved Gaussian-source success by 2/20 but reduced zero-source success by 3/20 and pooled success by 1/40. More frequent visual feedback changes the source-mode tradeoff rather than providing a stable policy improvement.
- Clearing actor AdamW state at each new Square rollout is active and behaviorally consequential: it improved zero-source success by 4/20 but reduced Gaussian-source success by 3/20, yielding only +1/40 pooled. Stale cross-rollout moments are not simply harmful; they help retain stochastic-source behavior.
- Uniform averaging within the actor-update trajectory repeats the same tradeoff: zero success rose by 1/20 while Gaussian-source success fell by 2/20. Weight-space smoothing is closed alongside BC-to-final interpolation.
- Exact advantage-sign stratification raised Gaussian-source Square success by 3/20 with only a 1/20 zero loss, but pooled gain was 2/40 against the locked 3/40 gate. The directional signal is retained; nearby strata and normalization variants are closed.
- Global flow straightening and within-observation flow ranking both improve their geometry targets without improving reward: corrected lower-curvature best-of-two reduced Square random success by 3/20 despite a 12.62% curvature reduction.
- Extending Square rollouts to the full 400-step horizon converts censored initial failures into observed terminals, but at equal interaction budget it leaves random success unchanged and reduces deterministic and pooled success.
- RFS identifies a new intervention boundary after the flow-actor routes were exhausted: keep the pretrained velocity field frozen and optimize explicit latent-source and output-residual distributions with a standard PPO likelihood ratio.
- Joint RFS modulation is active but not sufficient on Square: despite bitwise base freezing and nonzero updates in both branches, it reduced mean and sampled success by two points each. Its 224-dimensional exact joint PPO ratio clipped 73-76% of minibatch samples, unlike H33's inactive temporal-clipping premise in FPO++.
- Temporally factorized RFS ratios cut clipping by 57.56 points and preserve the exact on-policy gradient, but larger effective modulation updates worsened pooled success to 11/40. High joint clipping was protective rather than the dominant failure; RFS modulation is closed at this budget.
- Privileged simulator inputs do not guarantee a better sparse-reward critic under the official short optimization budget: H50 worsened cross-fit MSE in both folds and failed to improve return ranking consistently.
- Potential-based Square stage shaping is a valid active credit signal: its complete-episode return telescoped to the sparse objective within `3.75e-16`, while 76.02% of failed-episode nonterminal transitions carried non-negligible signed progress. Reward transfer remains untested until the locked paired screen completes.
- Potential-based Square stage shaping passed its first paired reward screen: zero success improved by 1/20, Gaussian-source success by 3/20, and pooled success by 4/40. This is the first H51 reward-positive signal, but it requires independent-seed confirmation before being treated as a stable method improvement.
- Potential-based stage shaping did not confirm: the independent Gaussian result reversed from a screen gain of +3/20 to -5/50, and the independent pooled result was -4/100. Control and candidate tie exactly at 58/140 when both locked evaluations are pooled, so valid local credit alone does not produce a stable Square improvement.
- Surgical output-head FPO++ changed only 114,800 parameters and preserved frozen parameters exactly. It raised held-out gradient cosine by 0.559 and 0.373 and cut deterministic endpoint drift by about 99%, but retained only 2.09% and 2.25% of full FPO++ surrogate gain. The apparent stability is an almost inert reward update, so this route is closed before reward screening.
- Mirrored complete-episode parameter differences are active but not reproducible in the locked Square subspace. H53 passed all validity gates and found 11/16 and 9/16 nonzero directions in its two seed blocks, yet direction scores correlated -0.500, only 1/6 common nonzero signs agreed, and both cross-fit top quartiles transferred negatively. Critic-free binary success therefore does not provide a usable local ES ranking signal at this budget.
- ACT-inspired overlap ensembling is a valid continuity mechanism but not a reward improvement here. Across the complete H54 Can candidate screen it reduced median chunk-boundary jump by 33.82%, retained a 49.75% contribution from the new visual prediction, and passed every numerical gate, yet tied the official control exactly at 16/20 deterministic successes.
- Initial execution phase has limited reward relevance on released Can: only 3/20 common seeds changed outcome across all eight first-prefix phases, and even a hindsight oracle gained only 2/20 over the official phase. This fails the prerequisite for BCP-style adaptive continuation, so continuation-head training is not justified.
- Hard gripper-support projection is active but reward-neutral on released Can. It materially changed 67.55% of executed gripper commands while preserving arm coordinates and initial states exactly, yet all 20 common-seed outcomes remained identical to control; intermediate gripper magnitudes are not the limiting failure mechanism.
- Coordinate-factorized source noise produced a positive but sub-threshold random-source signal: zeroing only the gripper latent preserved exact unit-Gaussian arm sources and changed 1/20 control success to 3/20, with two rescues and no losses. The +2/20 gain misses the locked +3 gate and is not stable benchmark evidence.
- Deterministic gripper substitution does not isolate a reward gain from H57's clue. Keeping every Gaussian arm output bitwise and replacing only the zero-source gripper trajectory changed which single seed succeeded but tied control at 1/20; gripper-specific output post-processing is closed.

## Lessons and Constraints

- Reproduce released baselines before changing objectives.
- Do not label the first 48k-step Can iteration an actor update: it is critic-only, and a meaningful short FPO++ screen requires at least two iterations.
- Same-observation best-of-two endpoint/path ranking remains closed; H59 instead tested an exact AdamW extragradient correction of the first genuine actor update.
- Exact one-step AdamW extragradient is also closed: it actively changed the update at comparable displacement, but failed both outcome-direction gates and collapsed held-out surrogate progress in both audited batches.
- The official manipulation main benchmark is not zero-only: Figure 4 and the released evaluator report zero and Gaussian random sampling for every checkpoint. The zero-only `best` artifact is a convenience selection rule, not the complete paper metric, so historical zero/random tradeoffs remain closed.
- The released Go2 post-evaluator has a minimum-one-episode-per-environment rule. With 4096 environments, each nominal `eval_episodes=10` checkpoint metric actually averages 4096 episodes; the reproduced final returns are unchanged, but the earlier 10-episode description was incorrect.
- Use short validation runs before full 1500-iteration or multi-seed jobs.
- Stop a candidate when it degrades the primary metric beyond seed noise or fails to improve its claimed mechanism.
- Keep inference-only warm starts separate from faithful WarmPrior training in claims and experiment labels.

## Open Questions

- Does conditional reflow preserve the multimodal exploration that gives flow policies their advantage?
- Is there a reward-relevant confidence signal available from the frozen flow policy that is distinct from path straightness?
- Should reflow be an offline post-training stage, an auxiliary online loss, or both?
- Which reward-aware FPO++ mechanism can improve Can success without relying on unstable BC-source exploration changes?
- A frozen BC velocity-field anchor strongly conflicts with FPO++ and prevents drift, but global PCGrad retains only 34-46% of one-step surrogate gain. Generic flow-space behavior protection is too broad; a deterministic endpoint anchor is the next narrower test.
- Narrowing the anchor to the deterministic zero-source action endpoint passes the fixed-batch mechanism audit, but H30 could not test reward superiority because its matched control saturated at 20/20 in both modes.
- Does a gain in the non-saturated Square screen survive an independent seed and healthy-EGL confirmation?
- Which intervention can change reward-relevant exploration on Square rather than merely preserve the deterministic endpoint?
- Which training-time intervention remains distinct from the closed source, estimator, critic, optimizer, interpolation, and inference-averaging routes?
- Can joint source-latent and bounded residual modulation improve Square reward while keeping the released flow actor bitwise frozen?
- Does temporally factorizing RFS's explicit PPO ratio restore usable updates under its observed high-dimensional clipping, or is the modulation direction itself harmful?
- Can a value-aware action-density transport method improve exploration without directly moving the pretrained flow field or learning a harmful additive modulation?
- Which critic-free exploration mechanism can use the frozen policy's multimodal action support without ranking candidates by curvature or a learned value model?
- Which intervention targets contact-relevant action semantics rather than flow geometry, global smoothness, or chunk-boundary continuity?

## Optimization Trajectory

Can released-checkpoint and Go2 official-seed reproduction are complete. WarmPrior, mixed-endpoint reflow, increased or reallocated Monte Carlo sampling, full-lambda GAE, guided source mixtures, mask reset, ESS-weighted mirror updates, online reflow stabilization, rollout-level advantage normalization, advantage-sign-stratified minibatches, antithetic CFM sampling, successful replay, epoch-resampled CFM losses, active-ratio actor stopping, bounded discounted-success critics, continuous-action DAE, rank-based GAE weighting, terminal-sign filtering, fixed and KL-adaptive lower actor learning rates, full-batch gradient accumulation, exact AdamW extragradient, BC/FPO++ midpoint interpolation, within-trajectory SWA, explicit Gaussian-bridge distillation, antithetic endpoint averaging, curvature-ranked best-of-two, full-horizon Square collection, shorter receding-horizon execution, temporally correlated Gaussian inference, ACT-inspired overlap ensembling, BCP-style adaptive continuation, fixed binary gripper projection, gripper-latent source factorization, deterministic gripper substitution, advantage-sign and endpoint-anchor gradient surgery, temporal per-timestep clipping, coordinate-wise median and geometric-median gradient aggregation, PPO-RB ratio rollback, equal-NFE midpoint sampling, checkpoint-native Square execution horizon restoration, rollout-local Adam, output-head-only surgical fine-tuning, and fixed-subspace mirrored parameter ES are closed as reward-improvement routes. Reflow and midpoint retain geometry/sampling-fidelity value only. The next candidate must improve reward without trading away Gaussian-source exploration or collapsing useful surrogate progress.
