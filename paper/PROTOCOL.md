# Paper Protocol: Symmetric Source Pairing for Flow-Policy Control

## Scope

This paper is a controlled empirical study of test-time source selection for
frozen FPO++ robot-control policies. It does not claim invention of antithetic
sampling, a new training algorithm, or a universally superior deployment
policy.

## Primary claim

At equal total neural-function evaluations (NFE), symmetric endpoint projection
reliably recovers the return deficit of the official random-source inference
rule across four independently trained Go2 policies. The associated endpoint
cancellation is consistent with an approximately affine, source-antisymmetric
flow map.

## Required controls and boundaries

- Report the four Go2 training seeds as the statistical units for cross-policy
  claims; episode-level bootstrap intervals may only describe within-policy
  paired evaluations.
- Compare symmetric pairing with random-source inference, two-source IID
  averaging at equal NFE, and deterministic zero-source inference.
- State that zero-source inference matches or exceeds symmetric pairing on Go2
  and Spot. Do not call symmetric pairing the best deployment policy.
- Report Spot as partial transfer: symmetric pairing beats random-source
  inference, but its interval against equal-NFE IID averaging crosses zero.
- Report the mirrored-rollout variance experiment as a negative boundary: local
  action antisymmetry does not yield a reliable closed-loop return control
  variate.
- Attribute antithetic noise and affine antisymmetry precedent to Jia et al.
  (arXiv:2506.06185); novelty is restricted to the robot-control evaluation,
  controls, mechanism measurements, and failure boundary.
- Describe throughput as policy-only inference on one H200 at batch 4096, not
  end-to-end simulator latency or single-robot latency.

## Frozen evidence

- Go2 equal-total-NFE and independent-policy studies: H69, H70, H73.
- Go2 affine mechanism: H75.
- Go2 batched throughput: H71.
- Spot cross-task control: H74.
- Go2 mirrored closed-loop rollout control: H77.

No additional result may be added without its own protocol preceding its
implementation and result commits.

## Statistical reporting

- Four-policy Go2 effects: mean, sample standard deviation, Student-t 95%
  interval, and exact one-sided sign-test p-value.
- Paired episode comparisons: paired point estimate, standard error, and the
  preregistered paired bootstrap 95% interval.
- Mechanism: per-policy observations plus cross-policy range/mean.
- Avoid language of statistical significance for the four-policy sign test;
  all four effects have the same sign, but the exact one-sided p-value is
  0.0625.

## Paper artifacts

- A complete anonymous LaTeX draft using an unmodified conference style.
- Figure-generation scripts that read committed JSON results rather than
  duplicating numbers by hand.
- Vector PDF and 300-DPI PNG exports for every data figure.
- Colorblind-safe colors plus redundant markers/hatching; grayscale legibility
  must be checked.
- A self-contained main table and figure captions that identify task, NFE,
  number of policies or episodes, and uncertainty definition.
- A reproducibility appendix mapping every paper number to a committed result,
  analysis script, and command.
- Bibliography entries fetched or verified against arXiv/Crossref metadata; no
  unverified citation may appear as fact.

## Completion gate

The draft is ready for scientist review only after it compiles without missing
references, all figures render without clipping, all reported numbers match the
committed JSON, and a claim audit finds no violation of the boundaries above.
