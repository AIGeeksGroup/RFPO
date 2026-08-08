# Analysis: Reflow CFM Gradient-Disagreement Audit

## Result

H2 is refuted. All gradients were finite and nonzero, and each checkpoint used the same 32 Can
dataset pairs and eight paired CFM samples. Reflow did not meet either preregistered consistency
gate on both training seeds.

| Training seed | Model | Mean gradient cosine | Gradient-norm CV | CFM-loss CV |
|---:|---|---:|---:|---:|
| 20260808 | control | 0.50271 | 0.03845 | 0.02478 |
| 20260808 | reflow | 0.50658 | 0.03735 | 0.02307 |
| 20260809 | control | 0.49733 | 0.03609 | 0.02457 |
| 20260809 | reflow | 0.51247 | 0.04054 | 0.02336 |

For seed 20260808, gradient cosine increased by only 0.00387 rather than the required 0.05, and
gradient-norm CV fell by only 2.9% rather than 10%. For seed 20260809, cosine increased by 0.01514
but gradient-norm CV worsened by 12.3%. CFM-loss CV fell modestly in both pairs, but this was not a
locked gate and did not transfer to consistent gradient norms.

## Interpretation

The replicated 22% reduction in flow curvature does not materially align the Monte Carlo CFM
gradients used by FPO++. Reflow lowers endpoint integration error and slightly homogenizes scalar
losses, but gradient direction and magnitude remain dominated by other variation. This breaks the
proposed mechanism for an online reflow auxiliary before any online training is needed.

Do not implement or train the online reflow regularizer, and do not broaden the parameter subset,
batch size, MC count, or audit threshold after observing this result. Retain conditional reflow only
as a supported geometry and low-step sampling-efficiency result.

Remote evidence is stored as `cfm_gradient_audit_seed20260808.json` and
`cfm_gradient_audit_seed20260809.json` inside the corresponding control/reflow result directories.
