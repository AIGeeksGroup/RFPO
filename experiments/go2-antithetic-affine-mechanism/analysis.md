# H75 Result: Cross-Policy Affine Antisymmetry

H75 passed every preregistered gate on the four independently trained official
Go2 policies. The audit used Euler-32, 128 environments, and eight consecutive
states per checkpoint, for 12,288 action coordinates per policy.

| Training seed | `cos(d+, d-)` | Antithetic residual | IID residual | Even RMS | Odd RMS |
|---:|---:|---:|---:|---:|---:|
| 42 | -0.9716 | 0.1184 | 0.7021 | 0.0370 | 0.3071 |
| 43 | -0.9643 | 0.1362 | 0.7127 | 0.0412 | 0.3042 |
| 44 | -0.9692 | 0.1238 | 0.7181 | 0.0409 | 0.3274 |
| 45 | -0.9721 | 0.1177 | 0.7066 | 0.0353 | 0.2967 |
| Mean | -0.9693 | 0.1240 | 0.7099 | 0.0386 | 0.3088 |

For every policy, the displacement induced by `-z` was nearly opposite to the
displacement induced by `z`. Symmetric averaging retained only 11.8-13.6% of a
single endpoint's RMS displacement from `F(o,0)`, whereas an equal-compute IID
pair retained 70.2-71.8%. The even component was roughly one eighth of the odd
component. This supplies cross-policy mechanism evidence for the source-
cancellation interpretation of H70/H73, while their paired closed-loop returns
remain the separate behavioral evidence.

All sources and endpoints were finite, negative sources were bitwise exact,
the deployed antithetic helper matched an independently constructed float32
mean bitwise, and actor parameters remained unchanged. The remote and local
aggregate analyses are byte-identical with SHA-256
`3c6198398d34618947ca84533e18601d0dcaf5a20d64463e92706b15b98022f3`.

This result does not establish novelty of antithetic noise or affine
antisymmetry, which Jia et al. (arXiv:2506.06185) already study in generative
models. It establishes that the same mechanism is stable across independently
trained flow policies in the closed-loop Go2 setting.
