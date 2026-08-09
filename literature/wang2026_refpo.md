# Wang et al. (2026): ReFPO

Wang et al., "ReFPO: Reflow Regularization for Flow Matching Policy Gradients," arXiv:2606.21086v1,
2026. The method adds the unweighted current conditional-flow-matching loss to the FPO objective:

```text
L_ReFPO = L_FPO + lambda * mean(current_CFM_loss)
```

The same stored time-noise pairs and current CFM losses used to construct the FPO proxy ratio are
reused, so Algorithm D.1 requires no additional velocity-network evaluation. The paper frames the
inner-loop argument as fixed-batch sensitivity rather than differentiation through policy data
collection. It reports `lambda=0.04` as the best MuJoCo Playground coefficient, compared with
`0.02`, `0.06`, and `0.08`, and `lambda=0.1` for GridWorld.

This is distinct from this project's refuted H2 audit. H2 asked whether an already reflow-trained
checkpoint aligns gradients across independent Monte Carlo CFM draws. ReFPO instead directly
regularizes the current fixed-batch CFM residual during each reward-weighted FPO inner update and
claims lower proxy-ratio volatility. The strongest objection is that an unweighted CFM term may act
as a behavior anchor: it can reduce ratio movement simply by suppressing useful policy improvement.

Source: https://arxiv.org/abs/2606.21086 (accessed 2026-08-09).
