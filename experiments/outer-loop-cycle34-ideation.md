# Outer Loop Cycle 34: Frozen Policy Modulation

H41-H47 show that preserving endpoints, changing execution horizons, optimizer state, weight
averaging, minibatch composition, curvature-based selection, and complete terminal collection alter
the zero/random tradeoff without improving pooled Square reward. These methods all continue to
modify or rank behavior through the flow actor itself.

RFS supplies a distinct intervention boundary: freeze the released flow actor and optimize an
explicit joint policy over its source latent and an output residual. Unlike FPO++, the PPO ratio is
the exact likelihood ratio of sampled modulation variables. Unlike inference source edits, the
source distribution is learned from reward while the residual branch can make local corrections.

H48 is prioritized over RLDT and VINE because it is implementable with the current observation
encoder and source-noise API, has a direct PPO interface, and supports a fixed 25,600-step Square
screen. RLDT requires a new value-driven density transport estimator, while VINE requires stable
gradients through every flow integration step; both have higher implementation and mechanism risk
before any reward evidence exists.
