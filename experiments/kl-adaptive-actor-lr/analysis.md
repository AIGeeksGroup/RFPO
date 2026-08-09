# H39 Paired Audit Analysis

## Validity

The seeded control and adaptive runs both produced 102 positive fully valid iteration-2 chunks and
audited the locked 64. Their two pre-update batch records match exactly, including surrogate,
gradient norm, and ratio statistics. Both started the actor update at LR `1e-5`; all values were
finite and the held-out gradients remained nonzero.

## Results

| Batch | Metric | Control | Adaptive | Gate | Result |
|---:|---|---:|---:|---:|---|
| 0 | epoch-10 gradient cosine | 0.18661 | 0.33768 | candidate >= control + 0.10 and >= 0.25 | pass |
| 1 | epoch-10 gradient cosine | 0.56490 | 0.63409 | candidate >= control + 0.10 and >= 0.25 | fail (+0.06920) |
| 0 | active positive ratio fraction | 0.27734 | 0.40625 | candidate >= control + 0.05 | pass |
| 1 | active positive ratio fraction | 0.22266 | 0.39062 | candidate >= control + 0.05 | pass |
| 0 | surrogate-gain retention | 0.03078 -> 0.00842 | 27.36% | >= 50% | fail |
| 1 | surrogate-gain retention | 0.03092 -> 0.00980 | 31.67% | >= 50% | fail |

The first minibatch had exactly zero KL and held LR. Every subsequent event exceeded the high
threshold: the controller recorded 79 decreases, no increases, and no in-band holds. LR reached the
locked `1e-6` lower bound on the seventh minibatch. KL peaked at `0.002654`, remained above the
`0.0002` decrease threshold throughout, and was `0.000478` on average in epoch 10. The activity gate
requires both an increase and a divergence-triggered decrease, so it also fails.

## Decision

H39 is refuted. The transferred controller raised active-ratio fractions, but did so by collapsing
to its lower LR bound and retaining less than one third of the control surrogate gain. It did not
deliver the required replicated direction gain, and its two-sided adaptive behavior never became
active in this regime. Stop without reward training, another seed, target tuning, factor tuning,
bound tuning, or combining entropy regularization with this controller.

