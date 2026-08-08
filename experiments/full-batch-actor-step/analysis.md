# H24 Full-Batch Actor Step Audit

## Result

H24 is refuted. Accumulating all eight minibatch gradients before one optimizer step per epoch did
not preserve the held-out positive-advantage gradient through ten FPO++ epochs.

| Epoch-10 metric | Control accumulation 1 | Candidate accumulation 8 | Candidate gate | Pass |
|---|---:|---:|---:|:---:|
| Pooled active positive ratio | 0.53711 | 0.42383 | >= 0.80 | No |
| Batch-0 gradient cosine | 0.29089 | 0.40900 | >= 0.60 and >= control + 0.20 | No |
| Batch-1 gradient cosine | 0.27988 | 0.21164 | >= 0.60 and >= control + 0.20 | No |
| Batch-0 surrogate gain | 0.01569 | 0.01226 | positive and >= 50% of control | Yes |
| Batch-1 surrogate gain | 0.02288 | 0.01147 | positive and >= 50% of control | Yes |

The candidate had 162 eligible positive chunks and audited the locked 64 chunks. Accumulation
retained useful surrogate movement, but the cosine response was inconsistent across batches:
batch 0 improved by 0.1181 while batch 1 worsened by 0.0682. The active positive-ratio fraction was
also 11.33 points below control. Sequential Adam updates on shuffled minibatches are therefore not
the sole cause of gradient rotation. No reward screen was run.

## Artifacts

- `candidate_accum8_seed20260822.json`
- Control: `../lower-actor-learning-rate/control_lr1e-5_seed20260822.json`
- Remote log: `can_step6000_fullbatchaccum8_16_osmesa_seed20260822.log`

The remote job exited normally, and physical GPU 1 returned to 0% utilization and 5 MiB.
