# H71 Result: Batched Antithetic Throughput

## Outcome

H71 passed every locked gate on one H200 with 4,096 frozen Go2 observations.

| Method | Median latency / 4,096 actions | Actions/s | Incremental peak CUDA memory |
| --- | ---: | ---: | ---: |
| random64 | 4.152 ms | 986,408 | 0.20 MB |
| antithetic32 sequential | 4.399 ms | 931,142 | 0.98 MB |
| antithetic32 batched | 3.685 ms | 1,111,405 | 2.56 MB |

The doubled-batch implementation was 1.194x as fast as sequential antithetic
inference and used 0.888x the latency of random64. Its maximum absolute action
difference from sequential inference was `4.77e-7` and RMS difference was
`5.17e-8`, well inside the locked `1e-5` numerical-equivalence tolerance. All
actions were finite and actor parameters were unchanged.

## Interpretation

H70's equal-NFE construction need not incur a wall-clock penalty at official
Go2 parallelism. The doubled batch exposes more parallel work while halving
integration depth, making the H70 candidate faster than both its sequential
implementation and the single-path Euler-64 baseline in this policy-only H200
measurement. Peak incremental allocation increased by about 2.36 MB relative
to random64, which is negligible on this device.

This is not an end-to-end simulator or single-robot latency result. Report the
device, batch size, warm-up, synchronization, and raw five-block timings. Use
the batched implementation in later rollout evaluations only after a paired
return-equivalence check.
