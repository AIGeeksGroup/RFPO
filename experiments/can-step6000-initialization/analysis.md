# Can Step-6000 Initialization Audit

Date: 2026-08-08

Server: `haoyu@36.33.157.152:30061`, NVIDIA H200, physical GPU 1

Checkpoint: `95j3noe4_step_6000`, EMA weights, 10 Euler sampling steps

Evaluation seed: 20260811, 10 parallel environments

| Sampling source | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| Gaussian random | 34 | 50 | 68% |
| Zero | 48 | 50 | 96% |

The preregistered random-source gate was 20/50; the checkpoint passed with 34/50. The result also
matches Figure A.8's reported high-quality Can base policy (96.11% zero / 64.36% random). The
adjacent Appendix D.4 prose says 64.06% random, a small internal discrepancy that does not affect
the conclusion.

H12 is supported. The step-6000 checkpoint removes the sparse-success initialization bottleneck
and can be used for short, paired method screening. It is a stronger initialization, not an
algorithmic improvement, and any surviving method must ultimately be evaluated from the official
step-1000 benchmark checkpoint.
