# Experiment queue (GPU1 watcher)

G1 reflow is training on GPU1. After it reaches 2000:

1. Eval G1 `reflow`, G1 `baseline`, G1 `all_ideas_fpo`
   (`eval_sampling_steps.py`, 2048 envs, 10 episodes, steps `64 32 16 8 4 1`, modes `zero` `random`)
2. Train Go2 uniform `reflow`: 16384 envs x 2500 iters, seed 42, no teacher / KD / adaptive

Do not use GPU0 / GPU2 / GPU3.

Watcher: `scripts/wait_g1_reflow_eval_then_go2_reflow.sh`
Go2 train: `scripts/run_go2_reflow_full.sh`

Logs:

- watcher: `/dev/shm/g1_step_sweep/launch_logs/watcher.out`
- G1 eval summary: `logs/isaaclab_fpo/g1_step_sweep_after_reflow/step_sweep_summary.txt`
- Go2 reflow: `/dev/shm/go2_reflow_full` and `logs/isaaclab_fpo/go2_reflow_full_2500`
