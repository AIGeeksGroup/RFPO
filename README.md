# RFPO: Rectified Flow Policy Optimization for Dexterous Hand-Object Interactions

This repository contains the cleaned RFPO training code used for dexterous
hand-object manipulation experiments in the ViViDex/SAPIEN environment. It keeps
the original ViViDex simulator stack as the environment foundation, and adds the
RFPO/FPO training pipeline, RM75 + Inspire hand support, experiment scripts, and
deployment notes.

The original ViViDex README has been preserved as
[`vividex-readme.md`](vividex-readme.md).

## Repository Contents

- `algos/rl/`: PPO and RFPO/FPO training code.
- `hand_imitation/`: SAPIEN manipulation environments and robot wrappers.
- `assets/robot/`: URDFs and meshes for UR5 + Allegro and RM75 + Inspire hand.
- `norm_trajectories/`: normalized reference trajectories, including RM75
  retargeted trajectories.
- `scripts/local/`: local training entrypoints for Allegro and RM75 experiments.
- `tools/`: BC collection, BC pretraining, RL training, evaluation, rendering,
  and RM75 retargeting utilities.
- `docs/`: RFPO design notes and real-robot BC+PPO/RFPO deployment notes.

## Main Training Entry Points

Run the paired Allegro BC+PPO / BC+RFPO pipeline:

```bash
scripts/local/run_allegro_bc_rfpo_vs_ppo_500k.sh
```

Run the paired RM75 BC+PPO / BC+RFPO pipeline:

```bash
scripts/local/run_rm75_bc_rfpo_vs_ppo.sh
```

Both scripts generate BC datasets and BC warm-start checkpoints when missing,
then launch the corresponding online PPO/RFPO training runs. Large generated
artifacts such as `.local_runs`, checkpoints, videos, and virtual environments
are intentionally excluded from git.

## Real-Robot Notes

For real-robot deployment, see
[`docs/real_robot_bc_ppo_rfpo.md`](docs/real_robot_bc_ppo_rfpo.md). The short
version is:

- collect real-robot demonstrations;
- convert them into `observations/actions` NPZ datasets aligned with the policy
  observation/action spaces;
- pretrain BC checkpoints;
- verify the BC policy before online RFPO;
- implement a real-robot environment with reward, done, reset, and safety logic
  before attempting online RL on hardware.

## Experiment Summary

See
[`RM75_ALLEGRO_EXPERIMENT_SUMMARY_20260629.md`](RM75_ALLEGRO_EXPERIMENT_SUMMARY_20260629.md)
for the current Allegro and RM75 experiment comparison.
