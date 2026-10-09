# RFPO: Rectified Flow Policy Optimization for Embodied Control

[[Paper](https://arxiv.org/abs/2610.10453)] [[Project page](https://aigeeksgroup.github.io/RFPO/)]

This release contains the **locomotion** implementation used by RFPO. It trains a flow-policy student in Isaac Lab and deploys that student with a small Euler integration budget, including one-step execution.

## Method

RFPO retains the on-policy FPO objective and adds three training-only components:

- **Reward-aware online Reflow:** rectifies student-induced transport paths, emphasizing positive-advantage states.
- **Frozen Gaussian PPO teacher:** provides multi-budget action-space supervision for the flow student.
- **Adaptive integration budget and zero-start mixing:** improve fidelity when the deployment budget is much smaller than the full 64-step policy.

The teacher and all auxiliary losses are removed at deployment; the deployed controller is one flow-policy student. The precise objective, variants, and evaluation protocol are documented in [docs/Paper-Method.md](docs/Paper-Method.md).

## Scope and layout

This repository intentionally includes locomotion only. The implementation supports Unitree Go2, Spot, H1, G1, and G1 whole-body tracking.

```text
isaaclab_experiments/
├── isaaclab_fpo/                 # RFPO algorithm and Isaac Lab extension
│   ├── isaaclab_fpo/algorithms/fpo.py
│   ├── isaaclab_fpo/task_cfgs.py # robot / RFPO variant configurations
│   └── scripts/                  # training and evaluation entry points
├── docs/                          # method notes and implementation details
├── thirdparty/IsaacLab/           # git submodule
└── thirdparty/whole_body_tracking/# git submodule
docs/
├── Paper-Method.md
└── Paper-Experiments.md
```

## Setup

Requirements: Linux, NVIDIA GPU, and CUDA 12.1 or newer.

```bash
git submodule update --init --recursive
cd isaaclab_experiments
bash setup_env.sh
source source_env.sh
```

## Train and evaluate

The default task configuration is the FPO++ baseline. RFPO's complete Go2 variant requires a compatible frozen Gaussian PPO checkpoint:

```bash
cd isaaclab_experiments

# FPO++ baseline
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless

# RFPO: reward-aware Reflow + adaptive budget + PPO-teacher KD
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  --fpo_variant all_ideas_teacher_kd \
  --teacher_checkpoint /path/to/gaussian_ppo_teacher.pt

# Evaluate a trained RFPO checkpoint at multiple Euler budgets.
python isaaclab_fpo/scripts/eval_sampling_steps.py --help
```

For the exact training settings and ablation variants, see `isaaclab_experiments/scripts/run_go2_all_ideas_teacher_kd.sh`, `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/task_cfgs.py`, and the method note above.

## Acknowledgements

The implementation builds on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and adapts components from [rsl_rl](https://github.com/leggedrobotics/rsl_rl). Their licenses and copyright notices are retained in the relevant source files.
