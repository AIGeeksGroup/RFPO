# RFPO

[[Paper](https://arxiv.org/abs/2610.10453)] [[Project page](https://aigeeksgroup.github.io/RFPO/)]

Locomotion-only code release for RFPO, built on Isaac Lab. The deployed policy is a flow-policy student; the Gaussian PPO teacher is training-only.

## Setup

```bash
git submodule update --init --recursive
cd isaaclab_experiments
bash setup_env.sh
source source_env.sh
```

## Training

```bash
# FPO++ baseline
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless

# RFPO
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  --fpo_variant all_ideas_teacher_kd \
  --teacher_checkpoint /path/to/gaussian_ppo_teacher.pt
```

The release includes no checkpoints. See `isaaclab_experiments/scripts/` for training and evaluation entry points.
