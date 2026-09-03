# Reproducing FPO++ Experiments

This document provides comprehensive instructions for reproducing manipulation experiments from the
``Flow Policy Gradients for Robot Control". The experiments evaluate four fine-tuning methods
(FPO++, Vanilla FPO, DPPO Learned Noise, DPPO Fixed Noise) across five robotic manipulation
tasks using pretrained behavior cloning (BC) checkpoints. The development environment is Ubuntu 24.04 and a single L40S GPU with 46GB of VRAM.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites and Environment Setup](#prerequisites-and-environment-setup)
3. [Base Policies](#base-policies)
4. [Experiment 1: Main Benchmark (60 runs)](#experiment-1-main-benchmark-60-runs)
5. [Experiment 2: Checkpoint Ablation (12 runs)](#experiment-2-checkpoint-ablation-12-runs)
6. [Experiment 3: FPO Ablation Study (18 runs)](#experiment-3-fpo-ablation-study-18-runs)
7. [Evaluating Base Policies](#evaluating-base-policies)
8. [Plotting Results](#plotting-results)
9. [W&B Run IDs for Cross-Referencing](#wandb-run-ids-for-cross-referencing)
10. [Hardware Requirements](#hardware-requirements)

---

## Overview

**Total runs:** 90 training runs + 10 base policy evaluations

| Experiment | Description | Runs |
|---|---|---|
| Main Benchmark | 5 tasks x 4 models x 3 seeds | 60 |
| Checkpoint Ablation | Can task (step_6000) x 4 models x 3 seeds | 12 |
| FPO Ablation | 2 tasks x 3 models x 3 seeds | 18 |

**Training script:** `finetune_online_rl.py` (all runs use `--distributed True`)
**W&B project:** `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning`

**Four fine-tuning methods:**

| Method | `loss_mode` | Key distinguishing parameters |
|---|---|---|
| FPO++ | `fpo` | `sde_sigma=0`, `cfm_loss_average_group_size=1`, `cfm_loss_use_huber=True`, clamping enabled |
| Vanilla FPO | `fpo` | `sde_sigma=0`, `cfm_loss_average_group_size=-1`, `cfm_loss_use_huber=False`, no clamping |
| DPPO Learned Noise | `dppo` | `sde_sigma=0.18`, `learn_sde_sigma=True`, noise injection enabled |
| DPPO Fixed Noise | `dppo` | Fixed `sde_sigma` per task, `learn_sde_sigma=False` |

---

## Prerequisites and Environment Setup

### 1. Set up the conda environment

Run the one-time setup script:

```bash
bash setup_env.sh
```

This installs:
- Python 3.10 via miniconda
- Conda environment `fpo_manipulation`
- robosuite (from `thirdparty/`)
- dexmimicgen (cloned from NVlabs)
- lerobot (from `thirdparty/`)
- ffmpeg 7.1.1
- Additional packages: matplotlib, seaborn, tyro, transformers==4.46.3
- Pinned requirements from `thirdparty/lerobot_requirements.txt`

### 2. Activate the environment

Before running any experiment:

```bash
source source_env.sh
```

### 3. W&B authentication

Ensure you are logged into W&B with access to the `SOME-WANDB-ENTITY` entity:

```bash
wandb login
```

---

## Base Policies

All fine-tuning experiments start from pretrained BC checkpoints. The checkpoints are available
from two sources:

### Downloading pretrained checkpoints

**Option 1: Google Drive (recommended)**

Pretrained checkpoints are hosted on [Google Drive](https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t?usp=sharing).
Download the full folder using `gdown`:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t -O downloaded_checkpoints
```

**Option 2: W&B artifacts**

Checkpoints can also be downloaded from the `SOME-WANDB-ENTITY/flow-bc` W&B project using
`eval_checkpoint.py`:

```bash
python eval_checkpoint.py \
  --wandb_run_id 95j3noe4 \
  --wandb_project flow-bc \
  --checkpoint_step step_1000 \
  --eval_env Can \
  --eval_num_episodes 0
```

### Available checkpoints

| Task | Env Name | Base Policy Run ID | Checkpoint Step (Main) | Checkpoint Step (Ablation) | Google Drive Directory |
|---|---|---|---|---|---|
| Can | `Can` | `95j3noe4` | `step_1000` | `step_6000` | `95j3noe4_step_1000`, `95j3noe4_step_6000` |
| Square | `Square` | `trc7rbt0` | `step_110000` | -- | `trc7rbt0_step_110000` |
| Box Clearance | `TwoArmBoxCleanup` | `lainyisy` | `step_10000` | -- | `lainyisy_step_10000` |
| Tray Lifting | `TwoArmLiftTray` | `ri0w9j39` | `step_20000` | -- | `ri0w9j39_step_20000` |
| Threading | `TwoArmThreading` | `6vqrn614` | `step_10000` | -- | `6vqrn614_step_10000` |

### Finetuning from downloaded checkpoints

Use `--base_policy_local_path` instead of `--base_policy_wandb_run_id` to finetune from
a local checkpoint:

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py --distributed True --base-policy-wandb-project flow-bc --load-ema True --wandb_project flow-bc-fpo-finetuning --wandb_enable True --gradient_accumulation_steps 1 --num_minibatches 8 --log_freq 1 --save_freq 2 --rollout_freq 2 --eval_num_episodes 200 --data_collection_steps 1600 --do_chunk_level_ppo True --eval_ema False --exploration_noise_std None --freeze_vision_encoder True --gae_lambda 0.99 --n_action_samples 8 --n_action_steps 16 --num_envs 3 --sampling_steps 10 --spo_clip_coef 0.01 --zero_sampling True --base_policy_wandb_run_id 95j3noe4 --checkpoint_step step_1000 --experiment finetune-fpo++-can --total_timesteps 5000000 --task Can --eval_env Can --discount 0.99 --sde_sigma 0 --cfm_loss_average_group_size 1 --cfm_loss_use_huber True --cfm_loss_huber_delta 0.5 --clip_coef 0.02 --max_grad_norm 5 --clamp_logratio 5 --clamp_old_cfm_loss 4 --trust_region_mode ppo --seed 0
```

Each checkpoint directory has this structure:

```
<run_id>_<step>/
├── optimizer.pt          # Optimizer state (not needed for finetuning)
└── policy/
    ├── config.json       # Model architecture and training config
    └── model.safetensors # Model weights (includes EMA weights)
```

### Pretraining base policies from scratch

To retrain the base policies, use `pretrain_flow_bc.py`. All base policies share:

```
--policy flowmatching
--network_architecture mlp --mlp_dims "[1024, 1024, 1024]"
--vision_backbone vit
--flow_network_output_param u --cfm_loss_mode u
--cfm_loss_use_huber False --grad_clip_norm 25
--horizon 16 --n_action_steps 8 --sampling_steps 10
--batch_size 512 --learning_rate 1e-4 --lr_backbone 1e-5
--weight_decay 1e-6 --ema_power 0.995
--enable_geometric_augmentations True --seed 3
```

Task-specific parameters:

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `dataset` | ankile/robomimic-mh-can-image | ankile/robomimic-mh-square-image | ankile/dexmg-two-arm-box-cleanup | ankile/dexmg-two-arm-lift-tray | ankile/dexmg-two-arm-threading |
| `image_observation_keys` | robot0_eye_in_hand_image | agentview_image | agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image | agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image | agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image |
| `eval_env` | Can | Square | TwoArmBoxCleanup | TwoArmLiftTray | TwoArmThreading |
| `steps` | 500000 | 1000000 | 1000000 | 1000000 | 1000000 |
| `max_num_episodes` | -- | 100 | -- | -- | -- |

Launch all 5:

```bash
DRY_RUN=1 bash scripts/run_pretrain_base_policies.sh   # preview
bash scripts/run_pretrain_base_policies.sh              # execute
```

---

## How to Launch Distributed Training

All experiments use PyTorch DDP via `torchrun`. The general command format is:

```bash
torchrun --nproc_per_node=$NUM_GPUS finetune_online_rl.py \
  --distributed True \
  [... experiment-specific parameters ...]
```

The helper scripts in `scripts/` default to `NUM_GPUS=1` but this can be overridden:

```bash
NUM_GPUS=4 bash scripts/run_main_benchmark.sh
```

To preview commands without executing (dry run):

```bash
DRY_RUN=1 bash scripts/run_main_benchmark.sh
```

---

## Experiment 1: Main Benchmark (60 runs)

**5 tasks x 4 models x 3 seeds = 60 runs**

### Shared parameters (all 60 runs)

```
--distributed True
--load-ema True
--gradient_accumulation_steps 1
--num_minibatches 8
--log_freq 1
--save_freq 2
--rollout_freq 2
--eval_num_episodes 200
--wandb_enable True
--data_collection_steps 1600
--do_chunk_level_ppo True
--eval_ema False
--exploration_noise_std None
--freeze_vision_encoder True
--gae_lambda 0.99
--n_action_samples 8
--n_action_steps 16
--num_envs 30
--sampling_steps 10
--spo_clip_coef 0.01
--zero_sampling True
```

### Task-specific shared parameters

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `task` / `eval_env` | Can | Square | TwoArmBoxCleanup | TwoArmLiftTray | TwoArmThreading |
| `base_policy_wandb_run_id` | 95j3noe4 | trc7rbt0 | lainyisy | ri0w9j39 | 6vqrn614 |
| `checkpoint_step` | step_1000 | step_110000 | step_10000 | step_20000 | step_10000 |
| `total_timesteps` | 5000000 | 8000000 | 5000000 | 8000000 | 8000000 |
| `discount` | 0.99 | 0.995 | 0.995 | 0.999 | 0.999 |

### 1.1 FPO++ (15 runs)

Model-specific: `loss_mode=fpo` (default), `sde_sigma=0`, `cfm_loss_average_group_size=1`, `cfm_loss_use_huber=True`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.02 | 0.01 | 0.03 | 0.03 | 0.01 |
| `max_grad_norm` | 5 | 25 | 5 | 1 | 1 |
| `cfm_loss_huber_delta` | 0.5 | 1 | 0.1 | 1 | 0.1 |
| `clamp_logratio` | 5 | None | 5 | 5 | 5 |
| `clamp_old_cfm_loss` | 4 | None | 4 | 4 | 4 |

Example command (Can, seed 0):

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-can \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 \
  --save_freq 2 \
  --rollout_freq 2 \
  --task Can \
  --eval_env Can \
  --eval_num_episodes 200 \
  --wandb_enable True \
  --data_collection_steps 1600 \
  --do_chunk_level_ppo True \
  --eval_ema False \
  --exploration_noise_std None \
  --freeze_vision_encoder True \
  --gae_lambda 0.99 \
  --n_action_samples 8 \
  --n_action_steps 16 \
  --num_envs 30 \
  --sampling_steps 10 \
  --spo_clip_coef 0.01 \
  --zero_sampling True \
  --discount 0.99 \
  --sde_sigma 0 \
  --cfm_loss_average_group_size 1 \
  --cfm_loss_use_huber True \
  --cfm_loss_huber_delta 0.5 \
  --clip_coef 0.02 \
  --max_grad_norm 5 \
  --clamp_logratio 5 \
  --clamp_old_cfm_loss 4 \
  --trust_region_mode ppo \
  --seed 0
```

### 1.2 DPPO Learned Noise (15 runs)

Model-specific: `loss_mode=dppo`, `sde_sigma=0.18`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| `max_grad_norm` | 25 | 5 | 1 | 25 | 1 |
| `cfm_loss_huber_delta` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `noise_injection_min` | 0.3 | 0.3 | 0.2 | 0.3 | 0.3 |
| `noise_injection_max` | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| `learn_sde_sigma` | True | True | True | True | True |

Additional shared: `cfm_loss_use_huber=True`, `cfm_loss_average_group_size=1`, `clamp_logratio=5`, `clamp_old_cfm_loss=4`, `trust_region_mode=ppo`

### 1.3 Vanilla FPO (15 runs)

Model-specific: `loss_mode=fpo` (default), `sde_sigma=0`, `cfm_loss_average_group_size=-1`, `cfm_loss_use_huber=False`, `clamp_logratio=None`, `clamp_old_cfm_loss=None`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| `max_grad_norm` | 25 | 25 | 5 | 25 | 25 |
| `cfm_loss_huber_delta` | 0.5 | 1 | 0.5 | 0.5 | 0.5 |

Additional shared: `trust_region_mode=ppo`

### 1.4 DPPO Fixed Noise (15 runs)

Model-specific: `loss_mode=dppo`, `cfm_loss_average_group_size=1`, `cfm_loss_use_huber=True`, `clamp_logratio=5`, `clamp_old_cfm_loss=4`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| `clip_coef` | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| `max_grad_norm` | 25 | 25 | 25 | 5 | 5 |
| `cfm_loss_huber_delta` | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| `sde_sigma` | 0.3 | 0.3 | 0.24 | 0.24 | 0.24 |

Additional shared: `trust_region_mode=ppo`

### Launch all 60 runs

```bash
# Preview commands first
DRY_RUN=1 bash scripts/run_main_benchmark.sh

# Run all experiments
bash scripts/run_main_benchmark.sh
```

---

## Experiment 2: Checkpoint Ablation (12 runs)

**Can task only, using `step_6000` checkpoint, 4 models x 3 seeds = 12 runs**

This ablation studies the effect of using a different base policy checkpoint
(`step_6000` instead of `step_1000` for the Can task).

### Parameters

All runs share the same task parameters as Can in the main benchmark except
`checkpoint_step=step_6000`.

| Parameter | FPO++ | Vanilla FPO | DPPO Learned | DPPO Fixed |
|---|---|---|---|---|
| `loss_mode` | fpo | fpo | dppo | dppo |
| `cfm_loss_average_group_size` | 1 | -1 | 1 | 1 |
| `cfm_loss_use_huber` | True | False | True | True |
| `clamp_logratio` | 5 | None | 5 | 5 |
| `clamp_old_cfm_loss` | 4 | None | 4 | 4 |
| `clip_coef` | 0.02 | 0.02 | 0.01 | 0.01 |
| `max_grad_norm` | 5 | 5 | 1 | 5 |
| `cfm_loss_huber_delta` | 0.5 | 0.5 | 0.5 | 0.5 |
| `sde_sigma` | 0 | 0 | 0.18 | 0.3 |
| `learn_sde_sigma` | -- | -- | True | False |
| `noise_injection_min` / `max` | -- | -- | 0.2 / 0.5 | -- |

### Launch all 12 runs

```bash
DRY_RUN=1 bash scripts/run_checkpoint_ablation.sh   # preview
bash scripts/run_checkpoint_ablation.sh              # execute
```

---

## Experiment 3: FPO Ablation Study (18 runs)

**Square & Threading tasks, 3 models x 3 seeds x 2 tasks = 18 runs**

This ablation compares FPO++ against ASPO and per-action ratio variants.

### Parameters

| Parameter | FPO++ | ASPO | Per-action ratio |
|---|---|---|---|
| `trust_region_mode` | ppo | aspo | ppo |
| `cfm_loss_average_group_size` | 1 | 1 | -1 |
| `cfm_loss_use_huber` | True | True | True |
| `clamp_logratio` | None (Square) / 5 (Threading) | 5 | 5 |
| `clamp_old_cfm_loss` | None (Square) / 4 (Threading) | 4 | 4 |

All other task-specific parameters (discount, max_grad_norm, clip_coef, etc.) match
the FPO++ row from the main benchmark for the respective task.

### Launch all 18 runs

```bash
DRY_RUN=1 bash scripts/run_fpo_ablation.sh   # preview
bash scripts/run_fpo_ablation.sh              # execute
```

---

## Evaluating Base Policies

To evaluate the pretrained BC base policies (both zero-sampling and random-sampling):

```bash
bash scripts/eval_base_policies.sh
```

This evaluates each of the 5 base policy checkpoints with:
- **Zero sampling** (default, deterministic inference)
- **Random sampling** (`--zero-sampling False`, stochastic inference)

Each evaluation runs 200 episodes across 30 parallel environments.

Example command:

```bash
python eval_checkpoint.py \
  --wandb_run_id 95j3noe4 \
  --wandb_project flow-bc \
  --checkpoint_step step_1000 \
  --eval_env Can \
  --eval_num_episodes 200 \
  --eval-num-envs 30 \
  --load-ema True
```

---

## Plotting Results

Use `plot_results.py` to generate training curve plots from W&B run data. The script fetches
metrics from the `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning` project and produces PDF figures.

### Plot modes

| Mode | Description | Default output |
|------|-------------|----------------|
| `main_benchmark` | All 5 tasks, 4 methods, with both zero and random sampling side by side | `main_benchmark_plot.pdf` |
| `fpoplusplus_ablation` | FPO++ ablation on Square & Threading: FPO++ vs ASPO vs per-action ratio | `fpoplusplus_ablation_plot.pdf` |
| `base_policy_ablation` | Can task with multiple views (zoomed, zero vs random sampling) | `base_policy_ablation_plot.pdf` |

### Example commands

```bash
# Main benchmark training curves (all tasks, all methods, zero + random sampling)
python plot_results.py --mode main_benchmark

# FPO++ ablation plots (Square & Threading)
python plot_results.py --mode fpoplusplus_ablation

# Base policy ablation (Can task, detailed views)
python plot_results.py --mode base_policy_ablation

# Save to a custom filename
python plot_results.py --mode main_benchmark --output my_custom_plot.pdf
```

---

## W&B Run IDs for Cross-Referencing

### Experiment 1: Main Benchmark

#### FPO++

| Task | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Can | `wbxzw7z3` | `e0h4wy1r` | `i6rmkgrh` |
| Square | `rsmunbo4` | `wzyv707a` | `z2u9ryms` |
| Box Clearance | `ujjdjtov` | `qlux3x9d` | `o6kv0feo` |
| Tray Lifting | `dcwh6cja` | `rdigmvk4` | `oi516dvb` |
| Threading | `bt3sl4ex` | `g2ldgoss` | `fu8wpmbd` |

#### DPPO Learned Noise

(Run IDs available in W&B project `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning`)

#### Vanilla FPO

(Run IDs available in W&B project `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning`)

#### DPPO Fixed Noise

(Run IDs available in W&B project `SOME-WANDB-ENTITY/flow-bc-fpo-finetuning`)

### Experiment 2: Checkpoint Ablation (Can, step_6000)

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `txf5crib` | `ylxhw9uu` | `6zp46k32` |
| Vanilla FPO | `gzm22fqh` | `opygcs30` | `89jg66ll` |
| DPPO Learned | `7lz0maky` | `ss13djab` | `nqkx8w0s` |
| DPPO Fixed | `4iz08o2h` | `0wo72noe` | `0yibse8t` |

### Experiment 3: FPO Ablation

**Square:**

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `z2u9ryms` | `wzyv707a` | `rsmunbo4` |
| ASPO | `whhhg3oc` | `safs53cm` | `069ligh5` |
| Per-action ratio | `eikmrvae` | `kf0ywqzj` | `4x5cxo0w` |

**Threading:**

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| FPO++ | `bt3sl4ex` | `fu8wpmbd` | `g2ldgoss` |
| ASPO | `4pwr3dzu` | `ir471vi8` | `nx0ktwru` |
| Per-action ratio | `l5hdijzn` | `pcu57hf1` | `xmklzaca` |

---

## Hardware Requirements

- **GPU:** All experiments use multi-GPU DDP training via `torchrun`. The scripts default to
  `NUM_GPUS=1` but can be configured. More GPUs will speed up training proportionally.
- **Memory:** Each run spawns 30 parallel simulation environments (`num_envs=30`) and runs
  200 evaluation episodes. Expect significant CPU and RAM usage.
- **Storage:** Each run saves checkpoints every 2 iterations (`save_freq=2`) and logs to W&B.
  Plan for substantial disk space for checkpoints across 90 runs.
- **Time:** The `total_timesteps` ranges from 5M to 8M per run. Wall-clock time depends on
  hardware but expect each run to take several hours to days on a single GPU.

---

## Helper Scripts

| Script | Description |
|---|---|
| `scripts/run_pretrain_base_policies.sh` | Pretrains all 5 base policies (Flow Matching BC) |
| `scripts/run_main_benchmark.sh` | Launches all 60 main benchmark runs |
| `scripts/run_checkpoint_ablation.sh` | Launches all 12 checkpoint ablation runs |
| `scripts/run_fpo_ablation.sh` | Launches all 18 FPO ablation runs |
| `scripts/eval_base_policies.sh` | Evaluates all 5 base policies (zero + random sampling) |

All scripts support:
- `DRY_RUN=1` -- print commands without executing
- `NUM_GPUS=N` -- configure number of GPUs for torchrun (default: 1)
