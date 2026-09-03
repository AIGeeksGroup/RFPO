# FPO Finetuning Experiments Summary

All runs use `finetune_online_rl.py` from `far_manipulation-fpo` with `--distributed True`.
W&B project: `far-wandb/flow-bc-fpo-finetuning`

---

## Table of Contents

1. [Main Benchmark (5 tasks x 4 models x 3 seeds = 60 runs)](#1-main-benchmark)
2. [Base Policy Checkpoint Ablation — Can task, step_6000 (4 models x 3 seeds = 12 runs)](#2-base-policy-checkpoint-ablation)
3. [FPO Ablation Study — Square & Threading (3 models x 3 seeds x 2 tasks = 18 runs)](#3-fpo-ablation-study)

---

## Base Policies

| Task | Base Policy Run | Checkpoint Step | Project |
|---|---|---|---|
| Can | `95j3noe4` | step_1000 (main) / step_6000 (ablation) | flow-bc |
| Square | `trc7rbt0` | step_110000 | flow-bc |
| Box Clearance (TwoArmBoxCleanup) | `lainyisy` | step_10000 | flow-bc |
| Tray Lifting (TwoArmLiftTray) | `ri0w9j39` | step_20000 | flow-bc |
| Threading (TwoArmThreading) | `6vqrn614` | step_10000 | flow-bc |

---

## 1. Main Benchmark

5 tasks, 4 models, 3 seeds (0, 1, 2) each. All seeds within a task/model group use identical hyperparameters.

### Shared parameters across all models/tasks

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
--data_collection_steps=1600
--do_chunk_level_ppo=True
--eval_ema=False
--exploration_noise_std=None
--freeze_vision_encoder=True
--gae_lambda=0.99
--n_action_samples=8
--n_action_steps=16
--num_envs=30
--sampling_steps=10
--spo_clip_coef=0.01
--trust_region_mode=ppo  (except ASPO variants)
--zero_sampling=True
```

### Task-specific shared parameters

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| base_policy_wandb_run_id | 95j3noe4 | trc7rbt0 | lainyisy | ri0w9j39 | 6vqrn614 |
| checkpoint_step | step_1000 | step_110000 | step_10000 | step_20000 | step_10000 |
| total_timesteps | 5000000 | 8000000 | 5000000 | 8000000 | 8000000 |
| discount | 0.99 | 0.995 | 0.995 | 0.999 | 0.999 |

---

### 1.1 FPO++ (15 runs)

**Run IDs:**

| Task | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Can | wbxzw7z3 (not confirmed order) | e0h4wy1r | i6rmkgrh (seed=0) |
| Square | rsmunbo4 | wzyv707a | z2u9ryms |
| Box Clearance | ujjdjtov | qlux3x9d | o6kv0feo |
| Tray Lifting | dcwh6cja | rdigmvk4 | oi516dvb |
| Threading | bt3sl4ex | g2ldgoss | fu8wpmbd |

**Model-specific parameters:** `loss_mode=fpo` (default), `sde_sigma=0`, `cfm_loss_average_group_size=1`, `cfm_loss_use_huber=True`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| clip_coef | 0.02 | 0.01 | 0.03 | 0.03 | 0.01 |
| max_grad_norm | 5 | 25 | 5 | 1 | 1 |
| cfm_loss_huber_delta | 0.5 | 1 | 0.1 | 1 | 0.1 |
| clamp_logratio | 5 | None | 5 | 5 | 5 |
| clamp_old_cfm_loss | 4 | None | 4 | 4 | 4 |

#### Can
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-can-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=5 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Square
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-square-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Box Clearance
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id lainyisy \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-boxcleaning-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.03 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=5 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Tray Lifting
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id ri0w9j39 \
  --load-ema True \
  --checkpoint_step step_20000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-tray-base-uul2gclip25-dec25-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmLiftTray --eval_env TwoArmLiftTray --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.03 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=1 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Threading
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-threading-base-uul2gclip25-dec25-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=1 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

---

### 1.2 DPPO – Learned Noise (15 runs)

**Model-specific parameters:** `loss_mode=dppo`, `sde_sigma=0.18`, `learn_sde_sigma=True` (CAN/Square/BoxClearance), noise injection enabled.

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| clip_coef | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| max_grad_norm | 25 | 5 | 1 | 25 | 1 |
| cfm_loss_huber_delta | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| noise_injection_min | 0.3 | 0.3 | 0.2 | 0.3 | 0.3 |
| noise_injection_max | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| learn_sde_sigma | True | True | True | (default=False) | (default=False) |

#### Can
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-can-base-uul2gclip25-dec28-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --learn_sde_sigma=True --loss_mode=dppo --max_grad_norm=25 \
  --n_action_samples=8 --n_action_steps=16 \
  --noise_injection_max=0.5 --noise_injection_min=0.3 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.18 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Square
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-square-base-uul2gclip25-dec28-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --learn_sde_sigma=True --loss_mode=dppo --max_grad_norm=5 \
  --n_action_samples=8 --n_action_steps=16 \
  --noise_injection_max=0.5 --noise_injection_min=0.3 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.18 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Box Clearance
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id lainyisy \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-boxcleaning-base-uul2gclip25-dec28-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.03 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --learn_sde_sigma=True --loss_mode=dppo --max_grad_norm=1 \
  --n_action_samples=8 --n_action_steps=16 \
  --noise_injection_max=0.5 --noise_injection_min=0.2 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.18 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Tray Lifting
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id ri0w9j39 \
  --load-ema True \
  --checkpoint_step step_20000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-tray-base-uul2gclip25-dec28-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmLiftTray --eval_env TwoArmLiftTray --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=25 \
  --n_action_samples=8 --n_action_steps=16 \
  --noise_injection_max=0.5 --noise_injection_min=0.3 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.18 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Threading
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-threading-base-uul2gclip25-dec28-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=1 \
  --n_action_samples=8 --n_action_steps=16 \
  --noise_injection_max=0.5 --noise_injection_min=0.3 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.18 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

---

### 1.3 Vanilla FPO (15 runs)

**Model-specific parameters:** `loss_mode=fpo` (default), `sde_sigma=0`, `cfm_loss_average_group_size=-1`, `cfm_loss_use_huber=False`, `clamp_logratio=None`, `clamp_old_cfm_loss=None`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| clip_coef | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 |
| max_grad_norm | 25 | 25 | 5 | 25 | 25 |
| cfm_loss_huber_delta | 0.5 | 1 | 0.5 | 0.5 | 0.5 |

#### Can
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-vanilla-fpo-can-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Square
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-square-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=False \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Box Clearance
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id lainyisy \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-vanilla-fpo-boxcleaning-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.03 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=5 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Tray Lifting
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id ri0w9j39 \
  --load-ema True \
  --checkpoint_step step_20000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-vanilla-fpo-tray-base-uul2gclip25-dec25-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmLiftTray --eval_env TwoArmLiftTray --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Threading
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-vanilla-fpo-threading-base-uul2gclip25-dec25-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

---

### 1.4 DPPO – Fixed Noise (15 runs)

**Model-specific parameters:** `loss_mode=dppo`, fixed `sde_sigma` (no learned noise), `cfm_loss_average_group_size=1`, `cfm_loss_use_huber=True`

| Parameter | Can | Square | Box Clearance | Tray Lifting | Threading |
|---|---|---|---|---|---|
| clip_coef | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 |
| max_grad_norm | 25 | 25 | 25 | 5 | 5 |
| cfm_loss_huber_delta | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 |
| sde_sigma | 0.3 | 0.3 | 0.24 | 0.24 | 0.24 |

#### Can
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_1000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-can-base-uul2gclip25-dec28-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=25 \
  --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.3 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Square
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-square-base-uul2gclip25-dec28-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=25 \
  --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.3 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Box Clearance
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id lainyisy \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-boxcleaning-base-uul2gclip25-dec28-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=25 \
  --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.24 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Tray Lifting
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id ri0w9j39 \
  --load-ema True \
  --checkpoint_step step_20000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-tray-base-uul2gclip25-dec28-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmLiftTray --eval_env TwoArmLiftTray --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=5 \
  --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.24 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

#### Threading
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-threading-base-uul2gclip25-dec28-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=5 \
  --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.24 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

---

## 2. Base Policy Checkpoint Ablation

All on **Can** task with base policy `95j3noe4` at **step_6000** (vs step_1000 in main benchmark). 3 seeds each.

### Cross-model comparison

| Parameter | FPO++ | Vanilla FPO | DPPO Learned | DPPO Fixed |
|---|---|---|---|---|
| loss_mode | fpo (default) | fpo (default) | dppo | dppo |
| cfm_loss_average_group_size | 1 | -1 | 1 | 1 |
| cfm_loss_use_huber | True | False | True | True |
| clamp_logratio | 5 | None | 5 | 5 |
| clamp_old_cfm_loss | 4 | None | 4 | 4 |
| clip_coef | 0.02 | 0.02 | 0.01 | 0.01 |
| max_grad_norm | 5 | 5 | 1 | 5 |
| sde_sigma | 0 | 0 | 0.18 | 0.3 |
| learn_sde_sigma | - | - | True | False |
| noise_injection_min/max | - | - | 0.2/0.5 | - |

### FPO++ (runs: txf5crib, ylxhw9uu, 6zp46k32)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_6000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-can-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=5 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

### Vanilla FPO (runs: gzm22fqh, opygcs30, 89jg66ll)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_6000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-vanilla-fpo-can-base-uul2gclip25-dec25-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.02 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=5 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

### DPPO – Learned Noise (runs: 7lz0maky, ss13djab, nqkx8w0s)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_6000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-can-base-uul2gclip25-dec28-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --learn_sde_sigma=True --loss_mode=dppo --max_grad_norm=1 \
  --n_action_samples=8 --n_action_steps=16 \
  --noise_injection_max=0.5 --noise_injection_min=0.2 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.18 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

### DPPO – Fixed Noise (runs: 4iz08o2h, 0wo72noe, 0yibse8t)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_6000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-dppo-can-base-uul2gclip25-dec28-v1 \
  --total_timesteps 5000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Can --eval_env Can --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.99 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --loss_mode=dppo --max_grad_norm=5 \
  --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0.3 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

---

## 3. FPO Ablation Study

Square and Threading tasks. 3 models: FPO++, ASPO, Per-action ratio. 3 seeds each.

### Key ablation differences

| Parameter | FPO++ | ASPO | Per-action ratio |
|---|---|---|---|
| trust_region_mode | ppo | **aspo** | ppo |
| cfm_loss_average_group_size | 1 | 1 | **-1** |
| clamp_logratio | None (Sq) / 5 (Th) | 5 | 5 |
| clamp_old_cfm_loss | None (Sq) / 4 (Th) | 4 | 4 |

All other hyperparameters are identical within each task.

### Square — FPO++ (runs: z2u9ryms, wzyv707a, rsmunbo4)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-square-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
  --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

### Square — ASPO (runs: whhhg3oc, safs53cm, 069ligh5)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-square-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=aspo --zero_sampling=True
```

### Square — Per-action ratio (runs: eikmrvae, kf0ywqzj, 4x5cxo0w)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id trc7rbt0 \
  --load-ema True \
  --checkpoint_step step_110000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-square-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task Square --eval_env Square --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.995 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=25 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

### Threading — FPO++ (runs: bt3sl4ex, fu8wpmbd, g2ldgoss)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-threading-base-uul2gclip25-dec25-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=1 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```

### Threading — ASPO (runs: 4pwr3dzu, ir471vi8, nx0ktwru)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-threading-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=1 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=aspo --zero_sampling=True
```

### Threading — Per-action ratio (runs: l5hdijzn, pcu57hf1, xmklzaca)
```bash
python finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 6vqrn614 \
  --load-ema True \
  --checkpoint_step step_10000 \
  --wandb_project flow-bc-fpo-finetuning \
  --experiment finetune-fpo-ablation-threading-base-uul2gclip25-jan1-v1 \
  --total_timesteps 8000000 \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --task TwoArmThreading --eval_env TwoArmThreading --eval_num_episodes 200 \
  --wandb_enable True \
  --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
  --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
  --data_collection_steps=1600 --discount=0.999 --do_chunk_level_ppo=True \
  --eval_ema=False --exploration_noise_std=None --freeze_vision_encoder=True \
  --gae_lambda=0.99 --max_grad_norm=1 --n_action_samples=8 --n_action_steps=16 \
  --num_envs=30 --sampling_steps=10 --sde_sigma=0 --seed=$SEED \
  --spo_clip_coef=0.01 --trust_region_mode=ppo --zero_sampling=True
```
