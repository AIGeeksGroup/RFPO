# FPO++ for Manipulation

This repository contains the code for the FPO++ for Manipulation. The main goal is to reproduce the results of the [FPO++ paper](https://arxiv.org/pdf/2602.02481) and provide robust baselines for future research.

This repository implements:

- Pretraining: Task-specific base policies trained via behavior cloning across five manipulation tasks.

- Fine-tuning: Four fine-tuning algorithms (FPO++, Vanilla FPO, and DPPO variants) adapted to flow matching so they can utilize the same base policy.

This README gives overview of the experiments and implementation. To reproduce the results from the paper, please refer to `docs/reproduce.md`.


## Directory Structure

```
.
├── docs/
│   └── reproduce.md                # Reproduction guide and plotting instructions
├── downloaded_checkpoints/         # Pretrained base policy checkpoints (download via gdown)
│   ├── README.md                   # Checkpoint details and usage instructions
│   ├── 95j3noe4_step_1000/        # Can (main benchmark)
│   ├── 95j3noe4_step_6000/        # Can (checkpoint ablation)
│   ├── trc7rbt0_step_110000/      # Square
│   ├── lainyisy_step_10000/       # Box Clearance
│   ├── ri0w9j39_step_20000/       # Tray Lifting
│   └── 6vqrn614_step_10000/       # Threading
├── scripts/
│   ├── eval_base_policies.sh       # Evaluate pretrained base policies
│   ├── run_checkpoint_ablation.sh  # Checkpoint ablation experiments
│   ├── run_fpo_ablation.sh         # FPO ablation experiments
│   ├── run_main_benchmark.sh       # Main benchmark experiments
│   ├── run_pretrain_base_policies.sh  # Pretrain base policies
│   ├── skypilot/                   # SkyPilot launch scripts for cloud execution
│   └── sweeps/                     # W&B sweep configs (pretrain & finetune) used for the main benchmark experiments in the paper
├── src/
│   ├── dexmg_env.py                # DexMimicGen environment wrapper
│   ├── flow_model.py               # Flow matching model
│   ├── flow_model_config.py        # Flow model configuration
│   ├── flow_net_mlp.py             # MLP flow network
│   ├── flow_net_residual_mlp.py    # Residual MLP flow network
│   ├── flow_net_unet.py            # UNet flow network
│   ├── noise_injection_network.py  # Noise injection network (for DPPO)
│   ├── utils.py                    # Utility functions
│   └── vit.py                      # Vision transformer backbone
├── thirdparty/
│   ├── dexmimicgen/                # DexMimicGen environments
│   ├── lerobot/                    # LeRobot dataset utilities
│   ├── miniconda3/                 # Local conda installation
│   └── robosuite/                  # Robosuite simulator
├── pretrain_flow_bc.py             # Pretraining entry point
├── finetune_online_rl.py           # Finetuning entry point
├── eval_checkpoint.py              # Checkpoint evaluation script
├── plot_results.py                 # Results plotting script
├── setup_env.sh                    # Environment setup (first-time)
├── source_env.sh                   # Environment activation (every session)
└── pyproject.toml                  # Project configuration
```


## Setup Environment

Before running any commands, activate the conda environment:
```bash
bash setup_env.sh          # first-time setup (installs conda, deps, etc.)
source source_env.sh       # activate the environment (run every session)
```


## Pretrained Checkpoints

Pretrained base policy checkpoints for all 5 tasks are available on
[Google Drive](https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t?usp=sharing).
Download them with:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1vQ3Tv-mwNZIFipp5Bv0SQlfYfIhlf8_t -O downloaded_checkpoints
```

To finetune from a downloaded checkpoint, use `--base_policy_local_path`:

```bash
torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base_policy_local_path downloaded_checkpoints/95j3noe4_step_1000 \
  --load-ema True \
  --task Can --eval_env Can \
  ...
```

See [docs/reproduce.md](docs/reproduce.md) for full details on all checkpoints and finetuning commands.


## Training

### Pretraining Flow Matching Base Policies via Behavior Cloning

**Available (dataset, task) pairs:**  

- ankile/robomimic-ph-can-image PickPlaceCan
- ankile/robomimic-ph-square-image NutAssemblySquare
- ankile/dexmg-two-arm-box-cleanup TwoArmBoxCleanup
- ankile/dexmg-two-arm-lift-tray TwoArmLiftTray
- ankile/dexmg-two-arm-threading TwoArmThreading
    
**Example commands:**

```bash
# MLP pretraining
python pretrain_flow_bc.py  --dataset ankile/dexmg-two-arm-threading --policy flowmatching --network_architecture mlp --horizon 8 --n_action_steps 8 --sampling_steps 10 --image_observation_keys "agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image" --eval_env TwoArmThreading  --eval_num_envs 1 --eval_num_episodes 5 --log_freq 10 --save_freq 200  --rollout_freq 200 --steps 6000 --wandb_enable True --wandb_project flow-bc
# Resume Unet pretraining
python pretrain_flow_bc.py --dataset ankile/robomimic-ph-square-image --policy flowmatching --network_architecture unet  --batch_size 4 --num_workers 4  --experiment square-image-flow-bc-resume-v1 --log_freq 10 --save_freq 200  --rollout_freq 200 --steps 6000 --eval_env Square  --eval_num_envs 2 --eval_num_episodes 10 --wandb_enable True --wandb_project flow-bc --resume_run_id 8v0hrimb
```

<details>
<summary><b>Some important parameters you might want to change for custom experiments:</b></summary>

- network_architecture: "mlp" or "unet"
- image_observation_keys: image observation keys to use for policy input (e.g., ["agentview_image", "robot0_eye_in_hand_image"]). If None, uses all image keys from dataset.
- horizon: prediction horizon
- n_action_steps: number of action steps to execute
- sampling_steps: number of sampling steps for flow model
- ema_power: EMA power for flow matching (0.0 means no EMA)
- grad_clip_norm: gradient clip norm
- batch_size: batch size
- num_workers: number of workers for data loading
- learning_rate: learning rate
- lr_backbone: learning rate for vision backbone
- weight_decay: weight decay
- flow_network_output_param: flow network output: "u" for velocity, "x" for data
- cfm_loss_mode: CFM loss mode: "u" for velocity, "x" for data, "eps" for epsilon
- cfm_loss_use_huber: whether to use Huber loss for CFM loss
- cfm_loss_huber_delta: Huber loss delta for CFM loss
- mlp_dims: hidden dimensions for MLP layers (only used if network_architecture="mlp")
- vision_backbone: vision backbone
- enable_geometric_augmentations: whether to enable geometric augmentations

</details>



### Finetuning Base Policies via Online RL

**Available finetuning algorithms:**
- FPO++
- Vanilla FPO
- DPPO with learned noise injection
- DPPO with fixed noise injection

The implementations of FPO++ and Vanilla FPO in this manipulation repository differ in one critical way, alongside a few minor details. The Vanilla FPO implementation aims to follow the original FPO design as closely as possible.

- Ratio Calculation: FPO++ uses a per-sample PPO ratio, whereas Vanilla FPO uses a per-action ratio.
- Trust Region Mode: Both use PPO trust region mode, as using ASPO in the fine-tuning setting hurts performance.

If you want more details about finetuning algorithms, please refer to the [FPO++ paper](https://arxiv.org/pdf/2602.02481) and `docs/reproduce.md`.

**Example commands:**

These are just example commands and in online RL, number of environments, discount factor, etc. should be tuned for each task.

```bash
# FPO++
python finetune_online_rl.py --base-policy-wandb-project flow-bc --base_policy_wandb_run_id wd9xdji9  --wandb_enable True --wandb_project flow-bc-fpo-finetuning --experiment finetune-fpo-can-image-v1  --log_freq 10 --save_freq 10  --rollout_freq 10  --eval_env Can  --eval_num_envs 5 --eval_num_episodes 10 --num_envs 4  --n_action_steps 4 --task Can --load-ema True --data-collection-steps 300
```


### Change Task-specific Horizon for Finetuning

Change this in `src/dexmg_env.py`
```python
self.horizon = {
            "TwoArmCoffee": 400,
            "TwoArmBoxCleanup": 300,
            "Lift": 100,
            "PickPlaceCan": 300,
            "NutAssemblySquare": 300,
        }.get(env_name, 1000)
```



## Evaluating Checkpoints and Plotting Results

### Evaluating Checkpoints

Use `eval_checkpoint.py` to evaluate any checkpoint (pretrained or finetuned) by its W&B run ID.

**Evaluating a pretrained base policy:**
```bash
python eval_checkpoint.py \
  --wandb_run_id 95j3noe4 \
  --wandb_project flow-bc \
  --checkpoint_step step_1000 \
  --eval_env Can \
  --eval_num_episodes 200 \
  --eval-num-envs 10 \
  --load-ema True
```

**Evaluating a finetuned policy:**
```bash
python eval_checkpoint.py \
  --wandb_run_id wbxzw7z3 \
  --wandb_project flow-bc-fpo-finetuning \
  --checkpoint_step best \
  --eval_env Can \
  --eval_num_episodes 200 \
  --eval-num-envs 10
```

**Key arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--wandb_run_id` | W&B run ID to download checkpoint from | -- |
| `--wandb_project` | W&B project name (`flow-bc` for pretrained, `flow-bc-fpo-finetuning` for finetuned) | -- |
| `--checkpoint_step` | Which checkpoint to evaluate: `latest`, `best`, or a specific step (e.g., `step_3000`) | `latest` |
| `--local_checkpoint_path` | Local path to checkpoint directory (alternative to W&B) | -- |
| `--load-ema` | Load EMA weights (use for pretrained base policies) | `False` |
| `--eval_env` | Environment name (e.g., `Can`, `Square`, `TwoArmBoxCleanup`, `TwoArmLiftTray`, `TwoArmThreading`) | `Lift` |
| `--eval_num_episodes` | Number of evaluation episodes | 50 |
| `--eval-num-envs` | Number of parallel environments | 2 |
| `--zero-sampling` | Use deterministic (zero) sampling; set to `False` for stochastic | `True` |

**Video rendering options:**

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `eval_camera_size` | Image observation size for policy input (what the policy sees) | 84 |
| `render_size` | Resolution (height, width) for saved rollout video frames | (240, 320) |

```bash
# Save video at 480x640 resolution
python eval_checkpoint.py --wandb_run_id wbxzw7z3 --wandb_project flow-bc-fpo-finetuning --eval_env Can --render_size 480 640

# Save video without text annotations (env idx, episode number, success/fail status)
python eval_checkpoint.py --wandb_run_id wbxzw7z3 --wandb_project flow-bc-fpo-finetuning --eval_env Can --annotate_video False
```

### Plotting Results

See [docs/reproduce.md](docs/reproduce.md) for plotting instructions using `plot_results.py`.




