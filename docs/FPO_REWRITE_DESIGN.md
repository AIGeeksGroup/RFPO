# ViViDex FPO Rewrite Design

## Goal

This repository rewrites the failed FPO experiment from a clean ViViDex copy. The target is not "PPO plus a flow-matching auxiliary loss". The target is to keep the ViViDex stage-2 state-RL protocol, rollout collection, GAE, critic learning, evaluation, and curriculum semantics, while replacing PPO's Gaussian policy log-probability ratio with the FPO/FPO++ CFM-loss ratio:

```text
ratio ~= exp(old_cfm_loss(action | obs) - new_cfm_loss(action | obs))
```

## What Stays From Original ViViDex

Original ViViDex stage-2 training is state-based online RL:

- `tools/train.py` builds the state environment through `make_env(...)`.
- The environment reward remains in `hand_imitation/env/rl_env/relocate_env.py`.
- Stage 0 uses fixed pose, stage 1 randomizes xy, stage 2 randomizes xy and z rotation.
- Curriculum advancement remains exactly PPO-compatible: advance while `mean_pregrasp_success > 0.95` and stage is 0 or 1.
- Evaluation still uses `make_eval_env(...)`.
- The action space remains ViViDex's normalized continuous Box action in `[-1, 1]`.

The rewrite deliberately does not change environment reward, reset logic, termination logic, object pose sampling, or imitation target construction.

## What Replaces PPO

PPO originally uses:

- Gaussian actor.
- `log_prob(new_action) - log_prob(old_action)`.
- PPO clipped surrogate.
- Value loss.

FPO now uses:

- A conditional flow-matching actor `FPOStatePolicy`.
- Zero-noise sampling for deterministic evaluation.
- Gaussian-noise sampling for online exploration.
- CFM loss samples `(eps, t)` stored at rollout time.
- Per-sample FPO++ ratio from stored old loss and current loss.
- The same PPO clipped surrogate form after replacing the ratio.
- A normal MLP critic over the same state observation.

The core files are:

- `algos/rl/fpo_core.py`: flow policy, CFM loss, FPO ratio/surrogate.
- `algos/rl/fpo_rollout_buffer.py`: rollout storage with CFM tensors.
- `algos/rl/fpo_trainer.py`: ViViDex on-policy training loop.
- `hand_imitation/utils/fpo_eval.py`: deterministic FPO evaluation callback.
- `algos/rl/config/agent/fpo.yaml`: default FPO hyperparameters.

## Why Flow-BC Is In The Main Path

The closest manipulation setting in the FPO paper is not a random flow policy trained from scratch. It is a pretrained flow behavior-cloning policy, followed by online FPO/FPO++ fine-tuning. The previous failed ViViDex FPO code trained a flow policy mostly from scratch and empirically failed already at pregrasp. This rewrite makes the expected path:

1. Collect successful state/action trajectories from a strong PPO policy.
2. Train a flow policy by CFM behavior cloning.
3. Fine-tune that flow policy online with FPO.

From-scratch FPO remains possible by leaving `agent.params.bc_checkpoint=null`, but it is not the recommended manipulation setup.

## Command Flow

Collect successful PPO trajectories:

```bash
cd /home/why/桌面/vividex_sapien_server_upload_fpo_control_full
python tools/collect_fpo_bc_dataset.py \
  -e /home/why/桌面/ppo_mustard_baseline \
  -o data/fpo_bc_mustard_stage2.npz \
  -n 200 \
  --stage 2 \
  --success_metric sr10
```

Pretrain the flow policy:

```bash
python tools/pretrain_fpo_bc.py \
  -d data/fpo_bc_mustard_stage2.npz \
  -o outputs/flow_bc_mustard \
  --steps 200000 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --actor_hidden_dims "[512, 512, 512]" \
  --activation relu \
  --sampling_steps 10 \
  --cfm_loss_use_huber
```

Fine-tune online with FPO:

```bash
python tools/train.py \
  agent=fpo \
  agent.params.bc_checkpoint=/home/why/桌面/vividex_sapien_server_upload_fpo_control_full/outputs/flow_bc_mustard/flow_bc_last.pt \
  total_timesteps=60000000 \
  n_envs=32 \
  n_eval_envs=5 \
  eval_freq=1000000
```

Evaluate:

```bash
python tools/evaluate_policy.py \
  -e <hydra-output-run-dir> \
  -n 100
```

Visualize:

```bash
python tools/visualize_policy.py \
  -e <hydra-output-run-dir>
```

## Important Implementation Details

The flow model trains in normalized observation and normalized action coordinates. `tools/pretrain_fpo_bc.py` computes and saves the normalizers; `FPOStateTrainer` reloads them from `agent.params.bc_checkpoint`.

During online rollout:

1. `policy.act(obs, deterministic=False)` starts from random noise and integrates the flow from `t=1` to `t=0`.
2. The output is denormalized and clipped to `[-1, 1]`.
3. The exact executed action is stored.
4. CFM samples `(eps, t)` and old CFM loss are stored with the transition.
5. GAE returns and advantages are computed after the rollout.

During update:

1. The same stored `(eps, t)` are reused.
2. Current CFM loss is recomputed under the updated actor.
3. Ratio is `exp(old_loss - current_loss)`.
4. PPO clipping is applied to that ratio.
5. The critic is updated with the same returns.

During evaluation:

1. `deterministic=True` means zero initial flow noise.
2. Curriculum advancement only checks `eval/mean_pregrasp_success > 0.95`.
3. No object-lift or reward threshold is used for curriculum advancement.

## Main Differences From The Deleted Attempt

- The repo was rebuilt from the official ViViDex code instead of continuing the old nested experiment directory.
- FPO core is isolated in small pure-PyTorch modules and tested without requiring SAPIEN or SB3.
- Curriculum no longer has extra object-lift/reward gates.
- Flow-BC pretraining is first-class instead of an afterthought.
- Observation/action normalizers are saved in checkpoints and reused during FPO fine-tuning.
- Evaluation uses zero-sampling, matching the FPO manipulation deployment setting.

## Verification

Pure FPO unit tests:

```bash
python3 -m unittest tests/test_fpo_core.py -v
```

Syntax check:

```bash
python3 -m py_compile \
  algos/rl/fpo_core.py \
  algos/rl/fpo_rollout_buffer.py \
  algos/rl/fpo_trainer.py \
  hand_imitation/utils/fpo_eval.py \
  tools/pretrain_fpo_bc.py \
  tools/collect_fpo_bc_dataset.py \
  tools/train.py \
  tools/evaluate_policy.py \
  tools/visualize_policy.py
```

Full simulator training still requires the original ViViDex environment with `gym`, `stable_baselines3`, `hydra`, `wandb`, `sapien`, and the project assets installed.
