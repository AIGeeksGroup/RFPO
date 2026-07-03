#!/usr/bin/env python3

"""Train an agent from states."""
import os
import weakref
import hydra
import torch
import wandb
import yaml
import cv2
import numpy as np
import _init_paths
import gym
import torch.nn.functional as F
from PIL import Image

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn

from algos import ActorCriticPolicy
from algos.rl.fpo_trainer import FPOStateTrainer
from hand_imitation.env.create_env import create_env
from hand_imitation.env.gym_wrapper import GymWrapper
from hand_imitation.utils.util import make_env, make_policy_kwargs, InfoCallback, FallbackCheckpoint
from hand_imitation.utils.eval import make_eval_env, EvalCallback
from hand_imitation.utils.fpo_eval import FPOEvalCallback


_PPO_BC_ANCHORS = weakref.WeakKeyDictionary()


def create_wandb_run(output_dir, wandb_cfg, job_config, run_id=None):
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f'{wandb_cfg.sweep_name_prefix}-{job_id}'
        notes = f'{override_dirname}'
    except:
        name, notes = None, None

    return wandb.init(project=wandb_cfg.project, dir=output_dir, config=job_config, group=wandb_cfg.group, sync_tensorboard=True, monitor_gym=True, save_code=True, name=name, notes=notes, id=run_id, resume=run_id is not None)


def resolve_resume_model(output_dir, requested_resume_model, agent_name):
    if requested_resume_model and os.path.isabs(requested_resume_model):
        return requested_resume_model
    default_name = 'restore_checkpoint.pt' if agent_name == 'FPO' else 'restore_checkpoint.zip'
    default_path = os.path.join(output_dir, default_name)
    if os.path.exists(default_path):
        return default_name
    return requested_resume_model


def rebuild_ppo_rollout_buffer(model):
    buffer_cls = DictRolloutBuffer if isinstance(model.observation_space, gym.spaces.Dict) else RolloutBuffer
    model.rollout_buffer = buffer_cls(
        model.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )


def apply_ppo_runtime_params(model, cfg, n_steps):
    params = cfg.agent.params
    model.n_steps = int(n_steps)
    model.gamma = float(params.gamma)
    model.gae_lambda = float(params.gae_lambda)
    model.learning_rate = float(params.learning_rate)
    model.ent_coef = float(params.ent_coef)
    model.vf_coef = float(params.vf_coef)
    model.clip_range = get_schedule_fn(float(params.clip_range))
    model.batch_size = int(params.batch_size)
    model.n_epochs = int(params.n_epochs)
    model._setup_lr_schedule()
    lr = model.lr_schedule(model._current_progress_remaining)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr


def _ppo_action_mean(policy, obs):
    latent_pi, _, _ = policy._get_latent(obs)
    return policy.action_net(latent_pi)


def ensure_ppo_bc_anchor_patch():
    if getattr(PPO, "_vividex_bc_anchor_patch", False):
        return
    base_train = PPO.train

    def train_with_optional_bc_anchor(self):
        base_train(self)
        state = _PPO_BC_ANCHORS.get(self)
        if state is None:
            return
        if state["decay_steps"] > 0:
            frac = max(0.0, 1.0 - float(self.num_timesteps) / state["decay_steps"])
            coef = state["coef_min"] + (state["coef_start"] - state["coef_min"]) * frac
        else:
            coef = state["coef_start"]
        if coef <= 0.0:
            return

        obs_tensor = state["observations"]
        action_tensor = state["actions"]
        last_raw_loss = None
        for _ in range(state["updates"]):
            idx = torch.randint(0, obs_tensor.shape[0], (state["batch_size"],), device=state["device"])
            pred = _ppo_action_mean(self.policy, obs_tensor[idx])
            target = action_tensor[idx]
            if state["loss_mode"] == "mse":
                raw_loss = F.mse_loss(pred, target)
            elif state["loss_mode"] == "l1":
                raw_loss = F.l1_loss(pred, target)
            else:
                raw_loss = F.huber_loss(pred, target, reduction="mean", delta=state["huber_delta"])
            loss = raw_loss * coef
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if state["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), state["grad_clip"])
            self.policy.optimizer.step()
            last_raw_loss = raw_loss

        if last_raw_loss is not None:
            self.logger.record("train/bc_anchor_loss", float(last_raw_loss.detach().cpu().item()))
            self.logger.record("train/bc_anchor_coef", coef)
            self.logger.record("train/bc_anchor_samples", int(obs_tensor.shape[0]))

    PPO.train = train_with_optional_bc_anchor
    PPO._vividex_bc_anchor_patch = True


def attach_ppo_bc_anchor(model, cfg):
    params = cfg.agent.params
    dataset = getattr(params, "bc_anchor_dataset", None)
    coef_start = float(getattr(params, "bc_anchor_coef", 0.0))
    if not dataset or coef_start <= 0.0:
        return

    ensure_ppo_bc_anchor_patch()
    data = np.load(os.path.expanduser(str(dataset)))
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    finite = np.isfinite(observations).all(axis=1) & np.isfinite(actions).all(axis=1)
    observations = observations[finite]
    actions = actions[finite]
    if observations.shape[1:] != model.observation_space.shape:
        raise ValueError(f"BC anchor obs shape {observations.shape[1:]} != env {model.observation_space.shape}")
    if actions.shape[1:] != model.action_space.shape:
        raise ValueError(f"BC anchor action shape {actions.shape[1:]} != env {model.action_space.shape}")

    device = model.device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
    _PPO_BC_ANCHORS[model] = {
        "observations": obs_tensor,
        "actions": action_tensor,
        "batch_size": min(int(getattr(params, "bc_anchor_batch_size", 256)), int(obs_tensor.shape[0])),
        "updates": max(int(getattr(params, "bc_anchor_updates", 1)), 1),
        "coef_start": coef_start,
        "coef_min": float(getattr(params, "bc_anchor_min_coef", 0.0)),
        "decay_steps": float(getattr(params, "bc_anchor_decay_steps", 500000)),
        "loss_mode": str(getattr(params, "bc_anchor_loss", "huber")),
        "huber_delta": float(getattr(params, "bc_anchor_huber_delta", 0.05)),
        "grad_clip": float(getattr(params, "bc_anchor_grad_clip", 1.0)),
        "device": device,
    }


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, '../algos/rl/config')
@hydra.main(config_path=cfg_path, config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig, resume_model=None):
    # logging
    cfg_yaml = OmegaConf.to_yaml(cfg)
    resume_model = cfg.resume_model
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if os.path.exists(os.path.join(output_dir, 'exp_config.yaml')):
        old_config = yaml.safe_load(open(os.path.join(output_dir, 'exp_config.yaml'), 'r'))
        params, wandb_id = old_config['params'], old_config['wandb_id']
        run = create_wandb_run(output_dir, cfg.wandb, params, wandb_id)
        old_agent_name = params.get('agent', {}).get('name', cfg.agent.name)
        resume_model = resolve_resume_model(output_dir, resume_model, old_agent_name)
        if resume_model:
            resume_path = resume_model if os.path.isabs(resume_model) else os.path.join(output_dir, resume_model)
            assert os.path.exists(resume_path), f'{resume_model} does not exist!'
    else:
        defaults = HydraConfig.get().runtime.choices
        params = yaml.safe_load(cfg_yaml)
        params['defaults'] = {k: defaults[k] for k in ('agent', 'env')}

        run = create_wandb_run(output_dir, cfg.wandb, params)
        save_dict = dict(wandb_id=run.id, params=params)
        yaml.dump(save_dict, open(os.path.join(output_dir, 'exp_config.yaml'), 'w'))
        print('Config:')
        print(cfg_yaml)
    
    # env = create_env(name=cfg.env.name, task_kwargs=cfg.env.task_kwargs, is_eval=True)
    if cfg.agent.name == 'PPO':
        # Construct the env
        total_timesteps = cfg.total_timesteps
        eval_freq = int(cfg.eval_freq // cfg.n_envs)
        save_freq = int(cfg.save_freq // cfg.n_envs)
        restore_freq = int(cfg.restore_checkpoint_freq // cfg.n_envs)
        n_steps = int(cfg.agent.params.n_steps // cfg.n_envs)
        multi_proc = bool(cfg.agent.multi_proc)
        env = make_env(multi_proc=multi_proc, is_eval=False, **cfg.env)
        initial_stage = max(0, min(2, int(getattr(cfg.agent.params, "initial_curriculum_stage", 0))))
        if initial_stage > 0:
            env.env_method('curriculum', initial_stage)

        if resume_model:
            model = PPO.load(resume_model, env)
            model.set_env(env)
            apply_ppo_runtime_params(model, cfg, n_steps)
            rebuild_ppo_rollout_buffer(model)
            model._last_obs = None
        else:
            model = PPO(
                            ActorCriticPolicy, 
                            env, verbose=1, 
                            tensorboard_log=f"{output_dir}/tb_logs/", 
                            n_steps=n_steps, 
                            gamma=cfg.agent.params.gamma,
                            gae_lambda=cfg.agent.params.gae_lambda,
                            learning_rate=cfg.agent.params.learning_rate,
                            ent_coef=cfg.agent.params.ent_coef,
                            vf_coef=cfg.agent.params.vf_coef,
                            clip_range=cfg.agent.params.clip_range,
                            batch_size=cfg.agent.params.batch_size,
                            n_epochs=cfg.agent.params.n_epochs,
                            policy_kwargs=make_policy_kwargs(cfg.agent.policy_kwargs)
                        )
        attach_ppo_bc_anchor(model, cfg)
        
        # initialize callbacks and train
        eval_env = make_eval_env(multi_proc, cfg.n_eval_envs, **cfg.env)
        if initial_stage > 0:
            eval_env.env_method('curriculum', initial_stage)
        eval_deterministic = bool(getattr(cfg.agent.params, "eval_deterministic", False))
        save_best_model = bool(getattr(cfg.agent.params, "save_best_model", False))
        eval_callback = EvalCallback(
            output_dir,
            eval_freq,
            eval_env,
            n_eval_episodes=cfg.eval_n_episodes,
            deterministic=eval_deterministic,
            initial_stage=initial_stage,
            save_best_model=save_best_model,
        )
        restore_callback = FallbackCheckpoint(output_dir, restore_freq)
        log_info = InfoCallback()
        checkpoint = CheckpointCallback(save_freq=save_freq, save_path=f'{output_dir}/logs/', name_prefix='rl_models')
        wandb = WandbCallback(model_save_path=f"{output_dir}/models/", verbose=2)
        train_logger = configure(f'{output_dir}/logs/', ["stdout", "log"])
        model.set_logger(train_logger)
        return model.learn(total_timesteps=total_timesteps, callback=[log_info, eval_callback, checkpoint, restore_callback, wandb], reset_num_timesteps=True)
    elif cfg.agent.name == 'FPO':
        total_timesteps = cfg.total_timesteps
        eval_freq = int(cfg.eval_freq)
        save_freq = int(cfg.save_freq)
        restore_freq = int(cfg.restore_checkpoint_freq)
        multi_proc = bool(cfg.agent.multi_proc)
        env = make_env(multi_proc=multi_proc, is_eval=False, **cfg.env)

        model = FPOStateTrainer(env, cfg=cfg, output_dir=output_dir)
        if resume_model:
            resume_path = resume_model if os.path.isabs(resume_model) else os.path.join(output_dir, resume_model)
            resume_load_optimizer = bool(getattr(cfg.agent.params, "resume_load_optimizer", True))
            model.load(resume_path, load_optimizer=resume_load_optimizer)

        eval_env = make_eval_env(multi_proc, cfg.n_eval_envs, **cfg.env)
        model.sync_curriculum(eval_env)
        eval_deterministic = bool(getattr(cfg.agent.params, "eval_deterministic", False))
        eval_callback = FPOEvalCallback(
            output_dir,
            n_eval_episodes=cfg.eval_n_episodes,
            deterministic=eval_deterministic,
        )
        train_logger = configure(f'{output_dir}/logs/', ["stdout", "log"])
        model.set_logger(train_logger)
        return model.learn(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            eval_callback=eval_callback,
            eval_freq=eval_freq,
            save_freq=save_freq,
            restore_freq=restore_freq,
            wandb_run=run,
        )
    else:
        raise NotImplementedError
    wandb.finish()


if __name__ == '__main__':
    # Load the config
    train()
