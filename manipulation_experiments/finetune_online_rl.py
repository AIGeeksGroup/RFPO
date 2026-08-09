#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict, Any

import imageio
import numpy as np
import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from termcolor import colored
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import tyro
import wandb

from lerobot.common.policies.pretrained import PreTrainedPolicy
from src.dexmg_env import VectorizedEnvWrapper, create_vectorized_env
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig
from src.rollout_bookkeeping import build_rollout_zero_sampling_mask, prepare_invalid_step_mask
from src.replay_audit import effective_sample_fraction, successful_chunk_mask
from src.cfm_sampling import sample_cfm_variables
from src.stratified_mc_audit import (
    advantage_stratified_sample_counts,
    gradient_estimator_metrics,
)
from src.heldout_ratio_early_stop import select_epoch_before_ratio_violation
from src.discounted_success_critic import (
    centered_average_rank_scores,
    discounted_returns_to_observed_terminal,
    spearman_rank_correlation,
)
from src.direct_advantage import (
    DirectAdvantageHead,
    direct_advantage_residuals,
    discounted_macro_rewards,
)
from src.advantage_weighting import (
    clipped_mirror_ratio_loss,
    ess_softmax_weights,
    normalize_advantages_from_moments,
)
from src.bc_anchor_pcgrad import (
    gradient_cosine as list_gradient_cosine,
    gradient_norm,
    project_conflicting_gradient,
)
from src.temporal_ratio_clipping import (
    clipped_ratio_objective,
    positive_active_fraction,
)
from src.median_microbatch_gradient import (
    aggregate_microbatch_gradients,
    geometric_median,
    middle_pair_mean,
    stable_vector_cosine,
)
from src.terminal_consistency_filter import terminal_consistent_weights
from src.ratio_rollback import rollback_clipped_ratio_loss
from src.adaptive_lr import adapt_learning_rate_from_kl

# ---- Multiprocessing start method (CUDA compat) ------------------------------
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


@dataclass
class FlowPPOConfig:
    task: str = "Lift"
    loss_mode: Literal["fpo", "dppo"] = "fpo"
    observation_type: str = "image"
    num_envs: int = 2
    reset_every_iteration: bool = True
    truncation_as_done: bool = True
    camera_size: int = 84
    camera_name_to_vis: str = "agentview"

    # Base policy
    base_policy_wandb_project: Optional[str] = "flow-bc"
    base_policy_wandb_run_id: Optional[str] = None
    checkpoint_step: Optional[str] = "latest"
    base_policy_local_path: Optional[str] = None
    init_flow_network: bool = False
    load_ema: bool = True
    policy: str = "flowmatching"

    # Training
    total_timesteps: int = 1_000_000
    data_collection_steps: int = 96
    num_minibatches: int = 4
    update_epochs: int = 10
    gradient_accumulation_steps: int = 1
    n_iterations_train_only_value: int = 1

    # Policy overrides
    transported_clip_value: Optional[float] = None
    n_action_steps: Optional[int] = None
    sampling_steps: Optional[int] = None
    cfm_loss_use_huber: Optional[bool] = None
    cfm_loss_huber_delta: Optional[float] = None
    flow_network_output_param: Optional[Literal["u", "x0"]] = None
    cfm_loss_mode: Optional[Literal["u", "x0", "eps"]] = None
    image_observation_keys: Optional[str] = None  # space-separated -> list later

    # FPO
    freeze_vision_encoder: bool = True
    do_chunk_level_ppo: bool = True
    do_average_cfm_loss_in_chunk: bool = False
    n_action_samples: int = 16
    clamp_old_cfm_loss: Optional[float] = None
    cfm_loss_average_group_size: int = 1
    trust_region_mode: Literal["ppo", "spo", "aspo", "rollback"] = "ppo"
    rollback_alpha: float = 0.3
    clamp_logratio: Optional[float] = None
    cfm_loss_weight_from_t: str = "constant"
    advantage_weighting: Literal["signed", "ess_softmax"] = "signed"
    advantage_weight_ess_fraction: float = 0.5
    advantage_normalization_scope: Literal["minibatch", "rollout"] = "minibatch"
    reset_cfm_invalid_mask_each_iteration: bool = False
    success_replay_audit: bool = False
    success_replay_audit_iteration: int = 2
    success_replay_audit_chunks: int = 64
    success_replay_audit_min_chunks: int = 32
    success_replay_audit_output_json: Optional[str] = None
    cfm_ratio_generalization_audit: bool = False
    cfm_ratio_generalization_audit_iteration: int = 2
    cfm_ratio_generalization_audit_chunks: int = 64
    cfm_ratio_generalization_audit_min_chunks: int = 32
    cfm_ratio_generalization_audit_output_json: Optional[str] = None
    advantage_stratified_mc_audit: bool = False
    advantage_stratified_mc_audit_iteration: int = 2
    advantage_stratified_mc_audit_batches: int = 2
    advantage_stratified_mc_audit_batch_size: int = 32
    advantage_stratified_mc_audit_repeats: int = 8
    advantage_stratified_mc_audit_reference_samples: int = 64
    advantage_stratified_mc_audit_uniform_samples: int = 8
    advantage_stratified_mc_audit_low_samples: int = 4
    advantage_stratified_mc_audit_high_samples: int = 12
    advantage_stratified_mc_audit_output_json: Optional[str] = None
    heldout_ratio_early_stop_audit: bool = False
    heldout_ratio_early_stop_audit_iteration: int = 2
    heldout_ratio_early_stop_audit_chunks: int = 64
    heldout_ratio_early_stop_threshold: float = 0.8
    heldout_ratio_early_stop_audit_output_json: Optional[str] = None
    discounted_success_critic_audit: bool = False
    discounted_success_critic_audit_iteration: int = 2
    discounted_success_critic_audit_chunks: int = 64
    discounted_success_critic_audit_output_json: Optional[str] = None
    critic_warmup_scheduler_audit: bool = False
    critic_warmup_scheduler_audit_iteration: int = 2
    critic_warmup_scheduler_audit_chunks: int = 64
    critic_warmup_scheduler_audit_output_json: Optional[str] = None
    direct_advantage_audit: bool = False
    direct_advantage_audit_iteration: int = 2
    direct_advantage_audit_chunks: int = 64
    direct_advantage_center_samples: int = 4
    direct_advantage_horizon_chunks: int = 4
    direct_advantage_center_batch_size: int = 16
    direct_advantage_audit_output_json: Optional[str] = None
    rank_advantage_audit: bool = False
    rank_advantage_audit_iteration: int = 2
    rank_advantage_audit_chunks: int = 64
    rank_advantage_audit_output_json: Optional[str] = None
    advantage_sign_pcgrad_audit: bool = False
    advantage_sign_pcgrad_audit_iteration: int = 2
    advantage_sign_pcgrad_audit_chunks: int = 64
    advantage_sign_pcgrad_min_per_sign: int = 8
    advantage_sign_pcgrad_audit_output_json: Optional[str] = None
    temporal_ratio_clipping_audit: bool = False
    temporal_ratio_clipping_audit_iteration: int = 2
    temporal_ratio_clipping_audit_chunks: int = 64
    temporal_ratio_clipping_audit_output_json: Optional[str] = None
    median_microbatch_gradient_audit: bool = False
    median_microbatch_gradient_audit_iteration: int = 2
    median_microbatch_gradient_audit_chunks: int = 192
    median_microbatch_gradient_method: Literal["coordinate", "geometric"] = "coordinate"
    median_microbatch_gradient_audit_output_json: Optional[str] = None
    terminal_consistency_filter_audit: bool = False
    terminal_consistency_filter_audit_iteration: int = 2
    terminal_consistency_filter_audit_chunks: int = 128
    terminal_consistency_filter_audit_output_json: Optional[str] = None
    bc_anchor_pcgrad_audit: bool = False
    bc_anchor_pcgrad_audit_iteration: int = 2
    bc_anchor_pcgrad_audit_chunks: int = 64
    bc_anchor_pcgrad_anchor_mode: Literal["velocity", "zero_endpoint"] = "velocity"
    bc_anchor_pcgrad_audit_output_json: Optional[str] = None
    zero_endpoint_pcgrad_train: bool = False
    rollout_zero_fraction: float = 0.0
    rollout_tempered_fraction: float = 0.0
    rollout_tempered_scale: float = 0.5
    exploration_noise_std: Optional[float] = None
    zero_sampling: bool = True  # deprecated. now we evaluate with both zero and non zero sampling
    save_non_zero_sampling_video: bool = False

    # DPPO
    sde_sigma: float = 0.08
    learn_sde_sigma: bool = False
    noise_injection_min: float = 0.2
    noise_injection_max: float = 0.5
    entropy_loss_coef: float = 0.001
    dppo_norm_factor: float = 1.0
    average_logprob_over_denoising_steps: bool = False # something like reinflow style

    # PPO-ish
    discount: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.01
    spo_clip_coef: float = 0.01
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.1
    adaptive_actor_lr: bool = False
    adaptive_actor_lr_target_kl: float = 1e-4

    # LRs
    learning_rate_actor: float = 1e-5
    learning_rate_critic: float = 1e-4
    optimizer_betas_actor: list[float] = field(default_factory=lambda: [0.9, 0.99])

    # Schedulers
    lr_scheduler_name: str = "constant"
    lr_scheduler_actor_warmup_steps: int = 5
    lr_scheduler_critic_warmup_steps: int = 1

    # Eval
    rollout_freq: Optional[int] = 10
    eval_env: Optional[str] = None
    eval_num_episodes: int = 10
    eval_camera_size: int = 84
    eval_ema: bool = False

    # System
    seed: Optional[int] = None
    torch_deterministic: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_freq: int = 10
    save_freq: int = 100
    debug: bool = False
    distributed: bool = False

    # W&B
    wandb_enable: bool = True
    wandb_project: Optional[str] = "flow-bc-fpo-finetuning"
    wandb_entity: str = "far-wandb"
    wandb_notes: Optional[str] = None
    wandb_continue_run_id: Optional[str] = None
    experiment: str = "finetune_fpo"

    # Run
    output_dir: Optional[str] = None


# ---- Helper modules ----------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, global_obs_dim: int = 1033):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(global_obs)


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + discount * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns


def clamp_ste(x, min=None, max=None):
    clamped = x.clamp(min=min, max=max)
    return x + (clamped - x).detach()


def _annotate_frame(
    frame: np.ndarray,
    env_idx: int,
    episode_num: int,
    total_episodes: int,
    episode_step: int,
    is_success: bool,
    font=None,
) -> np.ndarray:
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)
    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font); y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font); y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)
    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    save_dir: Path,
    global_step: int,
    num_episodes: int,
    task: str,
    zero_sampling: bool,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    was_training = policy.training
    policy.eval()

    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")
    logger.info(f"Running rollouts with environment: {env_name}")
    logger.info(f"Starting evaluation: {num_episodes} episodes using {num_parallel_envs} parallel environments")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_path = save_dir / f"eval_step_{global_step}_{now}.mp4"
    video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    obs, _ = env.reset()
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    policy.reset()

    successes_list = []
    done_episodes_list = []
    total_steps = 0
    t0 = time.perf_counter()

    while sum(done_episodes_list) < num_episodes:
        with torch.inference_mode():
            action, mdp_x_t_path = policy.select_action(obs, zero_sampling=zero_sampling, sde_sampling=False)

        obs, reward, terminated, truncated, info = env.step(action)
        frames = env.render()

        for env_idx in range(num_parallel_envs):
            episode_frames[env_idx].append(frames[env_idx])
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs
        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]
            policy.reset(env_ids=terminated_envs)

            for env_idx_tensor in terminated_envs:
                env_idx = int(env_idx_tensor.item())
                is_success = env_idx_tensor in success_envs
                done_episodes_list.append(1)
                successes_list.append(int(is_success))

                # discard the frames from the terminated environments since it is a new observation on the reseted new environment
                episode_frames[env_idx].pop(-1)
                episode_steps[env_idx] -= 1

                for step_idx, frame in enumerate(episode_frames[env_idx]):
                    annotated_frame = _annotate_frame(
                        frame=frame,
                        env_idx=env_idx,
                        episode_num=sum(done_episodes_list),
                        total_episodes=num_episodes,
                        episode_step=step_idx + 1,
                        is_success=is_success,
                        font=font,
                    )
                    video_writer.append_data(annotated_frame)

                episode_frames[env_idx] = []
                episode_steps[env_idx] = 0

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {sum(done_episodes_list)}, successes: {sum(successes_list)}, "
                f"FPS: {total_steps / (time.perf_counter() - t0):.1f}"
            )
    
    # trim the tails
    successes = sum(successes_list[:num_episodes])
    done_episodes = sum(done_episodes_list[:num_episodes])

    video_writer.close()
    success_rate = successes / done_episodes if done_episodes > 0 else 0.0

    if was_training:
        policy.train()

    elapsed = time.perf_counter() - t0
    fps = total_steps / elapsed if elapsed > 0 else 0.0
    eps_sec = done_episodes / elapsed if elapsed > 0 else 0.0

    logger.info(f"Evaluation completed: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%)")
    logger.info(f"Performance: {total_steps} total steps in {elapsed:.1f}s")
    logger.info(f"Average FPS: {fps:.1f} frames/sec | Episodes/sec: {eps_sec:.2f}")
    logger.info(f"Video saved: {video_path}")

    return success_rate, video_path, fps, int(successes), int(done_episodes)

def eval_all_ranks(
    *,
    env: VectorizedEnvWrapper,
    num_envs_per_process,
    actor_ddp_or_single,
    world_size: int,
    rank: int,
    is_ddp: bool,
    device_str: str,
    device,
    run_dir: Path,
    cfg: FlowPPOConfig,
    create_env_fn,
    global_step: int,
):
    """Run evaluation on every rank, reduce results, log from rank 0 only."""
    # Unwrap DDP for deepcopy
    base_actor = actor_ddp_or_single.module if is_ddp else actor_ddp_or_single
    eval_actor = copy.deepcopy(base_actor).to(device)
    eval_actor.eval()

    # Optional EMA for eval
    if cfg.eval_ema and hasattr(eval_actor, "enable_ema"):
        eval_actor.enable_ema()

    # Split total episodes across ranks (as balanced as possible)
    total_eps = cfg.eval_num_episodes
    base = total_eps // world_size
    rem = total_eps % world_size
    eps_this_rank = base + (1 if rank < rem else 0)
    if eps_this_rank == 0:
        eps_this_rank = 0  # some ranks may skip if episodes < world_size

    # Each rank can also choose its own num_envs for eval (keep small)
    num_envs_eval_rank = num_envs_per_process #min(cfg.eval_num_envs, max(1, eps_this_rank)) if eps_this_rank > 0 else 1

    # Buffers for local results
    local_successes = 0
    local_episodes  = 0
    local_fps_sum   = 0.0  # (optional) aggregate FPS across ranks
    local_video_path = None

    if eps_this_rank > 0:
        # Only rank 0 records a video to avoid N videos
        save_video = (rank == 0)
        eval_actor.init_action_buffers(num_envs_eval_rank)
        # Eval with zero sampling
        sr, video_path, fps, succ, eps = _run_rollouts(
            policy=eval_actor,
            env=env,
            save_dir=(run_dir / "videos") if save_video else (run_dir / "_scratch_no_video"),
            global_step=global_step,
            num_episodes=eps_this_rank,
            task=cfg.eval_env,
            zero_sampling=True,
        )
        local_successes = succ
        local_episodes  = eps
        local_fps_sum   = fps * eps  # weight fps by episodes
        local_video_path = video_path if save_video else None

        # Eval with non zero sampling
        sr_non_zero, video_path_non_zero, fps_non_zero, succ_non_zero, eps_non_zero = _run_rollouts(
            policy=eval_actor,
            env=env,
            save_dir=(run_dir / "videos") if save_video else (run_dir / "_scratch_no_video"),
            global_step=global_step,
            num_episodes=eps_this_rank,
            task=cfg.eval_env,
            zero_sampling=False,
        )
        local_successes_non_zero = succ_non_zero
        local_episodes_non_zero = eps_non_zero
        local_fps_sum_non_zero = fps_non_zero * eps_non_zero
        local_video_path_non_zero = video_path_non_zero if save_video else None


    # Reduce across ranks
    if is_ddp:
        t = torch.tensor(
            [local_successes, local_episodes, local_fps_sum, local_successes_non_zero, local_episodes_non_zero, local_fps_sum_non_zero],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        global_successes, global_episodes, global_fps_sum, global_successes_non_zero, global_episodes_non_zero, global_fps_sum_non_zero = t.tolist()
    else:
        global_successes, global_episodes, global_fps_sum, global_successes_non_zero, global_episodes_non_zero, global_fps_sum_non_zero = (
            float(local_successes), float(local_episodes), float(local_fps_sum), float(local_successes_non_zero), float(local_episodes_non_zero), float(local_fps_sum_non_zero)
        )

    assert global_episodes == global_episodes_non_zero, "global_episodes and global_episodes_non_zero should be the same"


    # Compute global metrics on rank 0
    if rank == 0:
        global_sr = (global_successes / max(1.0, global_episodes))
        global_sr_non_zero = (global_successes_non_zero / max(1.0, global_episodes_non_zero))
        # episode-weighted average FPS (approx)
        global_fps = (global_fps_sum / max(1.0, global_episodes))
        global_fps_non_zero = (global_fps_sum_non_zero / max(1.0, global_episodes_non_zero))

        if cfg.save_non_zero_sampling_video:
            local_video_path = local_video_path_non_zero

        return {
            "success_rate": global_sr,
            "success_rate_non_zero": global_sr_non_zero,
            "fps": global_fps,
            "fps_non_zero": global_fps_non_zero,
            "episodes": int(global_episodes),
            "video_path": local_video_path,  # only rank 0 produced one
        }
    return None


def save_checkpoint(checkpoint_dir: Path, step: int, policy: PreTrainedPolicy, optimizer):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(checkpoint_dir / "policy"))
    checkpoint_data = {"step": step, "optimizer_state_dict": optimizer.state_dict()}
    if hasattr(policy, "ema_model") and getattr(policy, "ema_model", None) is not None:
        checkpoint_data["ema_state_dict"] = policy.ema_model.state_dict()
    torch.save(checkpoint_data, checkpoint_dir / "optimizer.pt")


# ---- W&B artifact pull (rank 0 only) ----------------------------------------
def download_checkpoint_from_wandb_rank0(
    run_id: str,
    project: str,
    entity: str,
    artifact_alias: str = "latest",
    download_dir: Path = Path("./downloaded_checkpoints"),
) -> Path:
    """Download a checkpoint artifact from wandb."""
    logger.info(colored(f"Downloading checkpoint from W&B run {run_id} (artifact: {artifact_alias})...", "cyan"))

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    logger.info(colored(f"Fetching artifacts from run {run_id}...", "cyan"))
    artifacts = run.logged_artifacts()

    if not artifacts:
        raise ValueError(f"No artifacts found for run {run_id}")

    logger.info(colored(f"Found {len(artifacts)} total artifacts in run {run_id}", "cyan"))

    # Filter for checkpoint artifacts only
    checkpoint_artifacts = [a for a in artifacts if "checkpoint" in a.name]

    if not checkpoint_artifacts:
        raise ValueError(f"No checkpoint artifacts found for run {run_id}")

    logger.info(colored(f"Found {len(checkpoint_artifacts)} checkpoint artifacts", "cyan"))

    def get_step_from_artifact(artifact):
        """Extract step number from artifact metadata or name."""
        if hasattr(artifact, 'metadata') and artifact.metadata and 'step' in artifact.metadata:
            return artifact.metadata['step']
        import re
        match = re.search(r'step_(\d+)', artifact.name)
        if match:
            return int(match.group(1))
        return 0

    # Find the right artifact based on alias
    artifact = None

    if artifact_alias == "latest":
        checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
        artifact = checkpoint_artifacts[0]
        logger.info(colored(f"Looking for latest checkpoint...", "cyan"))
    elif artifact_alias == "best":
        best_artifacts = [a for a in checkpoint_artifacts if "best" in a.name.lower()]
        if best_artifacts:
            artifact = best_artifacts[0]
            logger.info(colored(f"Found 'best' checkpoint", "cyan"))
        else:
            logger.warning(colored(f"No 'best' checkpoint found, falling back to latest", "yellow"))
            checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
            artifact = checkpoint_artifacts[0]
    else:
        # Try to find by specific step or name
        matching_artifacts = [a for a in checkpoint_artifacts if artifact_alias in a.name.lower()]
        if matching_artifacts:
            artifact = matching_artifacts[0]
            logger.info(colored(f"Found checkpoint matching '{artifact_alias}'", "cyan"))
        else:
            logger.warning(colored(f"No checkpoint matching '{artifact_alias}' found, falling back to latest", "yellow"))
            checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
            artifact = checkpoint_artifacts[0]

    step_num = get_step_from_artifact(artifact)
    logger.info(colored(f"Selected artifact: {artifact.name} (step {step_num})", "green"))
    logger.info(colored(f"Available checkpoints (sorted by step):", "cyan"))

    # Show all available checkpoints
    checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
    for i, a in enumerate(checkpoint_artifacts[:10]):
        step = get_step_from_artifact(a)
        selected_marker = "✓" if a == artifact else " "
        logger.info(colored(f"  {selected_marker} {i+1}. {a.name} (step {step})", "cyan"))

    # Download the artifact
    download_path = download_dir / f"{run_id}_{artifact_alias}"
    artifact_dir = artifact.download(root=str(download_path))

    logger.info(colored(f"Checkpoint downloaded to: {artifact_dir}", "green"))
    return Path(artifact_dir)


# ---- Main --------------------------------------------------------------------
def main(cfg: FlowPPOConfig):
    # In case you train a policy from scratch with rewards
    # Normalize image_observation_keys -> list[str] (or None)
    cfg.image_observation_keys = (
        cfg.image_observation_keys.split(" ") if cfg.image_observation_keys is not None else None
    )

    # ---------------- DDP Setup ----------------
    is_ddp = cfg.distributed
    if is_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        # Rely on env vars; do NOT pass rank/world_size manually
        dist.init_process_group(backend="nccl")
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(cfg.device)

    logger.info(colored(f"[{rank}/{world_size}] Using device: {device}", "green"))

    if cfg.trust_region_mode == "rollback":
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("ratio rollback requires chunk-level FPO")
        if cfg.rollback_alpha <= 0.0:
            raise ValueError("rollback alpha must be positive")

    if cfg.success_replay_audit:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("success replay audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("success replay audit currently requires one process")
        if min(
            cfg.success_replay_audit_iteration,
            cfg.success_replay_audit_chunks,
            cfg.success_replay_audit_min_chunks,
        ) < 1:
            raise ValueError("success replay audit iteration and chunk counts must be positive")
    if cfg.cfm_ratio_generalization_audit:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("CFM ratio generalization audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("CFM ratio generalization audit currently requires one process")
        if min(
            cfg.cfm_ratio_generalization_audit_iteration,
            cfg.cfm_ratio_generalization_audit_chunks,
            cfg.cfm_ratio_generalization_audit_min_chunks,
        ) < 1:
            raise ValueError("CFM ratio audit iteration and chunk counts must be positive")
    if cfg.advantage_stratified_mc_audit:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("advantage-stratified MC audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("advantage-stratified MC audit currently requires one process")
        if min(
            cfg.advantage_stratified_mc_audit_iteration,
            cfg.advantage_stratified_mc_audit_batches,
            cfg.advantage_stratified_mc_audit_batch_size,
            cfg.advantage_stratified_mc_audit_repeats,
            cfg.advantage_stratified_mc_audit_low_samples,
        ) < 1:
            raise ValueError("advantage-stratified MC audit settings must be positive")
        if cfg.advantage_stratified_mc_audit_batch_size % 2:
            raise ValueError("advantage-stratified MC audit batch size must be even")
        if cfg.advantage_stratified_mc_audit_repeats < 2:
            raise ValueError("advantage-stratified MC audit requires at least two repeats")
        if (
            cfg.advantage_stratified_mc_audit_low_samples
            + cfg.advantage_stratified_mc_audit_high_samples
            != 2 * cfg.advantage_stratified_mc_audit_uniform_samples
        ):
            raise ValueError("stratified and uniform MC budgets must match")
        if (
            cfg.advantage_stratified_mc_audit_reference_samples
            <= cfg.advantage_stratified_mc_audit_high_samples
        ):
            raise ValueError("reference MC count must exceed the high allocation")
    if cfg.heldout_ratio_early_stop_audit:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("held-out ratio early-stop audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("held-out ratio early-stop audit currently requires one process")
        if min(
            cfg.heldout_ratio_early_stop_audit_iteration,
            cfg.heldout_ratio_early_stop_audit_chunks,
        ) < 1:
            raise ValueError("held-out ratio early-stop audit settings must be positive")
        if cfg.heldout_ratio_early_stop_audit_chunks % 2:
            raise ValueError("held-out ratio early-stop audit requires an even chunk count")
        if not 0.0 < cfg.heldout_ratio_early_stop_threshold <= 1.0:
            raise ValueError("held-out ratio early-stop threshold must be in (0, 1]")
    if cfg.discounted_success_critic_audit:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("discounted-success critic audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("discounted-success critic audit currently requires one process")
        if cfg.gradient_accumulation_steps != 1:
            raise ValueError("discounted-success critic audit requires gradient accumulation 1")
        if min(
            cfg.discounted_success_critic_audit_iteration,
            cfg.discounted_success_critic_audit_chunks,
        ) < 1:
            raise ValueError("discounted-success critic audit settings must be positive")
        if cfg.discounted_success_critic_audit_chunks % 2:
            raise ValueError("discounted-success critic audit requires an even chunk count")
    if cfg.critic_warmup_scheduler_audit:
        if cfg.discounted_success_critic_audit or cfg.direct_advantage_audit:
            raise ValueError("critic side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("critic warmup scheduler audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("critic warmup scheduler audit currently requires one process")
        if cfg.gradient_accumulation_steps != 1:
            raise ValueError("critic warmup scheduler audit requires gradient accumulation 1")
        if min(
            cfg.critic_warmup_scheduler_audit_iteration,
            cfg.critic_warmup_scheduler_audit_chunks,
        ) < 1:
            raise ValueError("critic warmup scheduler audit settings must be positive")
        if cfg.critic_warmup_scheduler_audit_chunks % 2:
            raise ValueError("critic warmup scheduler audit requires an even chunk count")
    if cfg.direct_advantage_audit:
        if cfg.discounted_success_critic_audit:
            raise ValueError("critic and direct-advantage side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("direct-advantage audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("direct-advantage audit currently requires one process")
        if min(
            cfg.direct_advantage_audit_iteration,
            cfg.direct_advantage_audit_chunks,
            cfg.direct_advantage_center_samples,
            cfg.direct_advantage_horizon_chunks,
            cfg.direct_advantage_center_batch_size,
        ) < 1:
            raise ValueError("direct-advantage audit settings must be positive")
        if cfg.direct_advantage_audit_chunks % 2:
            raise ValueError("direct-advantage audit requires an even chunk count")
    if cfg.rank_advantage_audit:
        if any(
            (
                cfg.discounted_success_critic_audit,
                cfg.critic_warmup_scheduler_audit,
                cfg.direct_advantage_audit,
            )
        ):
            raise ValueError("rank-advantage and advantage side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("rank-advantage audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("rank-advantage audit currently requires one process")
        if min(cfg.rank_advantage_audit_iteration, cfg.rank_advantage_audit_chunks) < 1:
            raise ValueError("rank-advantage audit settings must be positive")
        if cfg.rank_advantage_audit_chunks % 2:
            raise ValueError("rank-advantage audit requires an even chunk count")
    if cfg.advantage_sign_pcgrad_audit:
        if any(
            (
                cfg.discounted_success_critic_audit,
                cfg.critic_warmup_scheduler_audit,
                cfg.direct_advantage_audit,
                cfg.rank_advantage_audit,
                cfg.bc_anchor_pcgrad_audit,
                cfg.terminal_consistency_filter_audit,
            )
        ):
            raise ValueError("advantage-sign PCGrad and side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("advantage-sign PCGrad audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("advantage-sign PCGrad audit currently requires one process")
        if min(
            cfg.advantage_sign_pcgrad_audit_iteration,
            cfg.advantage_sign_pcgrad_audit_chunks,
            cfg.advantage_sign_pcgrad_min_per_sign,
        ) < 1:
            raise ValueError("advantage-sign PCGrad audit settings must be positive")
        if cfg.advantage_sign_pcgrad_audit_chunks % 2:
            raise ValueError("advantage-sign PCGrad audit requires an even chunk count")
        if (
            2 * cfg.advantage_sign_pcgrad_min_per_sign
            > cfg.advantage_sign_pcgrad_audit_chunks // 2
        ):
            raise ValueError("advantage-sign PCGrad support cannot fit in each audit batch")
    if cfg.temporal_ratio_clipping_audit:
        if any(
            (
                cfg.discounted_success_critic_audit,
                cfg.critic_warmup_scheduler_audit,
                cfg.direct_advantage_audit,
                cfg.rank_advantage_audit,
                cfg.advantage_sign_pcgrad_audit,
                cfg.bc_anchor_pcgrad_audit,
                cfg.terminal_consistency_filter_audit,
                cfg.median_microbatch_gradient_audit,
            )
        ):
            raise ValueError("temporal-ratio clipping and side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("temporal-ratio clipping audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("temporal-ratio clipping audit currently requires one process")
        if min(
            cfg.temporal_ratio_clipping_audit_iteration,
            cfg.temporal_ratio_clipping_audit_chunks,
        ) < 1:
            raise ValueError("temporal-ratio clipping audit settings must be positive")
        if cfg.temporal_ratio_clipping_audit_chunks % 2:
            raise ValueError("temporal-ratio clipping audit requires an even chunk count")
    if cfg.median_microbatch_gradient_audit:
        if any(
            (
                cfg.discounted_success_critic_audit,
                cfg.critic_warmup_scheduler_audit,
                cfg.direct_advantage_audit,
                cfg.rank_advantage_audit,
                cfg.advantage_sign_pcgrad_audit,
                cfg.temporal_ratio_clipping_audit,
                cfg.bc_anchor_pcgrad_audit,
            )
        ):
            raise ValueError("median-microbatch gradient and side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("median-microbatch gradient audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("median-microbatch gradient audit currently requires one process")
        if cfg.median_microbatch_gradient_audit_iteration < 1:
            raise ValueError("median-microbatch gradient audit iteration must be positive")
        if cfg.median_microbatch_gradient_audit_chunks != 192:
            raise ValueError("median-microbatch gradient audit requires exactly 192 chunks")
    if cfg.terminal_consistency_filter_audit:
        if any(
            (
                cfg.discounted_success_critic_audit,
                cfg.critic_warmup_scheduler_audit,
                cfg.direct_advantage_audit,
                cfg.rank_advantage_audit,
                cfg.advantage_sign_pcgrad_audit,
                cfg.temporal_ratio_clipping_audit,
                cfg.median_microbatch_gradient_audit,
                cfg.bc_anchor_pcgrad_audit,
            )
        ):
            raise ValueError("terminal-consistency filter and side audits are mutually exclusive")
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("terminal-consistency filter audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("terminal-consistency filter audit currently requires one process")
        if cfg.terminal_consistency_filter_audit_iteration < 1:
            raise ValueError("terminal-consistency filter audit iteration must be positive")
        if cfg.terminal_consistency_filter_audit_chunks != 128:
            raise ValueError("terminal-consistency filter audit requires exactly 128 chunks")
    if cfg.bc_anchor_pcgrad_audit:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("BC-anchor PCGrad audit requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("BC-anchor PCGrad audit currently requires one process")
        if min(
            cfg.bc_anchor_pcgrad_audit_iteration,
            cfg.bc_anchor_pcgrad_audit_chunks,
        ) < 1:
            raise ValueError("BC-anchor PCGrad audit settings must be positive")
        if cfg.bc_anchor_pcgrad_audit_chunks % 2:
            raise ValueError("BC-anchor PCGrad audit requires an even chunk count")
    if cfg.zero_endpoint_pcgrad_train:
        if cfg.loss_mode != "fpo" or not cfg.do_chunk_level_ppo:
            raise ValueError("zero-endpoint PCGrad training requires chunk-level FPO")
        if world_size != 1:
            raise ValueError("zero-endpoint PCGrad training currently requires one process")
        if cfg.gradient_accumulation_steps != 1:
            raise ValueError("zero-endpoint PCGrad training requires gradient accumulation 1")

    # Run dir on all ranks (avoid races)
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{cfg.experiment}_{run_start_time}" if cfg.output_dir is None else Path(cfg.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()
    if rank == 0:
        logger.info(colored(f"Run directory: {run_dir}", "green"))

    # Seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic
    if rank == 0:
        logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # ------------- Rank-0 I/O -> broadcast config/weights ----------------
    policy_config_payload: Dict[str, Any] | None = None
    state_dict_payload: Dict[str, Any] | None = None
    ema_state_payload: Dict[str, Any] | None = None

    if rank == 0:
        # Decide policy directory or load artifacts
        if cfg.base_policy_wandb_run_id is not None:
            alias = cfg.checkpoint_step if cfg.checkpoint_step is not None else "latest"
            if cfg.wandb_project is None:
                raise ValueError("--wandb_project is required when using --base_policy_wandb_run_id")
            ckpt_root = download_checkpoint_from_wandb_rank0(
                run_id=cfg.base_policy_wandb_run_id,
                project=cfg.base_policy_wandb_project,
                entity=cfg.wandb_entity,
                artifact_alias=alias,
                download_dir=run_dir / "downloaded_checkpoints",
            )
            policy_dir = ckpt_root / "policy"
        elif cfg.base_policy_local_path is not None:
            pd = Path(cfg.base_policy_local_path) / "policy"
            policy_dir = pd if pd.exists() else Path(cfg.base_policy_local_path)
        else:
            raise ValueError("Must provide either --base_policy_wandb_run_id or --base_policy_local_path")

        logger.info(colored(f"[Global] Policy directory: {policy_dir}", "cyan"))
        config_path = policy_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        # sanitize for FlowMatchingConfig
        config_dict.pop("type", None)
        config_dict.pop("normalization_mapping", None)
        policy_config_payload = config_dict

        weights_path = policy_dir / "model.safetensors"
        if not weights_path.exists():
            raise ValueError(f"Model weights file not found: {weights_path}")
        state_dict_payload = load_file(weights_path, device="cpu")

        # Load EMA from optimizer.pt if available
        optimizer_path = policy_dir.parent / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_blob = torch.load(optimizer_path, map_location="cpu")
            ema_state_payload = optimizer_blob.get("ema_state_dict", None)

    # Broadcast payloads
    if is_ddp:
        obj = [policy_config_payload, state_dict_payload, ema_state_payload]
        dist.broadcast_object_list(obj, src=0)
        policy_config_payload, state_dict_payload, ema_state_payload = obj

    # ---------- Build actor/critic identically on every rank --------------
    # Construct FlowMatchingConfig from payload
    policy_config = FlowMatchingConfig(**policy_config_payload)  # type: ignore[arg-type]

    # Derive features from input_features (same as pretraining)
    policy_config.image_features = [k for k in policy_config.input_features if "image" in k]
    policy_config.state_features = [k for k in policy_config.input_features if "state" in k or "pos" in k]
    logger.info(f"[Rank {rank}] Image features: {policy_config.image_features}")
    logger.info(f"[Rank {rank}] State features: {policy_config.state_features}")

    # Update cfg.image_observation_keys from config fnames
    cfg.image_observation_keys = [k.replace("observation.images.", "") for k in policy_config.image_features]
    logger.info(f"[Rank {rank}] Using image features from checkpoint config: {policy_config.image_features}")

    # Instantiate actor
    logger.info(colored(f"[Rank {rank}] Constructing FlowMatchingPolicy from config...", "cyan"))
    actor = FlowMatchingPolicy(policy_config, dataset_stats=None)

    # Load weights from broadcast payload
    actor.load_state_dict(state_dict_payload, strict=True)  # type: ignore[arg-type]
    logger.info(colored(f"[Rank {rank}] Loaded policy from checkpoint", "green"))

    # Apply EMA if provided + requested
    if getattr(actor, "ema_model", None) is not None and ema_state_payload is not None:
        actor.ema_model.load_state_dict(ema_state_payload)  # type: ignore[arg-type]
        if cfg.load_ema:
            actor.ema_model.copy_to(actor.model.parameters())
            if rank == 0:
                logger.info(colored("Loaded and applied EMA weights from checkpoint", "green"))
    elif cfg.load_ema and rank == 0:
        logger.warning(colored("--load_ema flag set but no EMA weights found", "yellow"))


    # --------- Apply policy overrides (must be identical across ranks) ----
    def log_override(name, new, old):
        logger.info(f"[Rank {rank}] Overriding {name} to {new} from base policy {old}")

    if cfg.n_action_steps is not None:
        # Only override if not already set in the base policy config
        logger.warning(f"Only if you are training from scratch, otherwise you should use the same n_action_steps as the base policy ({actor.config.n_action_steps})")
        log_override("n_action_steps", cfg.n_action_steps, actor.config.n_action_steps)
        actor.config.n_action_steps = cfg.n_action_steps
    else:
        cfg.n_action_steps = actor.config.n_action_steps

    if cfg.sampling_steps is not None:
        log_override("sampling_steps", cfg.sampling_steps, actor.config.sampling_steps)
        actor.config.sampling_steps = cfg.sampling_steps

    if cfg.init_flow_network:
        logger.info(colored("[Rank {rank}] Initialize flow action network to random weights", "cyan"))
        actor.model.initialize_layers()

    if cfg.cfm_loss_use_huber is not None:
        log_override("cfm_loss_use_huber", cfg.cfm_loss_use_huber, actor.config.cfm_loss_use_huber)
        actor.config.cfm_loss_use_huber = cfg.cfm_loss_use_huber

    if cfg.cfm_loss_huber_delta is not None:
        log_override("cfm_loss_huber_delta", cfg.cfm_loss_huber_delta, actor.config.cfm_loss_huber_delta)
        actor.config.cfm_loss_huber_delta = cfg.cfm_loss_huber_delta

    if cfg.flow_network_output_param is not None:
        log_override("flow_network_output_param", cfg.flow_network_output_param, actor.config.flow_network_output_param)
        actor.config.flow_network_output_param = cfg.flow_network_output_param

    if cfg.cfm_loss_mode is not None:
        log_override("cfm_loss_mode", cfg.cfm_loss_mode, actor.config.cfm_loss_mode)
        actor.config.cfm_loss_mode = cfg.cfm_loss_mode
    
    if cfg.transported_clip_value is not None:
        log_override("transported_clip_value", cfg.transported_clip_value, actor.config.transported_clip_value)
        actor.config.transported_clip_value = cfg.transported_clip_value

    if cfg.cfm_loss_weight_from_t is not None:
        log_override("cfm_loss_weight_from_t", cfg.cfm_loss_weight_from_t, actor.config.cfm_loss_weight_from_t)
        actor.config.cfm_loss_weight_from_t = cfg.cfm_loss_weight_from_t

    if cfg.exploration_noise_std is not None:
        logger.info(f"[Rank {rank}] Overriding exploration_noise_std to {cfg.exploration_noise_std} "
                    f"from base policy {getattr(actor, 'exploration_noise_std', None)}")
        actor.exploration_noise_std = cfg.exploration_noise_std
    if cfg.advantage_weighting == "ess_softmax":
        if cfg.loss_mode != "fpo":
            raise ValueError("ess_softmax advantage weighting is only supported for FPO")
        if cfg.trust_region_mode != "ppo":
            raise ValueError("ess_softmax advantage weighting requires the PPO trust region")

    if cfg.sde_sigma is not None:
        logger.info(f"[Rank {rank}] Overriding sde_sigma to {cfg.sde_sigma} in actor config "
                    f"from base policy {getattr(actor, 'config.sde_sigma', None)}")
        actor.config.sde_sigma = cfg.sde_sigma
    if cfg.learn_sde_sigma:
        logger.info(f"[Rank {rank}] Overriding learn_sde_sigma to {cfg.learn_sde_sigma} in actor config "
                    f"from base policy {getattr(actor, 'config.learn_sde_sigma', None)}")
        actor.config.learn_sde_sigma = cfg.learn_sde_sigma
        actor.config.noise_injection_min = cfg.noise_injection_min
        actor.config.noise_injection_max = cfg.noise_injection_max
        actor.initialize_noise_injection_network()

    assert actor.config.n_action_steps == cfg.n_action_steps
    assert cfg.n_action_steps <= actor.config.horizon, \
        f"n_action_steps ({cfg.n_action_steps}) must be <= horizon ({actor.config.horizon})"

    # Critic
    critic = Critic(global_obs_dim=actor.model.global_cond_dim)
    success_critic = (
        copy.deepcopy(critic)
        if cfg.discounted_success_critic_audit or cfg.critic_warmup_scheduler_audit
        else None
    )
    direct_advantage_head = (
        DirectAdvantageHead(
            observation_dim=actor.model.global_cond_dim,
            action_size=cfg.n_action_steps * actor.model.action_dim,
        )
        if cfg.direct_advantage_audit
        else None
    )

    # Move to device BEFORE DDP
    actor.to(device)
    if getattr(actor, "ema_model", None) is not None:
        actor.ema_model.to(device)
    critic.to(device)
    if success_critic is not None:
        success_critic.to(device)
    if direct_advantage_head is not None:
        direct_advantage_head.to(device)

    # Freeze vision encoder AFTER wrapping with DDP (same on all ranks)
    if cfg.freeze_vision_encoder:
        logger.info(colored(f"[Rank {rank}] Freezing vision encoder", "cyan"))
        for p in actor.model.vision_encoder.parameters():
            p.requires_grad = False

    endpoint_anchor_model = None
    if cfg.zero_endpoint_pcgrad_train:
        endpoint_anchor_model = copy.deepcopy(actor.model).eval()
        for parameter in endpoint_anchor_model.parameters():
            parameter.requires_grad = False

    # Wrap with DDP
    if is_ddp:
        # Remove ema model from actor before wrapping to avoid unused parameter issues
        actor.ema_model = None
        cfg.eval_ema = False
        logger.info(colored(f"[Rank {rank}] Removing EMA model for DDP compatibility", "cyan"))

        actor = DDP(
            actor,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        critic = DDP(
            critic,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )


    logger.info(f"[Rank {rank}] Actor: {actor.__class__.__name__}")
    actor_module = actor.module if is_ddp else actor
    logger.info(f"[Rank {rank}] N action steps: {actor_module.config.n_action_steps}, "
                f"prediction horizon: {actor_module.config.horizon}")
    logger.info(f"[Rank {rank}] Critic: {critic}")


    # ----------------- Environment setup -----------------
    n_action_samples = cfg.n_action_samples
    steps_per_iteration = cfg.data_collection_steps
    n_action_steps = cfg.n_action_steps

    group_size = n_action_samples if cfg.cfm_loss_average_group_size == -1 else cfg.cfm_loss_average_group_size
    n_groups = n_action_samples // group_size
    assert actor_module.config.horizon >= n_action_steps
    assert n_action_samples % group_size == 0
    assert steps_per_iteration % n_action_steps == 0
    
    if is_ddp:
        num_envs_per_process = cfg.num_envs // world_size + (1 if rank < (cfg.num_envs % world_size) else 0)
        global_env_offset = rank * (cfg.num_envs // world_size) + min(rank, cfg.num_envs % world_size)
        batch_size = cfg.data_collection_steps * cfg.num_envs // n_action_steps
        local_batch_size = cfg.data_collection_steps * num_envs_per_process // n_action_steps
        minibatch_size = max(local_batch_size // cfg.num_minibatches, 1)
        logger.info(f"[Rank {rank}] Handling {num_envs_per_process} envs | "
                    f"Local batch: {local_batch_size} | Minibatch: {minibatch_size}")
    else:
        num_envs_per_process = cfg.num_envs
        global_env_offset = 0
        batch_size = cfg.data_collection_steps * cfg.num_envs // n_action_steps
        local_batch_size = batch_size
        minibatch_size = max(batch_size // cfg.num_minibatches, 1)

    num_iterations = cfg.total_timesteps // batch_size
    assert cfg.gradient_accumulation_steps <= cfg.num_minibatches
    effective_minibatch_size = minibatch_size * cfg.gradient_accumulation_steps
    logger.info(f"[Rank {rank}] Grad accum steps: {cfg.gradient_accumulation_steps} | "
                f"Effective MB: {effective_minibatch_size} (base: {minibatch_size})")

    device_str = "cpu" if cfg.device == "cpu" else "cuda"
    env = create_vectorized_env(
        env_name=cfg.task,
        num_envs=num_envs_per_process,
        device=device_str,
        camera_size=cfg.camera_size,
        video_key="agentview",
        debug=cfg.debug,
        expected_image_keys=cfg.image_observation_keys,
        seeds=[
            cfg.seed + global_env_offset + env_id
            for env_id in range(num_envs_per_process)
        ],
    )
    # Init action buffers
    actor_module.init_action_buffers(num_envs_per_process)
    if cfg.rollout_zero_fraction > 0 and cfg.rollout_tempered_fraction > 0:
        raise ValueError("rollout_zero_fraction and rollout_tempered_fraction cannot both be enabled")
    if cfg.rollout_tempered_scale < 0:
        raise ValueError("rollout_tempered_scale must be non-negative")

    guided_fraction = max(cfg.rollout_zero_fraction, cfg.rollout_tempered_fraction)
    rollout_guided_mask = build_rollout_zero_sampling_mask(
        num_envs_per_process,
        guided_fraction,
        global_num_envs=cfg.num_envs,
        global_offset=global_env_offset,
        device=device,
    )
    rollout_zero_sampling_mask = rollout_guided_mask if cfg.rollout_zero_fraction > 0 else None
    rollout_source_sampling_scale = None
    source_group_name = "zero-source"
    if cfg.rollout_tempered_fraction > 0:
        rollout_source_sampling_scale = torch.ones(num_envs_per_process, device=device)
        rollout_source_sampling_scale[rollout_guided_mask] = cfg.rollout_tempered_scale
        source_group_name = f"scale-{cfg.rollout_tempered_scale:g}"
    guided_rollout_env_ids = set(torch.where(rollout_guided_mask.cpu())[0].tolist())
    logger.info(
        f"[Rank {rank}] Mixed-source rollout: {len(guided_rollout_env_ids)} {source_group_name} and "
        f"{num_envs_per_process - len(guided_rollout_env_ids)} scale-1 Gaussian environments"
    )
    logger.info(colored(f"[Rank {rank}] Initialized action buffers for {num_envs_per_process} environments", "green"))

    # Obs/action dims
    joint_pos_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_space.shape[1]
    image_keys = actor_module.config.image_features
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    n_images = len(image_keys)
    logger.info(f"[Rank {rank}] Obs state dim: {joint_pos_dim} | Action dim: {action_dim} | "
                f"Image keys: {image_keys} | Image shape: ({img_c},{img_h},{img_w}) | n_images={n_images}")

    # ----------------- Optimizers / sched ----------------
    params_actor = actor.parameters()
    optimizer_actor = optim.AdamW(
        params_actor,
        lr=cfg.learning_rate_actor,
        betas=tuple(cfg.optimizer_betas_actor),
        eps=1e-5,
        weight_decay=1e-6,
    )
    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler_name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler_actor_warmup_steps,
        num_training_steps=num_iterations,
    )
    critic_params = critic.parameters()
    optimizer_critic = optim.AdamW(critic_params, lr=cfg.learning_rate_critic, eps=1e-5, weight_decay=1e-6)
    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler_name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler_critic_warmup_steps,
        num_training_steps=num_iterations,
    )
    optimizer_success_critic = None
    lr_scheduler_success_critic = None
    critic_audit_initial_parameters = None
    if success_critic is not None:
        optimizer_success_critic = optim.AdamW(
            success_critic.parameters(),
            lr=cfg.learning_rate_critic,
            eps=1e-5,
            weight_decay=1e-6,
        )
        if cfg.discounted_success_critic_audit:
            lr_scheduler_success_critic = get_scheduler(
                name=cfg.lr_scheduler_name,
                optimizer=optimizer_success_critic,
                num_warmup_steps=cfg.lr_scheduler_critic_warmup_steps,
                num_training_steps=num_iterations,
            )
        if cfg.critic_warmup_scheduler_audit:
            critic_audit_initial_parameters = [
                parameter.detach().cpu().clone()
                for parameter in success_critic.parameters()
            ]
    optimizer_direct_advantage = None
    if direct_advantage_head is not None:
        optimizer_direct_advantage = optim.Adam(
            direct_advantage_head.parameters(),
            lr=cfg.learning_rate_critic,
            eps=1e-5,
        )
    logger.info(f"[Rank {rank}] Total timesteps: {cfg.total_timesteps}, batch size: {batch_size} | "
                f"MB: {minibatch_size}, iterations: {num_iterations}")

    # ----------------- W&B (rank 0 only) -----------------
    if cfg.wandb_enable and rank == 0:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when --wandb_enable")
        wandb_run_id = cfg.wandb_continue_run_id if cfg.wandb_continue_run_id else None
        wandb_resume_mode = "must" if cfg.wandb_continue_run_id else None
        wandb_config = {**vars(cfg), "num_iterations": num_iterations, "batch_size": batch_size, "minibatch_size": minibatch_size}
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=wandb_config,
            name=f"{cfg.experiment}_{cfg.policy}_{cfg.task}",
            id=wandb_run_id,
            resume=wandb_resume_mode,
            dir=str(run_dir),
            settings=wandb.Settings(),
        )
        logger.info(colored(f"W&B logging enabled (logs saved to {run_dir / 'wandb'})", "blue"))

    # ----------------- Training storage ------------------
    checkpoints_dir = run_dir / "checkpoints" if rank == 0 else None
    if rank == 0:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0  # maintain on rank 0 only
    iteration = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    direct_advantage_training_losses: list[float] = []
    endpoint_pcgrad_training_history: list[dict[str, Any]] = []

    obs_state_stored = torch.zeros((steps_per_iteration, num_envs_per_process, joint_pos_dim))
    actions_stored = torch.zeros((steps_per_iteration, num_envs_per_process, action_dim))
    mdp_x_t_paths_stored = torch.zeros((steps_per_iteration, actor_module.config.sampling_steps, num_envs_per_process, action_dim))
    rewards_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    dones_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    terminals_stored = torch.zeros(
        (steps_per_iteration, num_envs_per_process), dtype=torch.bool
    )
    values_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    cfm_losses_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples))
    cfm_loss_ts_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples))
    cfm_loss_epsilons_stored = torch.zeros((steps_per_iteration, num_envs_per_process, n_action_samples, action_dim))
    cfm_value_invalid_stored = torch.zeros((steps_per_iteration, num_envs_per_process))
    dppo_log_probs_stored = torch.zeros((steps_per_iteration, actor_module.config.sampling_steps, num_envs_per_process))

    next_done = torch.zeros(num_envs_per_process)
    next_obs, _ = env.reset()
    actor_module.reset()

    obs_images_stored_list: Dict[str, torch.Tensor] = {
        k: torch.zeros((steps_per_iteration, num_envs_per_process, 3, img_h, img_w))
        for k in next_obs.keys() if k.startswith("observation.images.")
    }

    def get_cfm_values(actor, obs, n_action_samples=1, cfm_loss_ts=None, cfm_loss_epsilons=None, debug=False, return_predictions=False):
        output = actor(
            obs,
            n_action_samples,
            cfm_loss_ts,
            cfm_loss_epsilons,
            debug=debug,
            return_cfm_predictions=return_predictions,
        )
        return output

    def replay_audit_objective(
        chunk_indices: torch.Tensor,
        chunk_weights: torch.Tensor,
        *,
        temporal_clipping: bool = False,
        b_actions: torch.Tensor,
        b_cfm_losses: torch.Tensor,
        b_cfm_loss_ts: torch.Tensor,
        b_cfm_loss_epsilons: torch.Tensor,
        b_cfm_value_invalid: torch.Tensor,
        b_obs_images: dict[str, torch.Tensor],
        b_obs_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a fixed-variable diagnostic chunk-level PPO objective."""
        audit_actions = b_actions[chunk_indices].to(device)
        audit_old_losses = b_cfm_losses[chunk_indices].to(device)
        audit_times = b_cfm_loss_ts[chunk_indices].to(device)
        audit_noises = b_cfm_loss_epsilons[chunk_indices].to(device)
        audit_valid = 1.0 - b_cfm_value_invalid[chunk_indices].to(device)
        audit_obs = {key: value[chunk_indices, 0].to(device) for key, value in b_obs_images.items()}
        audit_obs["observation.state"] = b_obs_state[chunk_indices, 0].to(device)
        audit_obs["action"] = audit_actions

        fixed_times = audit_times.permute(0, 2, 1).reshape(-1, n_action_steps, 1)
        if not (fixed_times[:, 0, 0] == fixed_times[:, -1, 0]).all():
            raise RuntimeError("replay audit received inconsistent CFM times within a chunk")
        fixed_times = fixed_times[:, 0:1, :]
        fixed_noises = audit_noises.permute(0, 2, 1, 3).reshape(
            -1, n_action_steps, action_dim
        )
        current_losses, _, _ = get_cfm_values(
            actor_module,
            audit_obs,
            n_action_samples,
            fixed_times,
            fixed_noises,
        )
        current_losses = current_losses.permute(1, 0, 2)
        old_losses = audit_old_losses.reshape(
            audit_old_losses.shape[0], -1, n_groups, group_size
        )
        current_losses = current_losses.reshape(
            current_losses.shape[0], -1, n_groups, group_size
        )
        if cfg.clamp_old_cfm_loss is not None:
            old_losses = old_losses.clamp(max=cfg.clamp_old_cfm_loss)

        if temporal_clipping:
            old_losses = old_losses.mean(dim=-1)
            current_losses = current_losses.mean(dim=-1)
            return clipped_ratio_objective(
                old_losses,
                current_losses,
                chunk_weights.to(device),
                audit_valid,
                cfg.clip_coef,
                temporal=True,
                clamp_logratio=cfg.clamp_logratio,
            )

        if cfg.do_average_cfm_loss_in_chunk:
            denominator = audit_valid.sum(dim=1).unsqueeze(-1).unsqueeze(-1).clamp_min(1.0)
            old_losses = (old_losses * audit_valid[:, :, None, None]).sum(dim=1) / denominator
            current_losses = (
                current_losses * audit_valid[:, :, None, None]
            ).sum(dim=1) / denominator
        else:
            old_losses = (old_losses * audit_valid[:, :, None, None]).sum(dim=1)
            current_losses = (current_losses * audit_valid[:, :, None, None]).sum(dim=1)

        logratio = old_losses.mean(dim=-1) - current_losses.mean(dim=-1)
        if cfg.clamp_logratio is not None:
            logratio = clamp_ste(
                logratio,
                min=-cfg.clamp_logratio,
                max=cfg.clamp_logratio,
            )
        ratios = logratio.exp()
        weights = chunk_weights.to(device).reshape(-1, 1)
        loss_unclipped = -weights * ratios
        loss_clipped = -weights * ratios.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
        audit_loss = torch.maximum(loss_unclipped, loss_clipped).mean()
        return ratios, audit_loss

    def replay_audit_gradient(
        chunk_indices: torch.Tensor,
        chunk_weights: torch.Tensor,
        *,
        temporal_clipping: bool = False,
        b_actions: torch.Tensor,
        b_cfm_losses: torch.Tensor,
        b_cfm_loss_ts: torch.Tensor,
        b_cfm_loss_epsilons: torch.Tensor,
        b_cfm_value_invalid: torch.Tensor,
        b_obs_images: dict[str, torch.Tensor],
        b_obs_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a diagnostic chunk-level PPO gradient without populating .grad."""
        ratios, audit_loss = replay_audit_objective(
            chunk_indices,
            chunk_weights,
            temporal_clipping=temporal_clipping,
            b_actions=b_actions,
            b_cfm_losses=b_cfm_losses,
            b_cfm_loss_ts=b_cfm_loss_ts,
            b_cfm_loss_epsilons=b_cfm_loss_epsilons,
            b_cfm_value_invalid=b_cfm_value_invalid,
            b_obs_images=b_obs_images,
            b_obs_state=b_obs_state,
        )
        parameters = [parameter for parameter in actor_module.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(audit_loss, parameters, allow_unused=True)
        vector = torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient)
                .detach()
                .flatten()
                .cpu()
                for parameter, gradient in zip(parameters, gradients, strict=True)
            ]
        )
        if not torch.isfinite(ratios).all() or not torch.isfinite(vector).all():
            raise RuntimeError("success replay audit produced non-finite ratios or gradients")
        return ratios.detach().cpu(), vector

    def actor_velocity_field(
        observation_conditioning: torch.Tensor,
        query_points: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the actor velocity field at fixed normalized flow-space queries."""
        time_embedding = actor_module.model.diffusion_step_encoder(query_times)
        time_embedding = time_embedding.reshape(query_points.shape[0], -1)
        network_output = actor_module.model(
            query_points, time_embedding, observation_conditioning
        )
        network_output = actor_module.config.mlp_output_scale * network_output
        if actor_module.config.transported_clip_value is not None:
            network_output = network_output.clamp(
                -actor_module.config.transported_clip_value,
                actor_module.config.transported_clip_value,
            )
        if actor_module.config.flow_network_output_param == "u":
            return network_output
        return (query_points - network_output) / query_times.clamp_min(1e-5)

    def actor_zero_source_endpoint(
        observation_conditioning: torch.Tensor,
        *,
        policy_model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Differentiably integrate the policy from its deterministic zero source."""
        policy_model = actor_module.model if policy_model is None else policy_model
        batch = observation_conditioning.shape[0]
        x_t = torch.zeros(
            (batch, actor_module.config.horizon, action_dim),
            device=observation_conditioning.device,
            dtype=observation_conditioning.dtype,
        )
        schedule = actor_module.get_schedule(x_t.device)
        for step in range(actor_module.config.sampling_steps):
            t_current = schedule[step]
            dt = schedule[step + 1] - t_current
            time_embedding = policy_model.diffusion_step_encoder(
                t_current.reshape(1)
            ).reshape(1, -1).expand(batch, -1)
            network_output = policy_model(
                x_t, time_embedding, observation_conditioning
            )
            network_output = actor_module.config.mlp_output_scale * network_output
            if actor_module.config.transported_clip_value is not None:
                network_output = network_output.clamp(
                    -actor_module.config.transported_clip_value,
                    actor_module.config.transported_clip_value,
                )
            if actor_module.config.flow_network_output_param == "u":
                velocity = network_output
            else:
                velocity = (x_t - network_output) / t_current.clamp_min(1e-5)
            x_t = x_t + velocity * dt
        return actor_module.config.actor_scale * x_t[:, :n_action_steps]

    def autograd_list(loss: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        parameters = [
            parameter for parameter in actor_module.parameters() if parameter.requires_grad
        ]
        gradients = list(torch.autograd.grad(loss, parameters, allow_unused=True))
        return parameters, gradients

    @torch.no_grad()
    def restore_parameters(
        parameters: list[torch.Tensor], values: list[torch.Tensor]
    ) -> None:
        for parameter, value in zip(parameters, values, strict=True):
            parameter.copy_(value)

    @torch.no_grad()
    def apply_virtual_actor_step(
        parameters: list[torch.Tensor], gradients: list[torch.Tensor | None]
    ) -> float:
        norm = gradient_norm(gradients)
        clip_scale = min(1.0, cfg.max_grad_norm / max(float(norm.item()), 1e-12))
        step_scale = cfg.learning_rate_actor * clip_scale
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is not None:
                parameter.add_(gradient, alpha=-step_scale)
        return step_scale

    @torch.no_grad()
    def direct_advantage_inputs(
        *,
        b_actions: torch.Tensor,
        b_obs_images: dict[str, torch.Tensor],
        b_obs_state: torch.Tensor,
        seed_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode chunk starts and sample fixed same-state actions for centering."""
        observation_batches = []
        behavior_batches = []
        center_batches = []
        center_samples = cfg.direct_advantage_center_samples
        generator = torch.Generator(device=device).manual_seed(cfg.seed + seed_offset)
        for start in range(0, b_actions.shape[0], cfg.direct_advantage_center_batch_size):
            end = min(start + cfg.direct_advantage_center_batch_size, b_actions.shape[0])
            observations = {
                key: value[start:end, 0].to(device)
                for key, value in b_obs_images.items()
            }
            observations["observation.state"] = b_obs_state[start:end, 0].to(device)
            normalized_observations = actor_module.normalize_inputs(copy.deepcopy(observations))
            observation_batches.append(
                actor_module.model.encode_observations(normalized_observations).detach()
            )

            behavior_actions = b_actions[start:end].to(device)
            behavior_batches.append(
                actor_module.normalize_targets({"action": behavior_actions})["action"].detach()
            )

            repeated_observations = {
                key: value.repeat_interleave(center_samples, dim=0)
                for key, value in observations.items()
            }
            source_noise = torch.randn(
                (
                    (end - start) * center_samples,
                    actor_module.config.horizon,
                    action_dim,
                ),
                generator=generator,
                device=device,
                dtype=behavior_actions.dtype,
            )
            sampled_actions, _ = actor_module.predict_action_chunk(
                repeated_observations,
                source_noise=source_noise,
            )
            sampled_actions = sampled_actions[:, :n_action_steps]
            sampled_actions = actor_module.normalize_targets(
                {"action": sampled_actions}
            )["action"]
            center_batches.append(
                sampled_actions.reshape(
                    end - start,
                    center_samples,
                    n_action_steps,
                    action_dim,
                ).detach()
            )
        return (
            torch.cat(observation_batches),
            torch.cat(behavior_batches),
            torch.cat(center_batches),
        )

    def stratified_mc_audit_gradient(
        chunk_indices: torch.Tensor,
        chunk_weights: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        seed_offset: int,
        b_actions: torch.Tensor,
        b_obs_images: dict[str, torch.Tensor],
        b_obs_state: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate one signed behavior-policy gradient with per-chunk MC counts."""
        if chunk_indices.numel() != chunk_weights.numel() or sample_counts.shape != chunk_weights.shape:
            raise ValueError("stratified MC audit inputs must have matching lengths")

        total_loss = torch.zeros((), device=device)
        for sample_count_tensor in torch.unique(sample_counts, sorted=True):
            sample_count = int(sample_count_tensor.item())
            group_positions = torch.where(sample_counts == sample_count_tensor)[0]
            group_indices = chunk_indices[group_positions]
            group_actions = b_actions[group_indices].to(device)
            group_obs = {
                key: value[group_indices, 0].to(device) for key, value in b_obs_images.items()
            }
            group_obs["observation.state"] = b_obs_state[group_indices, 0].to(device)
            group_obs["action"] = group_actions
            times, noises = sample_cfm_variables(
                batch_size=group_indices.numel(),
                num_samples=sample_count,
                horizon=n_action_steps,
                action_dim=action_dim,
                mode="iid",
                time_generator=torch.Generator(device=device).manual_seed(
                    cfg.seed + seed_offset + 31 * sample_count
                ),
                noise_generator=torch.Generator(device=device).manual_seed(
                    cfg.seed + seed_offset + 31 * sample_count + 1
                ),
                device=device,
                dtype=group_actions.dtype,
            )
            if (
                actor_module.config.cfm_loss_mode != "u"
                or actor_module.config.flow_network_output_param != "u"
            ):
                raise ValueError("stratified MC audit is locked to velocity prediction")
            with torch.no_grad():
                normalized_obs = actor_module.normalize_inputs(group_obs)
                obs_cond = actor_module.model.encode_observations(normalized_obs)
                normalized_actions = actor_module.normalize_targets(
                    {"action": group_actions}
                )["action"]
            batch_count, horizon, action_dimensions = normalized_actions.shape
            expanded_actions = normalized_actions.unsqueeze(1).expand(
                -1, sample_count, -1, -1
            ).reshape(-1, horizon, action_dimensions)
            expanded_obs_cond = obs_cond.unsqueeze(1).expand(
                -1, sample_count, -1
            ).reshape(batch_count * sample_count, -1)
            x_t = (1 - times) * expanded_actions + times * noises
            time_embedding = actor_module.model.diffusion_step_encoder(times).reshape(
                batch_count * sample_count, -1
            )
            network_output = actor_module.model(x_t, time_embedding, expanded_obs_cond)
            network_output = actor_module.config.mlp_output_scale * network_output
            if actor_module.config.transported_clip_value is not None:
                network_output = network_output.clamp(
                    -actor_module.config.transported_clip_value,
                    actor_module.config.transported_clip_value,
                )
            target_velocity = noises - expanded_actions
            current_losses = actor_module._compute_squared_error(
                network_output, target_velocity
            )
            current_losses = current_losses * actor_module._compute_cfm_loss_weight(times)
            current_chunk_losses = current_losses.mean(dim=-1).reshape(
                batch_count, sample_count, horizon
            ).sum(dim=-1)
            old_chunk_losses = current_chunk_losses.detach()
            logratio = old_chunk_losses - current_chunk_losses
            if cfg.clamp_logratio is not None:
                logratio = clamp_ste(
                    logratio,
                    min=-cfg.clamp_logratio,
                    max=cfg.clamp_logratio,
                )
            chunk_ratios = logratio.exp().mean(dim=1)
            group_weights = chunk_weights[group_positions].to(device)
            total_loss = total_loss - (group_weights * chunk_ratios).sum() / chunk_indices.numel()

        parameters = [parameter for parameter in actor_module.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(total_loss, parameters, allow_unused=True)
        vector = torch.cat(
            [
                (torch.zeros_like(parameter) if gradient is None else gradient)
                .detach()
                .flatten()
                .cpu()
                for parameter, gradient in zip(parameters, gradients, strict=True)
            ]
        )
        if not torch.isfinite(vector).all() or vector.norm() <= 0:
            raise RuntimeError("advantage-stratified MC audit produced an invalid gradient")
        return vector

    def get_log_prob_and_entropy(actor, obs):
        log_prob, entropy, sde_sigma = actor(obs, is_dppo=True)

        return log_prob, entropy, sde_sigma

    def get_action_and_value(actor_module, critic, obs, sde_sampling: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_copy = copy.deepcopy(obs)
        action, mdp_x_t_path = actor_module.select_action(
            obs,
            sde_sampling=sde_sampling,
            zero_sampling_mask=rollout_zero_sampling_mask,
            source_sampling_scale=rollout_source_sampling_scale,
        )
        obs_copy = actor_module.normalize_inputs(obs_copy)
        obs_cond = actor_module.model.encode_observations(obs_copy)
        value = critic(obs_cond)
        return action, mdp_x_t_path, value

    # ----------------- Training loop ---------------------
    logger.info(colored(f"[Rank {rank}] Starting the main training loop", "green"))

    while (iteration * cfg.data_collection_steps * cfg.num_envs) < cfg.total_timesteps:
        iteration += 1
        prepare_invalid_step_mask(
            cfm_value_invalid_stored,
            reset_each_iteration=cfg.reset_cfm_invalid_mask_each_iteration,
        )
        if rank == 0:
            logger.info(colored(
                f"Iteration: {iteration}/{num_iterations} | "
                f"Global step (approx): {global_step}/{cfg.total_timesteps}", "yellow"
            ))

        if cfg.reset_every_iteration:
            next_obs, _ = env.reset()
            actor_module.reset()

        done_episodes = 0
        successes = 0
        guided_done_episodes = 0
        guided_successes = 0
        random_done_episodes = 0
        random_successes = 0
        step = 0
        iteration_start_time = time.time()

        actor.eval()
        critic.eval()
        logger.info(f"[Rank {rank}] Starting data collection...")

        while step < steps_per_iteration:
            with torch.inference_mode():
                action_idx = 0
                first_obs_from_chunk = None

                while action_idx < n_action_steps and step < steps_per_iteration:
                    if first_obs_from_chunk is None:
                        first_obs_from_chunk = copy.deepcopy(next_obs)

                    dones_stored[step] = next_done
                    curr_obs = next_obs
                    sde_sampling = cfg.loss_mode == "dppo"
                    action, mdp_x_t_path, value = get_action_and_value(actor_module, critic, curr_obs, sde_sampling=sde_sampling)
                    assert mdp_x_t_path.shape == (num_envs_per_process, actor_module.config.sampling_steps, action_dim), \
                        f"mdp_x_t_path shape should be (num_envs_per_process, actor_module.config.sampling_steps, action_dim), but got {mdp_x_t_path.shape}"

                    next_obs, reward, next_done, truncated, _ = env.step(action)
                    if cfg.truncation_as_done:
                        next_done = next_done | truncated

                    for obs_key, obs_value in curr_obs.items():
                        if obs_key.startswith("observation.images."):
                            obs_images_stored_list[obs_key][step] = obs_value.cpu()
                    obs_state_stored[step] = curr_obs["observation.state"].cpu()

                    values_stored[step] = value.flatten().cpu()
                    actions_stored[step] = action.cpu()
                    mdp_x_t_paths_stored[step] = mdp_x_t_path.permute(1, 0, 2).cpu()
                    rewards_stored[step] = reward.view(-1).cpu()
                    next_done = next_done.view(-1).cpu()
                    terminals_stored[step] = next_done.bool()

                    if any(next_done):
                        done_episodes += next_done.sum().item()
                        successes += reward[torch.where(next_done)[0]].sum().item()
                        done_env_ids = torch.where(next_done)[0]
                        for env_idx_tensor in done_env_ids:
                            env_idx = int(env_idx_tensor.item())
                            success = int(reward[env_idx].item() == 1.0)
                            if env_idx in guided_rollout_env_ids:
                                guided_done_episodes += 1
                                guided_successes += success
                            else:
                                random_done_episodes += 1
                                random_successes += success
                        actor_module.reset(env_ids=done_env_ids)

                    if step > 0 and step % 100 == 0:
                        sps_local = step * num_envs_per_process / (time.time() - iteration_start_time + 1e-9)
                        if done_episodes > 0:
                            success_rate = successes / done_episodes
                            msg_sr = f"{success_rate:.2%} from {int(done_episodes)} episodes"
                        else:
                            msg_sr = "0.00% from 0 episodes"
                        logger.info(f"[Rank {rank}] step={step}/{steps_per_iteration}, "
                                    f"sps_local={sps_local:.2f}, SR={msg_sr}")

                    action_idx += 1
                    step += 1

                    # Only rank 0 advances a global_step approximation
                    if rank == 0:
                        global_step += cfg.num_envs

                # Get CFM values for the chunk
                actor_module.reset()
                next_done[:] = False

                curr_obs_chunk = first_obs_from_chunk
                curr_obs_chunk["action"] = actions_stored[step - action_idx:step].permute(1, 0, 2).to(device)

                # actions_stored
                # curr_obs_chunk["action"] = actions_stored[step - horizon:step] 
                
                # For FPO finetuning
                cfm_loss, cfm_loss_t, cfm_loss_eps = get_cfm_values(
                    actor, curr_obs_chunk, n_action_samples
                )
                cfm_losses_stored[step - action_idx:step] = cfm_loss.cpu()
                cfm_loss_ts_stored[step - action_idx:step] = cfm_loss_t.cpu()
                cfm_loss_epsilons_stored[step - action_idx:step] = cfm_loss_eps.cpu()

                # For DPPO finetuning
                # mdp_x_t_paths_stored: (n_action_steps, sampling_steps, num_envs_per_process, action_dim) 
                # ->: (num_envs_per_process, n_action_steps, sampling_steps, action_dim)
                curr_obs_chunk["mdp_x_t_path"] = mdp_x_t_paths_stored[step - action_idx:step].permute(2, 0, 1, 3).to(device)
                dppo_log_prob, dppo_entropy, dppo_sde_sigma = get_log_prob_and_entropy(actor, curr_obs_chunk) # dppo_log_prob: (num_envs, horizon, sampling_steps)
                dppo_log_probs_stored[step - action_idx:step] = dppo_log_prob.permute(1, 2, 0).cpu()
                # dppo_entropy and dppo_sde_sigma are not stored during data collection, only used during training

                # Should be used both for FPO and DPPO finetuning
                chunk_dones = dones_stored[step - action_idx:step]
                cfm_value_invalid_chunk = cfm_value_invalid_stored[step - action_idx:step]
                for i in range(num_envs_per_process):
                    chunk_done_idx = torch.where(chunk_dones[:, i] == 1)[0]
                    if len(chunk_done_idx) > 0:
                        cfm_value_invalid_chunk[chunk_done_idx:, i] = 1
                cfm_value_invalid_stored[step - action_idx:step] = cfm_value_invalid_chunk

        # Local SR for this rank
        success_rate_local = (successes / done_episodes) if done_episodes > 0 else 0.0
        sps_local_total = steps_per_iteration * num_envs_per_process / (time.time() - iteration_start_time + 1e-9)

        # Optionally reduce SR stats to rank 0 (not strictly necessary for training)
        if is_ddp:
            t = torch.tensor(
                [
                    successes,
                    done_episodes,
                    guided_successes,
                    guided_done_episodes,
                    random_successes,
                    random_done_episodes,
                ],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            (
                successes_global,
                episodes_global,
                guided_successes_global,
                guided_episodes_global,
                random_successes_global,
                random_episodes_global,
            ) = t.tolist()
            success_rate_global = (successes_global / episodes_global) if episodes_global > 0 else 0.0
        else:
            success_rate_global = success_rate_local
            successes_global = successes
            episodes_global = done_episodes
            guided_successes_global = guided_successes
            guided_episodes_global = guided_done_episodes
            random_successes_global = random_successes
            random_episodes_global = random_done_episodes

        guided_success_rate_global = (
            guided_successes_global / guided_episodes_global if guided_episodes_global > 0 else 0.0
        )
        random_success_rate_global = (
            random_successes_global / random_episodes_global if random_episodes_global > 0 else 0.0
        )

        if rank == 0:
            logger.info(
                f"[Rank {rank}] SR: {success_rate_global:.2%} from {int(episodes_global)} episodes | SPS_local(rank0 est): {sps_local_total:.2f}"
            )
            logger.info(
                f"[Rank {rank}] Source SR: guided={guided_success_rate_global:.2%} "
                f"({int(guided_successes_global)}/{int(guided_episodes_global)}) | "
                f"random={random_success_rate_global:.2%} "
                f"({int(random_successes_global)}/{int(random_episodes_global)})"
            )

        valid_cfm_steps = (cfm_value_invalid_stored == 0).sum().to(device=device, dtype=torch.float32)
        total_cfm_steps = torch.tensor(cfm_value_invalid_stored.numel(), device=device, dtype=torch.float32)
        if is_ddp:
            dist.all_reduce(valid_cfm_steps, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_cfm_steps, op=dist.ReduceOp.SUM)
        valid_cfm_action_fraction = (valid_cfm_steps / total_cfm_steps).item()
        if rank == 0:
            logger.info(
                f"[Rank {rank}] Valid CFM action fraction: {valid_cfm_action_fraction:.2%} "
                f"({int(valid_cfm_steps.item())}/{int(total_cfm_steps.item())})"
            )

        # ---------- Reshape for training ----------
        b_actions = actions_stored.reshape(-1, n_action_steps, num_envs_per_process, action_dim)
        b_actions = b_actions.permute(0, 2, 1, 3).reshape(-1, n_action_steps, action_dim)

        b_cfm_losses = cfm_losses_stored.reshape(-1, n_action_steps, num_envs_per_process, n_action_samples)
        b_cfm_losses = b_cfm_losses.permute(0, 2, 1, 3).reshape(-1, n_action_steps, n_action_samples)

        b_cfm_loss_ts = cfm_loss_ts_stored.reshape(-1, n_action_steps, num_envs_per_process, n_action_samples)
        b_cfm_loss_ts = b_cfm_loss_ts.permute(0, 2, 1, 3).reshape(-1, n_action_steps, n_action_samples)

        b_cfm_loss_epsilons = cfm_loss_epsilons_stored.reshape(-1, n_action_steps, num_envs_per_process, n_action_samples, action_dim)
        b_cfm_loss_epsilons = b_cfm_loss_epsilons.permute(0, 2, 1, 3, 4).reshape(-1, n_action_steps, n_action_samples, action_dim)

        b_dppo_log_probs = dppo_log_probs_stored.reshape(-1, n_action_steps, actor_module.config.sampling_steps, num_envs_per_process)
        b_dppo_log_probs = b_dppo_log_probs.permute(0, 3, 1, 2).reshape(-1, n_action_steps, actor_module.config.sampling_steps)

        b_mdp_x_t_paths = mdp_x_t_paths_stored.reshape(-1, n_action_steps, actor_module.config.sampling_steps, num_envs_per_process, action_dim)
        b_mdp_x_t_paths = b_mdp_x_t_paths.permute(0, 3, 1, 2, 4).reshape(-1, n_action_steps, actor_module.config.sampling_steps, action_dim)

        b_cfm_value_invalid = cfm_value_invalid_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_cfm_value_invalid = b_cfm_value_invalid.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_values = values_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_values = b_values.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_dones = dones_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_dones = b_dones.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_rewards = rewards_stored.reshape(-1, n_action_steps, num_envs_per_process)
        b_rewards = b_rewards.permute(0, 2, 1).reshape(-1, n_action_steps)

        b_terminals = terminals_stored.reshape(
            -1, n_action_steps, num_envs_per_process
        ).permute(0, 2, 1).reshape(-1, n_action_steps)

        b_obs_images = {
            k: obs_images_stored_list[k].reshape(-1, n_action_steps, num_envs_per_process, 3, img_h, img_w)
            for k in obs_images_stored_list.keys()
        }
        b_obs_images = {
            k: v.permute(0, 2, 1, 3, 4, 5).reshape(-1, n_action_steps, 3, img_h, img_w)
            for k, v in b_obs_images.items()
        }

        b_obs_state = obs_state_stored.reshape(-1, n_action_steps, num_envs_per_process, joint_pos_dim)
        b_obs_state = b_obs_state.permute(0, 2, 1, 3).reshape(-1, n_action_steps, joint_pos_dim)

        # ---------- Next value ----------
        if cfg.freeze_vision_encoder:
            actor_module.model.vision_encoder.eval()
        next_obs_cond = actor_module.normalize_inputs(next_obs)
        next_obs_cond = actor_module.model.encode_observations(next_obs_cond)
        next_value = critic(next_obs_cond).reshape(1, -1).cpu()

        # ---------- Advantages ----------
        advantages, returns = calculate_advantage(
            values_stored, next_value, rewards_stored, dones_stored, next_done,
            steps_per_iteration, cfg.discount, cfg.gae_lambda
        )
        b_advantages = advantages.reshape(-1, n_action_steps, num_envs_per_process).permute(0, 2, 1).reshape(-1, n_action_steps)
        b_returns = returns.reshape(-1, n_action_steps, num_envs_per_process).permute(0, 2, 1).reshape(-1, n_action_steps)

        mc_returns_stored, mc_valid_stored = discounted_returns_to_observed_terminal(
            rewards_stored,
            terminals_stored,
            cfg.discount,
        )
        b_mc_returns = mc_returns_stored.reshape(
            -1, n_action_steps, num_envs_per_process
        ).permute(0, 2, 1).reshape(-1, n_action_steps)
        b_mc_valid = mc_valid_stored.reshape(
            -1, n_action_steps, num_envs_per_process
        ).permute(0, 2, 1).reshape(-1, n_action_steps)

        direct_advantage_values = None
        if (
            direct_advantage_head is not None
            and iteration == cfg.direct_advantage_audit_iteration
        ):
            actor_module.eval()
            direct_advantage_head.eval()
            (
                direct_observations,
                direct_behavior_actions,
                direct_center_actions,
            ) = direct_advantage_inputs(
                b_actions=b_actions,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
                seed_offset=iteration * 7919,
            )
            with torch.no_grad():
                direct_advantage_values = direct_advantage_head(
                    direct_observations,
                    direct_behavior_actions,
                    direct_center_actions,
                ).cpu()

        candidate_b_values = None
        candidate_advantages = None
        if success_critic is not None:
            success_critic.eval()
            candidate_value_batches = []
            with torch.no_grad():
                for value_start in range(0, local_batch_size, minibatch_size):
                    value_end = min(value_start + minibatch_size, local_batch_size)
                    value_images = {
                        key: value[value_start:value_end].reshape(-1, 3, img_h, img_w).to(device)
                        for key, value in b_obs_images.items()
                    }
                    value_images["observation.state"] = b_obs_state[
                        value_start:value_end
                    ].reshape(-1, joint_pos_dim).to(device)
                    value_obs = actor_module.normalize_inputs(value_images)
                    value_cond = actor_module.model.encode_observations(value_obs)
                    candidate_output = success_critic(value_cond)
                    if cfg.discounted_success_critic_audit:
                        candidate_output = candidate_output.sigmoid()
                    candidate_value_batches.append(
                        candidate_output.reshape(
                            value_end - value_start, n_action_steps
                        ).cpu()
                    )
                candidate_b_values = torch.cat(candidate_value_batches, dim=0)
                candidate_values_stored = candidate_b_values.reshape(
                    -1, num_envs_per_process, n_action_steps
                ).permute(0, 2, 1).reshape(steps_per_iteration, num_envs_per_process)
                candidate_next_obs = actor_module.normalize_inputs(copy.deepcopy(next_obs))
                candidate_next_cond = actor_module.model.encode_observations(
                    candidate_next_obs
                )
                candidate_next_value = success_critic(candidate_next_cond)
                if cfg.discounted_success_critic_audit:
                    candidate_next_value = candidate_next_value.sigmoid()
                candidate_next_value = candidate_next_value.reshape(1, -1).cpu()
            candidate_advantages_stored, _ = calculate_advantage(
                candidate_values_stored,
                candidate_next_value,
                rewards_stored,
                dones_stored,
                next_done,
                steps_per_iteration,
                cfg.discount,
                cfg.gae_lambda,
            )
            candidate_advantages = candidate_advantages_stored.reshape(
                -1, n_action_steps, num_envs_per_process
            ).permute(0, 2, 1).reshape(-1, n_action_steps)

        rollout_advantage_mean = torch.tensor(float("nan"), device=device)
        rollout_advantage_std = torch.tensor(float("nan"), device=device)
        if cfg.norm_adv and cfg.advantage_normalization_scope == "rollout":
            advantage_columns = slice(0, 1) if cfg.do_chunk_level_ppo else slice(None)
            rollout_advantages = b_advantages[:, advantage_columns].to(device)
            rollout_valid_mask = (1.0 - b_cfm_value_invalid[:, advantage_columns]).to(
                device=device, dtype=torch.bool
            )
            valid_advantages = rollout_advantages[rollout_valid_mask]
            moment_sums = torch.stack(
                [
                    valid_advantages.sum(),
                    valid_advantages.square().sum(),
                    valid_advantages.new_tensor(valid_advantages.numel()),
                ]
            )
            if is_ddp:
                dist.all_reduce(moment_sums, op=dist.ReduceOp.SUM)
            if moment_sums[2] <= 0:
                raise RuntimeError("rollout advantage normalization requires a valid sample")
            rollout_advantage_mean = moment_sums[0] / moment_sums[2]
            rollout_advantage_variance = (
                moment_sums[1] / moment_sums[2] - rollout_advantage_mean.square()
            ).clamp_min(0.0)
            rollout_advantage_std = rollout_advantage_variance.sqrt()
            b_advantages[:, advantage_columns] = normalize_advantages_from_moments(
                rollout_advantages,
                rollout_advantage_mean,
                rollout_advantage_variance,
            ).cpu()
            if rank == 0:
                logger.info(
                    "[Rank %d] Rollout advantage moments: mean=%.6f std=%.6f count=%d",
                    rank,
                    rollout_advantage_mean.item(),
                    rollout_advantage_std.item(),
                    int(moment_sums[2].item()),
                )

        if (
            cfg.terminal_consistency_filter_audit
            and iteration == cfg.terminal_consistency_filter_audit_iteration
        ):
            eligible_mask = b_mc_valid[:, 0].bool() & (
                b_cfm_value_invalid.sum(dim=1) == 0
            )
            eligible_indices = torch.where(eligible_mask)[0]
            required_chunks = cfg.terminal_consistency_filter_audit_chunks
            if eligible_indices.numel() < required_chunks:
                raise RuntimeError(
                    "terminal-consistency filter audit requires "
                    f"{required_chunks} fully valid labeled chunks, "
                    f"found {eligible_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 11003
            )
            audit_indices = eligible_indices[
                torch.randperm(
                    eligible_indices.numel(), generator=audit_generator
                )[:required_chunks]
            ]
            replica_size = required_chunks // 2
            replica_results = []
            actor_module.eval()
            for replica in range(2):
                replica_indices = audit_indices[
                    replica * replica_size : (replica + 1) * replica_size
                ]
                control_weights = b_advantages[replica_indices, 0].float()
                control_std = control_weights.std()
                if not torch.isfinite(control_std) or control_std <= 0:
                    raise RuntimeError(
                        f"terminal-consistency replica {replica} has degenerate GAE weights"
                    )
                control_weights = (
                    control_weights - control_weights.mean()
                ) / (control_std + 1e-8)
                reference_weights = b_mc_returns[replica_indices, 0].float()
                reference_weights = reference_weights - reference_weights.mean()
                candidate_weights, retained_mask = terminal_consistent_weights(
                    control_weights, reference_weights
                )
                positive_retained = int((candidate_weights > 0).sum().item())
                negative_retained = int((candidate_weights < 0).sum().item())
                retained_fraction = float(retained_mask.float().mean().item())

                _, reference_gradient = replay_audit_gradient(
                    replica_indices,
                    reference_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, control_gradient = replay_audit_gradient(
                    replica_indices,
                    control_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, candidate_gradient = replay_audit_gradient(
                    replica_indices,
                    candidate_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                vectors = {
                    "reference": reference_gradient,
                    "control": control_gradient,
                    "candidate": candidate_gradient,
                }
                norms = {name: vector.norm() for name, vector in vectors.items()}
                finite_nonzero = all(
                    torch.isfinite(vector).all().item()
                    and torch.isfinite(norms[name]).item()
                    and norms[name].item() > 0
                    for name, vector in vectors.items()
                )
                if not finite_nonzero:
                    raise RuntimeError(
                        "terminal-consistency audit produced a non-finite or zero gradient"
                    )

                def vector_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
                    return stable_vector_cosine(first, second)

                control_reference_cosine = vector_cosine(
                    control_gradient, reference_gradient
                )
                candidate_reference_cosine = vector_cosine(
                    candidate_gradient, reference_gradient
                )
                candidate_control_cosine = vector_cosine(
                    candidate_gradient, control_gradient
                )
                norm_ratio = float(
                    (norms["candidate"] / norms["control"]).item()
                )
                gates = {
                    "finite_nonzero_and_sign_support": (
                        finite_nonzero
                        and positive_retained >= 8
                        and negative_retained >= 8
                    ),
                    "candidate_reference_absolute": candidate_reference_cosine >= 0.75,
                    "candidate_reference_gain": (
                        candidate_reference_cosine - control_reference_cosine >= 0.10
                    ),
                    "candidate_control_direction": candidate_control_cosine >= 0.75,
                    "candidate_control_norm": 0.5 <= norm_ratio <= 1.2,
                    "active_nonsparse_filter": 0.4 <= retained_fraction <= 0.9,
                }
                replica_results.append(
                    {
                        "replica": replica,
                        "num_chunks": replica_size,
                        "retained_chunks": int(retained_mask.sum().item()),
                        "retained_fraction": retained_fraction,
                        "positive_retained": positive_retained,
                        "negative_retained": negative_retained,
                        "reference_gradient_norm": float(norms["reference"].item()),
                        "control_gradient_norm": float(norms["control"].item()),
                        "candidate_gradient_norm": float(norms["candidate"].item()),
                        "control_reference_cosine": control_reference_cosine,
                        "candidate_reference_cosine": candidate_reference_cosine,
                        "candidate_reference_cosine_gain": (
                            candidate_reference_cosine - control_reference_cosine
                        ),
                        "candidate_control_cosine": candidate_control_cosine,
                        "candidate_control_norm_ratio": norm_ratio,
                        "gates": gates,
                        "passed": all(gates.values()),
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            audit_result = {
                "iteration": iteration,
                "seed": cfg.seed,
                "num_eligible_chunks": int(eligible_indices.numel()),
                "num_chunks_audited": required_chunks,
                "num_replicas": 2,
                "cfm_samples": n_action_samples,
                "replicas": replica_results,
                "passed": all(result["passed"] for result in replica_results),
            }
            audit_output_path = (
                Path(cfg.terminal_consistency_filter_audit_output_json)
                if cfg.terminal_consistency_filter_audit_output_json is not None
                else run_dir / "terminal_consistency_filter_audit.json"
            )
            audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            audit_output_path.write_text(
                json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Terminal-consistency filter audit: %s",
                json.dumps(audit_result, sort_keys=True),
            )

        if (
            cfg.median_microbatch_gradient_audit
            and iteration == cfg.median_microbatch_gradient_audit_iteration
        ):
            eligible_mask = b_mc_valid[:, 0].bool() & (
                b_cfm_value_invalid.sum(dim=1) == 0
            )
            eligible_indices = torch.where(eligible_mask)[0]
            required_chunks = cfg.median_microbatch_gradient_audit_chunks
            if eligible_indices.numel() < required_chunks:
                raise RuntimeError(
                    "median-microbatch gradient audit requires "
                    f"{required_chunks} fully valid labeled chunks, "
                    f"found {eligible_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 10009
            )
            audit_indices = eligible_indices[
                torch.randperm(
                    eligible_indices.numel(), generator=audit_generator
                )[:required_chunks]
            ]
            replica_size = required_chunks // 2
            microbatch_size = replica_size // 4
            replica_results = []
            actor_module.eval()
            for replica in range(2):
                replica_start = replica * replica_size
                replica_indices = audit_indices[
                    replica_start : replica_start + replica_size
                ]
                reference_weights = b_mc_returns[replica_indices, 0].float()
                reference_weights = reference_weights - reference_weights.mean()
                _, reference_gradient = replay_audit_gradient(
                    replica_indices,
                    reference_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )

                microbatch_gradients = []
                microbatch_weight_stats = []
                for microbatch in range(4):
                    microbatch_start = microbatch * microbatch_size
                    microbatch_indices = replica_indices[
                        microbatch_start : microbatch_start + microbatch_size
                    ]
                    weights = b_advantages[microbatch_indices, 0].float()
                    weight_mean = weights.mean()
                    weight_std = weights.std()
                    if not torch.isfinite(weight_std) or weight_std <= 0:
                        raise RuntimeError(
                            "median-microbatch gradient audit has degenerate GAE weights "
                            f"in replica {replica}, microbatch {microbatch}"
                        )
                    normalized_weights = (weights - weight_mean) / (weight_std + 1e-8)
                    _, gradient = replay_audit_gradient(
                        microbatch_indices,
                        normalized_weights,
                        b_actions=b_actions,
                        b_cfm_losses=b_cfm_losses,
                        b_cfm_loss_ts=b_cfm_loss_ts,
                        b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                        b_cfm_value_invalid=b_cfm_value_invalid,
                        b_obs_images=b_obs_images,
                        b_obs_state=b_obs_state,
                    )
                    microbatch_gradients.append(gradient)
                    microbatch_weight_stats.append(
                        {
                            "microbatch": microbatch,
                            "gae_mean": float(weight_mean.item()),
                            "gae_std": float(weight_std.item()),
                            "gradient_norm": float(gradient.norm().item()),
                        }
                    )

                if cfg.median_microbatch_gradient_method == "coordinate":
                    control_gradient, candidate_gradient = aggregate_microbatch_gradients(
                        microbatch_gradients
                    )
                    candidate_iterations = 0
                    candidate_converged = True
                else:
                    control_gradient = torch.stack(microbatch_gradients).mean(dim=0)
                    geometric_result = geometric_median(microbatch_gradients)
                    candidate_gradient = geometric_result.value
                    candidate_iterations = geometric_result.iterations
                    candidate_converged = geometric_result.converged
                vectors = {
                    "reference": reference_gradient,
                    "control": control_gradient,
                    "candidate": candidate_gradient,
                    **{
                        f"microbatch_{index}": gradient
                        for index, gradient in enumerate(microbatch_gradients)
                    },
                }
                norms = {name: vector.norm() for name, vector in vectors.items()}
                finite_nonzero = all(
                    torch.isfinite(vector).all().item()
                    and torch.isfinite(norms[name]).item()
                    and norms[name].item() > 0
                    for name, vector in vectors.items()
                )
                if not finite_nonzero:
                    raise RuntimeError(
                        "median-microbatch gradient audit produced a non-finite or zero gradient"
                    )

                def vector_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
                    return float(
                        torch.nn.functional.cosine_similarity(
                            first.unsqueeze(0), second.unsqueeze(0), dim=1
                        ).item()
                    )

                control_reference_cosine = vector_cosine(
                    control_gradient, reference_gradient
                )
                candidate_reference_cosine = vector_cosine(
                    candidate_gradient, reference_gradient
                )
                candidate_control_cosine = vector_cosine(
                    candidate_gradient, control_gradient
                )
                norm_ratio = float(
                    (norms["candidate"] / norms["control"]).item()
                )
                microbatch_reference_cosines = [
                    vector_cosine(gradient, reference_gradient)
                    for gradient in microbatch_gradients
                ]
                median_microbatch_reference_cosine = float(
                    middle_pair_mean(
                        torch.tensor(microbatch_reference_cosines)
                    ).item()
                )
                gates = {
                    "finite_nonzero": finite_nonzero,
                    "candidate_converged": candidate_converged,
                    "candidate_reference_absolute": candidate_reference_cosine >= 0.75,
                    "candidate_reference_gain": (
                        candidate_reference_cosine - control_reference_cosine >= 0.10
                    ),
                    "candidate_control_direction": candidate_control_cosine >= 0.75,
                    "candidate_control_norm": 0.5 <= norm_ratio <= 1.2,
                    "not_below_typical_microbatch": (
                        candidate_reference_cosine
                        >= median_microbatch_reference_cosine
                    ),
                }
                replica_results.append(
                    {
                        "replica": replica,
                        "num_chunks": replica_size,
                        "microbatch_size": microbatch_size,
                        "reference_gradient_norm": float(norms["reference"].item()),
                        "control_gradient_norm": float(norms["control"].item()),
                        "candidate_gradient_norm": float(norms["candidate"].item()),
                        "candidate_solver_iterations": candidate_iterations,
                        "candidate_solver_converged": candidate_converged,
                        "control_reference_cosine": control_reference_cosine,
                        "candidate_reference_cosine": candidate_reference_cosine,
                        "candidate_reference_cosine_gain": (
                            candidate_reference_cosine - control_reference_cosine
                        ),
                        "candidate_control_cosine": candidate_control_cosine,
                        "candidate_control_norm_ratio": norm_ratio,
                        "microbatch_reference_cosines": microbatch_reference_cosines,
                        "median_microbatch_reference_cosine": (
                            median_microbatch_reference_cosine
                        ),
                        "microbatches": microbatch_weight_stats,
                        "gates": gates,
                        "passed": all(gates.values()),
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            audit_result = {
                "iteration": iteration,
                "seed": cfg.seed,
                "num_eligible_chunks": int(eligible_indices.numel()),
                "num_chunks_audited": required_chunks,
                "num_replicas": 2,
                "microbatches_per_replica": 4,
                "cfm_samples": n_action_samples,
                "aggregation_method": cfg.median_microbatch_gradient_method,
                "replicas": replica_results,
                "passed": all(result["passed"] for result in replica_results),
            }
            audit_output_path = (
                Path(cfg.median_microbatch_gradient_audit_output_json)
                if cfg.median_microbatch_gradient_audit_output_json is not None
                else run_dir / "median_microbatch_gradient_audit.json"
            )
            audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            audit_output_path.write_text(
                json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Median-microbatch gradient audit: %s",
                json.dumps(audit_result, sort_keys=True),
            )

        if (
            cfg.advantage_sign_pcgrad_audit
            and iteration == cfg.advantage_sign_pcgrad_audit_iteration
        ):
            eligible_mask = b_mc_valid[:, 0].bool() & (
                b_cfm_value_invalid.sum(dim=1) == 0
            )
            eligible_indices = torch.where(eligible_mask)[0]
            required_chunks = cfg.advantage_sign_pcgrad_audit_chunks
            if eligible_indices.numel() < required_chunks:
                raise RuntimeError(
                    "advantage-sign PCGrad audit requires "
                    f"{required_chunks} fully valid labeled chunks, "
                    f"found {eligible_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 8081
            )
            audit_indices = eligible_indices[
                torch.randperm(
                    eligible_indices.numel(), generator=audit_generator
                )[:required_chunks]
            ]
            audit_batch_size = required_chunks // 2
            batch_results = []
            actor_module.eval()
            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * audit_batch_size,
                    (audit_batch + 1) * audit_batch_size,
                )
                batch_indices = audit_indices[batch_slice]
                control_weights = b_advantages[batch_indices, 0].float()
                control_std = control_weights.std()
                if not torch.isfinite(control_std) or control_std <= 0:
                    raise RuntimeError(
                        "advantage-sign PCGrad audit has degenerate GAE weights"
                    )
                control_weights = (
                    control_weights - control_weights.mean()
                ) / (control_std + 1e-8)
                positive_mask = control_weights > 0
                negative_mask = control_weights < 0
                positive_count = int(positive_mask.sum().item())
                negative_count = int(negative_mask.sum().item())
                if min(positive_count, negative_count) < cfg.advantage_sign_pcgrad_min_per_sign:
                    raise RuntimeError(
                        "advantage-sign PCGrad audit lacks sign support in batch "
                        f"{audit_batch}: positive={positive_count}, negative={negative_count}"
                    )

                positive_weights = torch.where(
                    positive_mask, control_weights, torch.zeros_like(control_weights)
                )
                negative_weights = torch.where(
                    negative_mask, control_weights, torch.zeros_like(control_weights)
                )
                reference_weights = b_mc_returns[batch_indices, 0].float()
                reference_weights = reference_weights - reference_weights.mean()

                _, reference_gradient = replay_audit_gradient(
                    batch_indices,
                    reference_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, control_gradient = replay_audit_gradient(
                    batch_indices,
                    control_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, positive_gradient = replay_audit_gradient(
                    batch_indices,
                    positive_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, negative_gradient = replay_audit_gradient(
                    batch_indices,
                    negative_weights,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                projected_negative_list, sign_dot, projection_active = (
                    project_conflicting_gradient(
                        [negative_gradient], [positive_gradient]
                    )
                )
                projected_negative = projected_negative_list[0]
                if projected_negative is None:
                    raise RuntimeError("advantage-sign projection lost its gradient")
                candidate_gradient = positive_gradient + projected_negative
                recombined_gradient = positive_gradient + negative_gradient

                vectors = {
                    "reference": reference_gradient,
                    "control": control_gradient,
                    "positive": positive_gradient,
                    "negative": negative_gradient,
                    "candidate": candidate_gradient,
                }
                norms = {name: vector.norm() for name, vector in vectors.items()}
                if any(
                    (not torch.isfinite(norm).item()) or norm.item() <= 0
                    for norm in norms.values()
                ):
                    raise RuntimeError(
                        "advantage-sign PCGrad audit produced an invalid gradient norm"
                    )
                decomposition_cosine = torch.nn.functional.cosine_similarity(
                    control_gradient.unsqueeze(0),
                    recombined_gradient.unsqueeze(0),
                    dim=1,
                ).item()
                if decomposition_cosine < 0.99999:
                    raise RuntimeError(
                        "advantage-sign gradient decomposition does not reproduce control"
                    )
                control_reference_cosine = torch.nn.functional.cosine_similarity(
                    control_gradient.unsqueeze(0),
                    reference_gradient.unsqueeze(0),
                    dim=1,
                ).item()
                candidate_reference_cosine = torch.nn.functional.cosine_similarity(
                    candidate_gradient.unsqueeze(0),
                    reference_gradient.unsqueeze(0),
                    dim=1,
                ).item()
                candidate_control_cosine = torch.nn.functional.cosine_similarity(
                    candidate_gradient.unsqueeze(0),
                    control_gradient.unsqueeze(0),
                    dim=1,
                ).item()
                batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": audit_batch_size,
                        "positive_chunks": positive_count,
                        "negative_chunks": negative_count,
                        "positive_gradient_norm": float(norms["positive"].item()),
                        "negative_gradient_norm": float(norms["negative"].item()),
                        "control_gradient_norm": float(norms["control"].item()),
                        "candidate_gradient_norm": float(norms["candidate"].item()),
                        "reference_gradient_norm": float(norms["reference"].item()),
                        "positive_negative_gradient_dot": float(sign_dot.item()),
                        "positive_negative_gradient_cosine": float(
                            torch.nn.functional.cosine_similarity(
                                positive_gradient.unsqueeze(0),
                                negative_gradient.unsqueeze(0),
                                dim=1,
                            ).item()
                        ),
                        "projection_active": projection_active,
                        "control_decomposition_cosine": decomposition_cosine,
                        "control_reference_cosine": control_reference_cosine,
                        "candidate_reference_cosine": candidate_reference_cosine,
                        "candidate_reference_cosine_gain": (
                            candidate_reference_cosine - control_reference_cosine
                        ),
                        "candidate_control_cosine": candidate_control_cosine,
                        "candidate_control_norm_ratio": float(
                            (norms["candidate"] / norms["control"]).item()
                        ),
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            audit_result = {
                "iteration": iteration,
                "num_eligible_chunks": int(eligible_indices.numel()),
                "num_chunks_audited": required_chunks,
                "num_batches": 2,
                "min_per_sign": cfg.advantage_sign_pcgrad_min_per_sign,
                "cfm_samples": n_action_samples,
                "batches": batch_results,
            }
            audit_output_path = (
                Path(cfg.advantage_sign_pcgrad_audit_output_json)
                if cfg.advantage_sign_pcgrad_audit_output_json is not None
                else run_dir / "advantage_sign_pcgrad_audit.json"
            )
            audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            audit_output_path.write_text(
                json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Advantage-sign PCGrad audit: %s",
                json.dumps(audit_result, sort_keys=True),
            )

        if (
            cfg.temporal_ratio_clipping_audit
            and iteration == cfg.temporal_ratio_clipping_audit_iteration
        ):
            eligible_mask = b_mc_valid[:, 0].bool() & (
                b_cfm_value_invalid.sum(dim=1) == 0
            )
            eligible_indices = torch.where(eligible_mask)[0]
            required_chunks = cfg.temporal_ratio_clipping_audit_chunks
            if eligible_indices.numel() < required_chunks:
                raise RuntimeError(
                    "temporal-ratio clipping audit requires "
                    f"{required_chunks} fully valid labeled chunks, "
                    f"found {eligible_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 9091
            )
            audit_indices = eligible_indices[
                torch.randperm(
                    eligible_indices.numel(), generator=audit_generator
                )[:required_chunks]
            ]
            audit_batch_size = required_chunks // 2
            batch_results = []
            actor_module.eval()
            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * audit_batch_size,
                    (audit_batch + 1) * audit_batch_size,
                )
                batch_indices = audit_indices[batch_slice]
                weights = b_advantages[batch_indices, 0].float()
                weight_std = weights.std()
                if not torch.isfinite(weight_std) or weight_std <= 0:
                    raise RuntimeError(
                        "temporal-ratio clipping audit has degenerate GAE weights"
                    )
                weights = (weights - weights.mean()) / (weight_std + 1e-8)
                if not (weights > 0).any():
                    raise RuntimeError(
                        "temporal-ratio clipping audit requires positive advantages"
                    )

                control_pre_ratios, control_pre_loss = replay_audit_objective(
                    batch_indices,
                    weights,
                    temporal_clipping=False,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                parameters, control_pre_gradient = autograd_list(control_pre_loss)
                candidate_pre_ratios, candidate_pre_loss = replay_audit_objective(
                    batch_indices,
                    weights,
                    temporal_clipping=True,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, candidate_pre_gradient = autograd_list(candidate_pre_loss)
                base_values = [parameter.detach().clone() for parameter in parameters]
                pre_cosine = float(
                    list_gradient_cosine(
                        candidate_pre_gradient, control_pre_gradient
                    ).item()
                )
                control_pre_norm = float(gradient_norm(control_pre_gradient).item())
                candidate_pre_norm = float(gradient_norm(candidate_pre_gradient).item())
                step_scale = apply_virtual_actor_step(parameters, control_pre_gradient)

                control_post_ratios, control_post_loss = replay_audit_objective(
                    batch_indices,
                    weights,
                    temporal_clipping=False,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, control_post_gradient = autograd_list(control_post_loss)
                candidate_post_ratios, candidate_post_loss = replay_audit_objective(
                    batch_indices,
                    weights,
                    temporal_clipping=True,
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, candidate_post_gradient = autograd_list(candidate_post_loss)
                restore_parameters(parameters, base_values)

                control_post_norm = float(gradient_norm(control_post_gradient).item())
                candidate_post_norm = float(gradient_norm(candidate_post_gradient).item())
                control_post_cosine = float(
                    list_gradient_cosine(
                        control_post_gradient, control_pre_gradient
                    ).item()
                )
                candidate_post_cosine = float(
                    list_gradient_cosine(
                        candidate_post_gradient, control_pre_gradient
                    ).item()
                )
                valid_mask = torch.ones(
                    (audit_batch_size, n_action_steps),
                    device=device,
                    dtype=torch.bool,
                )
                device_weights = weights.to(device)
                control_active = float(
                    positive_active_fraction(
                        control_post_ratios,
                        device_weights,
                        valid_mask,
                        cfg.clip_coef,
                    ).item()
                )
                candidate_active = float(
                    positive_active_fraction(
                        candidate_post_ratios,
                        device_weights,
                        valid_mask,
                        cfg.clip_coef,
                    ).item()
                )
                tensors = (
                    control_pre_ratios,
                    candidate_pre_ratios,
                    control_post_ratios,
                    candidate_post_ratios,
                    control_pre_loss,
                    candidate_pre_loss,
                    control_post_loss,
                    candidate_post_loss,
                )
                if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
                    raise RuntimeError(
                        "temporal-ratio clipping audit produced non-finite values"
                    )
                if min(
                    control_pre_norm,
                    candidate_pre_norm,
                    control_post_norm,
                    candidate_post_norm,
                ) <= 0:
                    raise RuntimeError(
                        "temporal-ratio clipping audit produced a zero gradient"
                    )
                batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": audit_batch_size,
                        "virtual_step_scale": step_scale,
                        "pre_gradient_cosine": pre_cosine,
                        "pre_candidate_control_norm_ratio": (
                            candidate_pre_norm / control_pre_norm
                        ),
                        "control_post_gradient_cosine": control_post_cosine,
                        "candidate_post_gradient_cosine": candidate_post_cosine,
                        "candidate_post_cosine_gain": (
                            candidate_post_cosine - control_post_cosine
                        ),
                        "control_positive_active_fraction": control_active,
                        "candidate_positive_active_fraction": candidate_active,
                        "candidate_positive_active_gain": (
                            candidate_active - control_active
                        ),
                        "candidate_control_post_norm_ratio": (
                            candidate_post_norm / control_post_norm
                        ),
                        "control_pre_loss": float(control_pre_loss.detach().item()),
                        "candidate_pre_loss": float(candidate_pre_loss.detach().item()),
                        "control_post_loss": float(control_post_loss.detach().item()),
                        "candidate_post_loss": float(candidate_post_loss.detach().item()),
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            audit_result = {
                "iteration": iteration,
                "num_eligible_chunks": int(eligible_indices.numel()),
                "num_chunks_audited": required_chunks,
                "num_batches": 2,
                "cfm_samples": n_action_samples,
                "action_timesteps": n_action_steps,
                "batches": batch_results,
            }
            audit_output_path = (
                Path(cfg.temporal_ratio_clipping_audit_output_json)
                if cfg.temporal_ratio_clipping_audit_output_json is not None
                else run_dir / "temporal_ratio_clipping_audit.json"
            )
            audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            audit_output_path.write_text(
                json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Temporal-ratio clipping audit: %s",
                json.dumps(audit_result, sort_keys=True),
            )

        critic_side_audit_iteration = (
            cfg.discounted_success_critic_audit_iteration
            if cfg.discounted_success_critic_audit
            else cfg.critic_warmup_scheduler_audit_iteration
        )
        if (
            (cfg.discounted_success_critic_audit or cfg.critic_warmup_scheduler_audit)
            and iteration == critic_side_audit_iteration
        ):
            if candidate_b_values is None or candidate_advantages is None:
                raise RuntimeError("discounted-success critic audit has no candidate values")
            value_mask = b_mc_valid.bool()
            control_value_targets = b_mc_returns[value_mask]
            control_value_predictions = b_values[value_mask]
            candidate_value_predictions = candidate_b_values[value_mask]
            if control_value_targets.numel() < 2:
                raise RuntimeError("discounted-success critic audit has too few value targets")
            control_value_mse = torch.mean(
                (control_value_predictions - control_value_targets).square()
            ).item()
            candidate_value_mse = torch.mean(
                (candidate_value_predictions - control_value_targets).square()
            ).item()
            control_value_spearman = spearman_rank_correlation(
                control_value_predictions, control_value_targets
            )
            candidate_value_spearman = spearman_rank_correlation(
                candidate_value_predictions, control_value_targets
            )

            gradient_candidate_mask = b_mc_valid[:, 0].bool() & (
                b_cfm_value_invalid.sum(dim=1) == 0
            )
            gradient_candidate_indices = torch.where(gradient_candidate_mask)[0]
            required_chunks = (
                cfg.discounted_success_critic_audit_chunks
                if cfg.discounted_success_critic_audit
                else cfg.critic_warmup_scheduler_audit_chunks
            )
            if gradient_candidate_indices.numel() < required_chunks:
                raise RuntimeError(
                    "discounted-success critic audit requires "
                    f"{required_chunks} fully valid labeled chunks, "
                    f"found {gradient_candidate_indices.numel()}"
                )
            gradient_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 5011
            )
            gradient_indices = gradient_candidate_indices[
                torch.randperm(
                    gradient_candidate_indices.numel(), generator=gradient_generator
                )[:required_chunks]
            ]
            control_weights = b_advantages[gradient_indices, 0].float()
            candidate_weights = candidate_advantages[gradient_indices, 0].float()
            reference_weights = b_mc_returns[gradient_indices, 0].float()
            reference_weights = reference_weights - reference_weights.mean()
            audit_batch_size = required_chunks // 2
            gradient_batch_results = []
            actor_module.eval()
            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * audit_batch_size,
                    (audit_batch + 1) * audit_batch_size,
                )
                batch_indices = gradient_indices[batch_slice]
                _, reference_gradient = replay_audit_gradient(
                    batch_indices,
                    reference_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, control_gradient = replay_audit_gradient(
                    batch_indices,
                    control_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, candidate_gradient = replay_audit_gradient(
                    batch_indices,
                    candidate_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                reference_norm = reference_gradient.norm().item()
                if reference_norm <= 0 or not torch.isfinite(reference_gradient).all():
                    raise RuntimeError("discounted-success audit produced an invalid reference gradient")
                control_cosine = torch.nn.functional.cosine_similarity(
                    control_gradient.unsqueeze(0), reference_gradient.unsqueeze(0), dim=1
                ).item()
                candidate_cosine = torch.nn.functional.cosine_similarity(
                    candidate_gradient.unsqueeze(0), reference_gradient.unsqueeze(0), dim=1
                ).item()
                gradient_batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": audit_batch_size,
                        "reference_gradient_norm": reference_norm,
                        "control_gradient_cosine_to_reference": control_cosine,
                        "candidate_gradient_cosine_to_reference": candidate_cosine,
                        "candidate_cosine_gain": candidate_cosine - control_cosine,
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            success_critic_result = {
                "iteration": iteration,
                "num_value_targets": int(control_value_targets.numel()),
                "num_gradient_chunks_available": int(
                    gradient_candidate_indices.numel()
                ),
                "num_gradient_chunks_audited": required_chunks,
                "control_value_mse": control_value_mse,
                "candidate_value_mse": candidate_value_mse,
                "candidate_value_mse_relative_change": (
                    candidate_value_mse / max(control_value_mse, 1e-12) - 1.0
                ),
                "control_value_spearman": control_value_spearman,
                "candidate_value_spearman": candidate_value_spearman,
                "batches": gradient_batch_results,
            }
            if cfg.critic_warmup_scheduler_audit:
                control_parameters = list(critic.parameters())
                candidate_parameters = list(success_critic.parameters())
                control_delta_sq = 0.0
                candidate_delta_sq = 0.0
                cross_delta_sq = 0.0
                for initial, control_parameter, candidate_parameter in zip(
                    critic_audit_initial_parameters,
                    control_parameters,
                    candidate_parameters,
                    strict=True,
                ):
                    control_cpu = control_parameter.detach().cpu()
                    candidate_cpu = candidate_parameter.detach().cpu()
                    control_delta_sq += float((control_cpu - initial).square().sum().item())
                    candidate_delta_sq += float((candidate_cpu - initial).square().sum().item())
                    cross_delta_sq += float(
                        (candidate_cpu - control_cpu).square().sum().item()
                    )
                success_critic_result.update(
                    {
                        "control_parameter_delta_norm": control_delta_sq**0.5,
                        "candidate_parameter_delta_norm": candidate_delta_sq**0.5,
                        "candidate_control_parameter_distance": cross_delta_sq**0.5,
                        "control_iteration1_learning_rate": 0.0,
                        "candidate_iteration1_learning_rate": float(
                            optimizer_success_critic.param_groups[0]["lr"]
                        ),
                    }
                )
            critic_audit_output_json = (
                cfg.discounted_success_critic_audit_output_json
                if cfg.discounted_success_critic_audit
                else cfg.critic_warmup_scheduler_audit_output_json
            )
            critic_audit_filename = (
                "discounted_success_critic_audit.json"
                if cfg.discounted_success_critic_audit
                else "critic_warmup_scheduler_audit.json"
            )
            success_critic_output_path = (
                Path(critic_audit_output_json)
                if critic_audit_output_json is not None
                else run_dir / critic_audit_filename
            )
            success_critic_output_path.parent.mkdir(parents=True, exist_ok=True)
            success_critic_output_path.write_text(
                json.dumps(success_critic_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Discounted-success critic audit: %s",
                json.dumps(success_critic_result, sort_keys=True),
            )

        if (
            direct_advantage_head is not None
            and iteration == cfg.direct_advantage_audit_iteration
        ):
            if direct_advantage_values is None or not direct_advantage_training_losses:
                raise RuntimeError("direct-advantage audit has no trained candidate weights")
            direct_candidate_mask = b_mc_valid[:, 0].bool() & (
                b_cfm_value_invalid.sum(dim=1) == 0
            )
            direct_candidate_indices = torch.where(direct_candidate_mask)[0]
            required_chunks = cfg.direct_advantage_audit_chunks
            if direct_candidate_indices.numel() < required_chunks:
                raise RuntimeError(
                    "direct-advantage audit requires "
                    f"{required_chunks} fully valid labeled chunks, "
                    f"found {direct_candidate_indices.numel()}"
                )
            rank_targets = b_mc_returns[direct_candidate_indices, 0].float()
            rank_control = b_advantages[direct_candidate_indices, 0].float()
            rank_candidate = direct_advantage_values[direct_candidate_indices].float()
            if not torch.isfinite(rank_candidate).all() or rank_candidate.std() <= 0:
                raise RuntimeError("direct-advantage audit produced invalid candidate weights")
            control_spearman = spearman_rank_correlation(rank_control, rank_targets)
            candidate_spearman = spearman_rank_correlation(rank_candidate, rank_targets)

            gradient_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 6151
            )
            gradient_indices = direct_candidate_indices[
                torch.randperm(
                    direct_candidate_indices.numel(), generator=gradient_generator
                )[:required_chunks]
            ]
            control_weights = b_advantages[gradient_indices, 0].float()
            candidate_weights = direct_advantage_values[gradient_indices].float()
            reference_weights = b_mc_returns[gradient_indices, 0].float()
            reference_weights = reference_weights - reference_weights.mean()
            audit_batch_size = required_chunks // 2
            gradient_batch_results = []
            actor_module.eval()
            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * audit_batch_size,
                    (audit_batch + 1) * audit_batch_size,
                )
                batch_indices = gradient_indices[batch_slice]
                _, reference_gradient = replay_audit_gradient(
                    batch_indices,
                    reference_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, control_gradient = replay_audit_gradient(
                    batch_indices,
                    control_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, candidate_gradient = replay_audit_gradient(
                    batch_indices,
                    candidate_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                reference_norm = reference_gradient.norm().item()
                if reference_norm <= 0 or not torch.isfinite(reference_gradient).all():
                    raise RuntimeError("direct-advantage audit produced an invalid reference gradient")
                control_cosine = torch.nn.functional.cosine_similarity(
                    control_gradient.unsqueeze(0), reference_gradient.unsqueeze(0), dim=1
                ).item()
                candidate_cosine = torch.nn.functional.cosine_similarity(
                    candidate_gradient.unsqueeze(0), reference_gradient.unsqueeze(0), dim=1
                ).item()
                gradient_batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": audit_batch_size,
                        "reference_gradient_norm": reference_norm,
                        "control_gradient_cosine_to_reference": control_cosine,
                        "candidate_gradient_cosine_to_reference": candidate_cosine,
                        "candidate_cosine_gain": candidate_cosine - control_cosine,
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            first_loss = direct_advantage_training_losses[0]
            final_loss = direct_advantage_training_losses[-1]
            direct_advantage_result = {
                "iteration": iteration,
                "num_rank_chunks": int(direct_candidate_indices.numel()),
                "num_gradient_chunks_audited": required_chunks,
                "center_samples": cfg.direct_advantage_center_samples,
                "horizon_chunks": cfg.direct_advantage_horizon_chunks,
                "control_advantage_spearman": control_spearman,
                "candidate_advantage_spearman": candidate_spearman,
                "candidate_spearman_gain": candidate_spearman - control_spearman,
                "candidate_weight_mean": float(rank_candidate.mean().item()),
                "candidate_weight_std": float(rank_candidate.std(unbiased=False).item()),
                "training_losses": direct_advantage_training_losses,
                "training_loss_relative_change": final_loss / max(first_loss, 1e-12) - 1.0,
                "batches": gradient_batch_results,
            }
            direct_advantage_output_path = (
                Path(cfg.direct_advantage_audit_output_json)
                if cfg.direct_advantage_audit_output_json is not None
                else run_dir / "direct_advantage_audit.json"
            )
            direct_advantage_output_path.parent.mkdir(parents=True, exist_ok=True)
            direct_advantage_output_path.write_text(
                json.dumps(direct_advantage_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Direct-advantage audit: %s",
                json.dumps(direct_advantage_result, sort_keys=True),
            )

        if cfg.rank_advantage_audit and iteration == cfg.rank_advantage_audit_iteration:
            rank_mask = b_mc_valid[:, 0].bool() & (b_cfm_value_invalid.sum(dim=1) == 0)
            rank_indices = torch.where(rank_mask)[0]
            required_chunks = cfg.rank_advantage_audit_chunks
            if rank_indices.numel() < required_chunks:
                raise RuntimeError(
                    "rank-advantage audit requires "
                    f"{required_chunks} fully valid labeled chunks, found {rank_indices.numel()}"
                )

            all_control_weights = b_advantages[rank_indices, 0].float()
            all_candidate_weights = centered_average_rank_scores(all_control_weights)
            all_reference_weights = b_mc_returns[rank_indices, 0].float()
            all_reference_weights = all_reference_weights - all_reference_weights.mean()
            if (
                not torch.isfinite(all_candidate_weights).all()
                or all_candidate_weights.std() <= 0
                or all_candidate_weights.min() >= 0
                or all_candidate_weights.max() <= 0
            ):
                raise RuntimeError("rank-advantage audit produced invalid candidate weights")
            order_spearman = spearman_rank_correlation(
                all_control_weights, all_candidate_weights
            )
            if not np.isclose(order_spearman, 1.0, atol=1e-12):
                raise RuntimeError("rank-advantage audit did not preserve weight ordering")

            gradient_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 7211
            )
            selected_positions = torch.randperm(
                rank_indices.numel(), generator=gradient_generator
            )[:required_chunks]
            gradient_indices = rank_indices[selected_positions]
            control_weights = all_control_weights[selected_positions]
            candidate_weights = all_candidate_weights[selected_positions]
            reference_weights = all_reference_weights[selected_positions]
            audit_batch_size = required_chunks // 2
            gradient_batch_results = []
            actor_module.eval()
            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * audit_batch_size,
                    (audit_batch + 1) * audit_batch_size,
                )
                batch_indices = gradient_indices[batch_slice]
                _, reference_gradient = replay_audit_gradient(
                    batch_indices,
                    reference_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, control_gradient = replay_audit_gradient(
                    batch_indices,
                    control_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                _, candidate_gradient = replay_audit_gradient(
                    batch_indices,
                    candidate_weights[batch_slice],
                    b_actions=b_actions,
                    b_cfm_losses=b_cfm_losses,
                    b_cfm_loss_ts=b_cfm_loss_ts,
                    b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                    b_cfm_value_invalid=b_cfm_value_invalid,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                reference_norm = reference_gradient.norm().item()
                if reference_norm <= 0 or not torch.isfinite(reference_gradient).all():
                    raise RuntimeError("rank-advantage audit produced an invalid reference gradient")
                control_cosine = torch.nn.functional.cosine_similarity(
                    control_gradient.unsqueeze(0), reference_gradient.unsqueeze(0), dim=1
                ).item()
                candidate_cosine = torch.nn.functional.cosine_similarity(
                    candidate_gradient.unsqueeze(0), reference_gradient.unsqueeze(0), dim=1
                ).item()
                gradient_batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": audit_batch_size,
                        "reference_gradient_norm": reference_norm,
                        "control_gradient_cosine_to_reference": control_cosine,
                        "candidate_gradient_cosine_to_reference": candidate_cosine,
                        "candidate_cosine_gain": candidate_cosine - control_cosine,
                    }
                )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()
            rank_advantage_result = {
                "iteration": iteration,
                "num_eligible_chunks": int(rank_indices.numel()),
                "num_gradient_chunks_audited": required_chunks,
                "control_candidate_spearman": order_spearman,
                "candidate_weight_mean": float(all_candidate_weights.mean().item()),
                "candidate_weight_std": float(
                    all_candidate_weights.std(unbiased=False).item()
                ),
                "batches": gradient_batch_results,
            }
            rank_advantage_output_path = (
                Path(cfg.rank_advantage_audit_output_json)
                if cfg.rank_advantage_audit_output_json is not None
                else run_dir / "rank_advantage_audit.json"
            )
            rank_advantage_output_path.parent.mkdir(parents=True, exist_ok=True)
            rank_advantage_output_path.write_text(
                json.dumps(rank_advantage_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Rank-advantage audit: %s",
                json.dumps(rank_advantage_result, sort_keys=True),
            )

        replay_audit_indices = None
        replay_audit_pre_gradient = None
        replay_audit_selection_cosine = None
        if cfg.success_replay_audit and iteration == cfg.success_replay_audit_iteration:
            success_mask = successful_chunk_mask(rewards_stored, dones_stored, n_action_steps)
            valid_chunk_mask = b_cfm_value_invalid.sum(dim=1) < n_action_steps
            success_indices = torch.where(success_mask & valid_chunk_mask)[0]
            if success_indices.numel() < cfg.success_replay_audit_min_chunks:
                raise RuntimeError(
                    "success replay audit requires at least "
                    f"{cfg.success_replay_audit_min_chunks} chunks, found {success_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(cfg.seed + iteration * 1009)
            replay_audit_indices = success_indices[
                torch.randperm(success_indices.numel(), generator=audit_generator)[
                    : cfg.success_replay_audit_chunks
                ]
            ]
            unit_weights = torch.ones(replay_audit_indices.numel())
            pre_ratios, replay_audit_pre_gradient = replay_audit_gradient(
                replay_audit_indices,
                unit_weights,
                b_actions=b_actions,
                b_cfm_losses=b_cfm_losses,
                b_cfm_loss_ts=b_cfm_loss_ts,
                b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                b_cfm_value_invalid=b_cfm_value_invalid,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
            )
            logger.info(
                "Success replay pre-update ratios: mean=%.6f std=%.6f min=%.6f max=%.6f",
                pre_ratios.mean().item(),
                pre_ratios.std(unbiased=False).item(),
                pre_ratios.min().item(),
                pre_ratios.max().item(),
            )

            positive_mask = (b_advantages[:, 0] > 0) & valid_chunk_mask
            positive_indices = torch.where(positive_mask)[0]
            if positive_indices.numel() < cfg.success_replay_audit_min_chunks:
                raise RuntimeError(
                    "success replay audit requires at least "
                    f"{cfg.success_replay_audit_min_chunks} positive chunks, "
                    f"found {positive_indices.numel()}"
                )
            positive_indices = positive_indices[
                torch.randperm(positive_indices.numel(), generator=audit_generator)[
                    : cfg.success_replay_audit_chunks
                ]
            ]
            positive_weights = b_advantages[positive_indices, 0].clamp_min(0)
            positive_weights = positive_weights / positive_weights.mean().clamp_min(1e-12)
            _, positive_gradient = replay_audit_gradient(
                positive_indices,
                positive_weights,
                b_actions=b_actions,
                b_cfm_losses=b_cfm_losses,
                b_cfm_loss_ts=b_cfm_loss_ts,
                b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                b_cfm_value_invalid=b_cfm_value_invalid,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
            )
            replay_audit_selection_cosine = torch.nn.functional.cosine_similarity(
                replay_audit_pre_gradient.unsqueeze(0), positive_gradient.unsqueeze(0), dim=1
            ).item()
            replay_audit_positive_gradient_norm = positive_gradient.norm().item()
            del positive_gradient

        ratio_audit_indices = None
        ratio_audit_stored_pre_gradient = None
        ratio_audit_heldout_pre_gradient = None
        ratio_audit_heldout_tensors = None
        if (
            cfg.cfm_ratio_generalization_audit
            and iteration == cfg.cfm_ratio_generalization_audit_iteration
        ):
            valid_chunk_mask = b_cfm_value_invalid.sum(dim=1) < n_action_steps
            positive_indices = torch.where((b_advantages[:, 0] > 0) & valid_chunk_mask)[0]
            if positive_indices.numel() < cfg.cfm_ratio_generalization_audit_min_chunks:
                raise RuntimeError(
                    "CFM ratio audit requires at least "
                    f"{cfg.cfm_ratio_generalization_audit_min_chunks} positive chunks, "
                    f"found {positive_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(cfg.seed + iteration * 2017)
            ratio_audit_indices = positive_indices[
                torch.randperm(positive_indices.numel(), generator=audit_generator)[
                    : cfg.cfm_ratio_generalization_audit_chunks
                ]
            ]
            unit_weights = torch.ones(ratio_audit_indices.numel())
            stored_pre_ratios, ratio_audit_stored_pre_gradient = replay_audit_gradient(
                ratio_audit_indices,
                unit_weights,
                b_actions=b_actions,
                b_cfm_losses=b_cfm_losses,
                b_cfm_loss_ts=b_cfm_loss_ts,
                b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                b_cfm_value_invalid=b_cfm_value_invalid,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
            )

            audit_count = ratio_audit_indices.numel()
            heldout_actions = b_actions[ratio_audit_indices]
            heldout_images = {
                key: value[ratio_audit_indices] for key, value in b_obs_images.items()
            }
            heldout_states = b_obs_state[ratio_audit_indices]
            heldout_invalid = b_cfm_value_invalid[ratio_audit_indices]
            heldout_times, heldout_noises = sample_cfm_variables(
                batch_size=audit_count,
                num_samples=n_action_samples,
                horizon=n_action_steps,
                action_dim=action_dim,
                mode="iid",
                time_generator=torch.Generator(device=device).manual_seed(cfg.seed + 424_242),
                noise_generator=torch.Generator(device=device).manual_seed(cfg.seed + 424_243),
                device=device,
                dtype=heldout_actions.dtype,
            )
            heldout_obs = {
                key: value[:, 0].to(device) for key, value in heldout_images.items()
            }
            heldout_obs["observation.state"] = heldout_states[:, 0].to(device)
            heldout_obs["action"] = heldout_actions.to(device)
            with torch.no_grad():
                heldout_losses, returned_times, returned_noises = get_cfm_values(
                    actor_module,
                    heldout_obs,
                    n_action_samples,
                    heldout_times,
                    heldout_noises,
                )
            heldout_losses = heldout_losses.permute(1, 0, 2).cpu()
            heldout_times_stored = returned_times.permute(1, 0, 2).cpu()
            heldout_noises_stored = returned_noises.permute(1, 0, 2, 3).cpu()
            heldout_local_indices = torch.arange(audit_count)
            heldout_pre_ratios, ratio_audit_heldout_pre_gradient = replay_audit_gradient(
                heldout_local_indices,
                unit_weights,
                b_actions=heldout_actions,
                b_cfm_losses=heldout_losses,
                b_cfm_loss_ts=heldout_times_stored,
                b_cfm_loss_epsilons=heldout_noises_stored,
                b_cfm_value_invalid=heldout_invalid,
                b_obs_images=heldout_images,
                b_obs_state=heldout_states,
            )
            ratio_audit_heldout_tensors = (
                heldout_local_indices,
                heldout_actions,
                heldout_losses,
                heldout_times_stored,
                heldout_noises_stored,
                heldout_invalid,
                heldout_images,
                heldout_states,
            )
            logger.info(
                "CFM ratio audit pre-update: stored_std=%.6g heldout_std=%.6g",
                stored_pre_ratios.std(unbiased=False).item(),
                heldout_pre_ratios.std(unbiased=False).item(),
            )

        early_stop_audit_tensors = None
        early_stop_pre_gradients = None
        early_stop_pre_surrogates = None
        early_stop_pre_results = None
        early_stop_epoch_results = []
        early_stop_positive_chunks_available = None
        if (
            cfg.heldout_ratio_early_stop_audit
            and iteration == cfg.heldout_ratio_early_stop_audit_iteration
        ):
            fully_valid_chunk_mask = b_cfm_value_invalid.sum(dim=1) == 0
            early_stop_positive_indices = torch.where(
                (b_advantages[:, 0] > 0) & fully_valid_chunk_mask
            )[0]
            early_stop_positive_chunks_available = int(
                early_stop_positive_indices.numel()
            )
            if (
                early_stop_positive_indices.numel()
                < cfg.heldout_ratio_early_stop_audit_chunks
            ):
                raise RuntimeError(
                    "held-out ratio early-stop audit requires "
                    f"{cfg.heldout_ratio_early_stop_audit_chunks} positive valid chunks, "
                    f"found {early_stop_positive_indices.numel()}"
                )
            early_stop_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 4013
            )
            early_stop_indices = early_stop_positive_indices[
                torch.randperm(
                    early_stop_positive_indices.numel(),
                    generator=early_stop_generator,
                )[: cfg.heldout_ratio_early_stop_audit_chunks]
            ]
            early_stop_weights = b_advantages[early_stop_indices, 0].float()
            early_stop_actions = b_actions[early_stop_indices]
            early_stop_images = {
                key: value[early_stop_indices] for key, value in b_obs_images.items()
            }
            early_stop_states = b_obs_state[early_stop_indices]
            early_stop_invalid = b_cfm_value_invalid[early_stop_indices]
            audit_count = early_stop_indices.numel()
            early_stop_times, early_stop_noises = sample_cfm_variables(
                batch_size=audit_count,
                num_samples=n_action_samples,
                horizon=n_action_steps,
                action_dim=action_dim,
                mode="iid",
                time_generator=torch.Generator(device=device).manual_seed(
                    cfg.seed + 525_242
                ),
                noise_generator=torch.Generator(device=device).manual_seed(
                    cfg.seed + 525_243
                ),
                device=device,
                dtype=early_stop_actions.dtype,
            )
            early_stop_obs = {
                key: value[:, 0].to(device) for key, value in early_stop_images.items()
            }
            early_stop_obs["observation.state"] = early_stop_states[:, 0].to(device)
            early_stop_obs["action"] = early_stop_actions.to(device)
            actor_module.eval()
            with torch.no_grad():
                early_stop_losses, returned_times, returned_noises = get_cfm_values(
                    actor_module,
                    early_stop_obs,
                    n_action_samples,
                    early_stop_times,
                    early_stop_noises,
                )
            early_stop_losses = early_stop_losses.permute(1, 0, 2).cpu()
            early_stop_times_stored = returned_times.permute(1, 0, 2).cpu()
            early_stop_noises_stored = returned_noises.permute(1, 0, 2, 3).cpu()
            early_stop_local_indices = torch.arange(audit_count)
            early_stop_audit_tensors = (
                early_stop_local_indices,
                early_stop_weights,
                early_stop_actions,
                early_stop_losses,
                early_stop_times_stored,
                early_stop_noises_stored,
                early_stop_invalid,
                early_stop_images,
                early_stop_states,
            )
            batch_size = audit_count // 2
            early_stop_pre_gradients = []
            early_stop_pre_surrogates = []
            early_stop_pre_results = []
            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * batch_size, (audit_batch + 1) * batch_size
                )
                batch_indices = early_stop_local_indices[batch_slice]
                batch_weights = early_stop_weights[batch_slice]
                pre_ratios, pre_gradient = replay_audit_gradient(
                    batch_indices,
                    batch_weights,
                    b_actions=early_stop_actions,
                    b_cfm_losses=early_stop_losses,
                    b_cfm_loss_ts=early_stop_times_stored,
                    b_cfm_loss_epsilons=early_stop_noises_stored,
                    b_cfm_value_invalid=early_stop_invalid,
                    b_obs_images=early_stop_images,
                    b_obs_state=early_stop_states,
                )
                early_stop_pre_gradients.append(pre_gradient)
                pre_surrogate = float(
                    (batch_weights[:, None] * pre_ratios).mean().item()
                )
                early_stop_pre_surrogates.append(pre_surrogate)
                early_stop_pre_results.append(
                    {
                        "batch": audit_batch,
                        "gradient_norm": float(pre_gradient.norm().item()),
                        "ratio_mean": float(pre_ratios.mean().item()),
                        "ratio_std": float(pre_ratios.std(unbiased=False).item()),
                        "ratio_min": float(pre_ratios.min().item()),
                        "ratio_max": float(pre_ratios.max().item()),
                        "surrogate": pre_surrogate,
                    }
                )
            logger.info(
                "Held-out ratio early-stop audit prepared: chunks=%d positive_available=%d",
                audit_count,
                early_stop_positive_chunks_available,
            )
            actor_module.train()

        if (
            cfg.advantage_stratified_mc_audit
            and iteration == cfg.advantage_stratified_mc_audit_iteration
        ):
            valid_chunk_indices = torch.where(
                b_cfm_value_invalid.sum(dim=1) == 0
            )[0]
            required_chunks = (
                cfg.advantage_stratified_mc_audit_batches
                * cfg.advantage_stratified_mc_audit_batch_size
            )
            if valid_chunk_indices.numel() < required_chunks:
                raise RuntimeError(
                    "advantage-stratified MC audit requires "
                    f"{required_chunks} valid chunks, found {valid_chunk_indices.numel()}"
                )
            audit_generator = torch.Generator().manual_seed(cfg.seed + iteration * 3011)
            selected_indices = valid_chunk_indices[
                torch.randperm(valid_chunk_indices.numel(), generator=audit_generator)[
                    :required_chunks
                ]
            ]
            actor_module.eval()
            stratified_batch_results = []
            for audit_batch in range(cfg.advantage_stratified_mc_audit_batches):
                batch_start = audit_batch * cfg.advantage_stratified_mc_audit_batch_size
                batch_end = batch_start + cfg.advantage_stratified_mc_audit_batch_size
                batch_indices = selected_indices[batch_start:batch_end]
                batch_weights = b_advantages[batch_indices, 0].float()
                candidate_counts = advantage_stratified_sample_counts(
                    batch_weights,
                    low_samples=cfg.advantage_stratified_mc_audit_low_samples,
                    high_samples=cfg.advantage_stratified_mc_audit_high_samples,
                )
                uniform_counts = torch.full_like(
                    candidate_counts,
                    cfg.advantage_stratified_mc_audit_uniform_samples,
                )
                reference_counts = torch.full_like(
                    candidate_counts,
                    cfg.advantage_stratified_mc_audit_reference_samples,
                )
                reference_gradient = stratified_mc_audit_gradient(
                    batch_indices,
                    batch_weights,
                    reference_counts,
                    seed_offset=500_000 + 10_000 * audit_batch,
                    b_actions=b_actions,
                    b_obs_images=b_obs_images,
                    b_obs_state=b_obs_state,
                )
                uniform_gradients = []
                candidate_gradients = []
                for audit_repeat in range(cfg.advantage_stratified_mc_audit_repeats):
                    uniform_gradients.append(
                        stratified_mc_audit_gradient(
                            batch_indices,
                            batch_weights,
                            uniform_counts,
                            seed_offset=600_000 + 10_000 * audit_batch + 100 * audit_repeat,
                            b_actions=b_actions,
                            b_obs_images=b_obs_images,
                            b_obs_state=b_obs_state,
                        )
                    )
                    candidate_gradients.append(
                        stratified_mc_audit_gradient(
                            batch_indices,
                            batch_weights,
                            candidate_counts,
                            seed_offset=700_000 + 10_000 * audit_batch + 100 * audit_repeat,
                            b_actions=b_actions,
                            b_obs_images=b_obs_images,
                            b_obs_state=b_obs_state,
                        )
                    )
                uniform_gradients_tensor = torch.stack(uniform_gradients)
                candidate_gradients_tensor = torch.stack(candidate_gradients)
                uniform_metrics = gradient_estimator_metrics(
                    uniform_gradients_tensor, reference_gradient
                )
                candidate_metrics = gradient_estimator_metrics(
                    candidate_gradients_tensor, reference_gradient
                )
                repeat_average_cosine = torch.nn.functional.cosine_similarity(
                    uniform_gradients_tensor.mean(dim=0).unsqueeze(0),
                    candidate_gradients_tensor.mean(dim=0).unsqueeze(0),
                    dim=1,
                ).item()
                stratified_batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": int(batch_indices.numel()),
                        "advantage_abs_median": float(batch_weights.abs().median().item()),
                        "reference_gradient_norm": float(reference_gradient.norm().item()),
                        "uniform": uniform_metrics,
                        "candidate": candidate_metrics,
                        "candidate_mean_mse_relative_change": (
                            candidate_metrics["mean_normalized_gradient_mse"]
                            / max(uniform_metrics["mean_normalized_gradient_mse"], 1e-12)
                            - 1.0
                        ),
                        "repeat_average_gradient_cosine": repeat_average_cosine,
                    }
                )
                del reference_gradient, uniform_gradients_tensor, candidate_gradients_tensor

            stratified_result = {
                "iteration": iteration,
                "num_valid_chunks_available": int(valid_chunk_indices.numel()),
                "batch_size": cfg.advantage_stratified_mc_audit_batch_size,
                "num_batches": cfg.advantage_stratified_mc_audit_batches,
                "num_repeats": cfg.advantage_stratified_mc_audit_repeats,
                "reference_samples": cfg.advantage_stratified_mc_audit_reference_samples,
                "uniform_samples": cfg.advantage_stratified_mc_audit_uniform_samples,
                "low_samples": cfg.advantage_stratified_mc_audit_low_samples,
                "high_samples": cfg.advantage_stratified_mc_audit_high_samples,
                "batches": stratified_batch_results,
            }
            stratified_output_path = (
                Path(cfg.advantage_stratified_mc_audit_output_json)
                if cfg.advantage_stratified_mc_audit_output_json is not None
                else run_dir / "advantage_stratified_mc_audit.json"
            )
            stratified_output_path.parent.mkdir(parents=True, exist_ok=True)
            stratified_output_path.write_text(
                json.dumps(stratified_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Advantage-stratified MC audit: %s",
                json.dumps(stratified_result, sort_keys=True),
            )
            actor_module.train()

        if (
            cfg.bc_anchor_pcgrad_audit
            and iteration == cfg.bc_anchor_pcgrad_audit_iteration
        ):
            valid_positive = (b_cfm_value_invalid.sum(dim=1) == 0) & (
                b_advantages[:, 0] > 0
            )
            eligible_indices = torch.where(valid_positive)[0]
            audit_count = cfg.bc_anchor_pcgrad_audit_chunks
            if eligible_indices.numel() < audit_count:
                raise RuntimeError(
                    "BC-anchor PCGrad audit requires "
                    f"{audit_count} positive fully valid chunks, found "
                    f"{eligible_indices.numel()}"
                )
            selection_generator = torch.Generator().manual_seed(
                cfg.seed + iteration * 4001
            )
            selected_indices = eligible_indices[
                torch.randperm(
                    eligible_indices.numel(), generator=selection_generator
                )[:audit_count]
            ]
            audit_actions = b_actions[selected_indices]
            audit_images = {
                key: value[selected_indices] for key, value in b_obs_images.items()
            }
            audit_states = b_obs_state[selected_indices]
            audit_invalid = b_cfm_value_invalid[selected_indices]
            audit_weights = b_advantages[selected_indices, 0].float()
            audit_times, audit_noises = sample_cfm_variables(
                batch_size=audit_count,
                num_samples=n_action_samples,
                horizon=n_action_steps,
                action_dim=action_dim,
                mode="iid",
                time_generator=torch.Generator(device=device).manual_seed(
                    cfg.seed + 810_001
                ),
                noise_generator=torch.Generator(device=device).manual_seed(
                    cfg.seed + 810_002
                ),
                device=device,
                dtype=audit_actions.dtype,
            )
            behavior_obs = {
                key: value[:, 0].to(device) for key, value in audit_images.items()
            }
            behavior_obs["observation.state"] = audit_states[:, 0].to(device)
            behavior_obs["action"] = audit_actions.to(device)
            actor_module.eval()
            with torch.no_grad():
                audit_old_losses, returned_times, returned_noises = get_cfm_values(
                    actor_module,
                    behavior_obs,
                    n_action_samples,
                    audit_times,
                    audit_noises,
                )
            audit_old_losses = audit_old_losses.permute(1, 0, 2).cpu()
            audit_times_stored = returned_times.permute(1, 0, 2).cpu()
            audit_noises_stored = returned_noises.permute(1, 0, 2, 3).cpu()
            local_indices = torch.arange(audit_count)
            pcgrad_batch_size = audit_count // 2
            batch_results = []

            for audit_batch in range(2):
                batch_slice = slice(
                    audit_batch * pcgrad_batch_size,
                    (audit_batch + 1) * pcgrad_batch_size,
                )
                batch_indices = local_indices[batch_slice]
                batch_weights = audit_weights[batch_slice]
                batch_observations = {
                    key: value[batch_slice, 0].to(device)
                    for key, value in audit_images.items()
                }
                batch_observations["observation.state"] = audit_states[
                    batch_slice, 0
                ].to(device)
                normalized_observations = actor_module.normalize_inputs(
                    copy.deepcopy(batch_observations)
                )
                with torch.no_grad():
                    observation_conditioning = actor_module.model.encode_observations(
                        normalized_observations
                    ).detach()
                query_points = None
                query_times = None
                if cfg.bc_anchor_pcgrad_anchor_mode == "velocity":
                    query_generator = torch.Generator(device=device).manual_seed(
                        cfg.seed + 820_000 + audit_batch
                    )
                    query_points = torch.randn(
                        (pcgrad_batch_size, n_action_steps, action_dim),
                        generator=query_generator,
                        device=device,
                        dtype=audit_actions.dtype,
                    )
                    query_times = torch.rand(
                        (pcgrad_batch_size, 1, 1),
                        generator=query_generator,
                        device=device,
                        dtype=audit_actions.dtype,
                    )
                    with torch.no_grad():
                        anchor_behavior = actor_velocity_field(
                            observation_conditioning, query_points, query_times
                        ).detach()
                else:
                    with torch.no_grad():
                        anchor_behavior = actor_zero_source_endpoint(
                            observation_conditioning
                        ).detach()

                def fixed_rl_objective() -> torch.Tensor:
                    _, objective = replay_audit_objective(
                        batch_indices,
                        batch_weights,
                        b_actions=audit_actions,
                        b_cfm_losses=audit_old_losses,
                        b_cfm_loss_ts=audit_times_stored,
                        b_cfm_loss_epsilons=audit_noises_stored,
                        b_cfm_value_invalid=audit_invalid,
                        b_obs_images=audit_images,
                        b_obs_state=audit_states,
                    )
                    return objective

                def anchor_objective() -> torch.Tensor:
                    if cfg.bc_anchor_pcgrad_anchor_mode == "velocity":
                        behavior = actor_velocity_field(
                            observation_conditioning, query_points, query_times
                        )
                    else:
                        behavior = actor_zero_source_endpoint(
                            observation_conditioning
                        )
                    return (behavior - anchor_behavior).square().mean()

                parameters = [
                    parameter
                    for parameter in actor_module.parameters()
                    if parameter.requires_grad
                ]
                base_values = [parameter.detach().clone() for parameter in parameters]

                first_loss = fixed_rl_objective()
                _, first_gradient = autograd_list(first_loss)
                first_step_scale = apply_virtual_actor_step(parameters, first_gradient)
                common_values = [parameter.detach().clone() for parameter in parameters]

                pre_rl_loss = fixed_rl_objective()
                _, pre_rl_gradient = autograd_list(pre_rl_loss)
                pre_bc_loss = anchor_objective()
                _, pre_bc_gradient = autograd_list(pre_bc_loss)
                candidate_gradient, gradient_dot_value, conflict = (
                    project_conflicting_gradient(pre_rl_gradient, pre_bc_gradient)
                )
                rl_bc_cosine = float(
                    list_gradient_cosine(pre_rl_gradient, pre_bc_gradient).item()
                )
                rl_norm = float(gradient_norm(pre_rl_gradient).item())
                bc_norm = float(gradient_norm(pre_bc_gradient).item())
                candidate_norm = float(gradient_norm(candidate_gradient).item())

                control_step_scale = apply_virtual_actor_step(
                    parameters, pre_rl_gradient
                )
                control_rl_loss = fixed_rl_objective()
                control_bc_loss = anchor_objective()
                _, control_post_gradient = autograd_list(fixed_rl_objective())
                control_gradient_cosine = float(
                    list_gradient_cosine(control_post_gradient, pre_rl_gradient).item()
                )

                restore_parameters(parameters, common_values)
                candidate_step_scale = apply_virtual_actor_step(
                    parameters, candidate_gradient
                )
                candidate_rl_loss = fixed_rl_objective()
                candidate_bc_loss = anchor_objective()
                _, candidate_post_gradient = autograd_list(fixed_rl_objective())
                candidate_gradient_cosine = float(
                    list_gradient_cosine(candidate_post_gradient, pre_rl_gradient).item()
                )
                restore_parameters(parameters, base_values)

                batch_results.append(
                    {
                        "batch": audit_batch,
                        "num_chunks": pcgrad_batch_size,
                        "first_step_scale": first_step_scale,
                        "control_step_scale": control_step_scale,
                        "candidate_step_scale": candidate_step_scale,
                        "pre_rl_gradient_norm": rl_norm,
                        "pre_bc_gradient_norm": bc_norm,
                        "candidate_gradient_norm": candidate_norm,
                        "candidate_gradient_norm_retention": candidate_norm
                        / max(rl_norm, 1e-12),
                        "rl_bc_gradient_dot": float(gradient_dot_value.item()),
                        "rl_bc_gradient_cosine": rl_bc_cosine,
                        "projection_active": conflict,
                        "pre_rl_loss": float(pre_rl_loss.detach().item()),
                        "control_rl_loss": float(control_rl_loss.detach().item()),
                        "candidate_rl_loss": float(candidate_rl_loss.detach().item()),
                        "control_rl_surrogate_gain": float(
                            (pre_rl_loss - control_rl_loss).detach().item()
                        ),
                        "candidate_rl_surrogate_gain": float(
                            (pre_rl_loss - candidate_rl_loss).detach().item()
                        ),
                        "pre_bc_anchor_mse": float(pre_bc_loss.detach().item()),
                        "control_bc_anchor_mse": float(
                            control_bc_loss.detach().item()
                        ),
                        "candidate_bc_anchor_mse": float(
                            candidate_bc_loss.detach().item()
                        ),
                        "control_bc_mse_increase": float(
                            (control_bc_loss - pre_bc_loss).detach().item()
                        ),
                        "candidate_bc_mse_increase": float(
                            (candidate_bc_loss - pre_bc_loss).detach().item()
                        ),
                        "control_post_rl_gradient_cosine": control_gradient_cosine,
                        "candidate_post_rl_gradient_cosine": candidate_gradient_cosine,
                    }
                )

            audit_result = {
                "iteration": iteration,
                "num_eligible_chunks": int(eligible_indices.numel()),
                "num_chunks_audited": audit_count,
                "num_batches": 2,
                "cfm_samples": n_action_samples,
                "anchor_mode": cfg.bc_anchor_pcgrad_anchor_mode,
                "virtual_learning_rate": cfg.learning_rate_actor,
                "max_grad_norm": cfg.max_grad_norm,
                "batches": batch_results,
            }
            audit_output_path = (
                Path(cfg.bc_anchor_pcgrad_audit_output_json)
                if cfg.bc_anchor_pcgrad_audit_output_json is not None
                else run_dir / "bc_anchor_pcgrad_audit.json"
            )
            audit_output_path.parent.mkdir(parents=True, exist_ok=True)
            audit_output_path.write_text(
                json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "BC-anchor PCGrad audit: %s",
                json.dumps(audit_result, sort_keys=True),
            )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()

        endpoint_anchor_conditioning = None
        endpoint_anchor_targets = None
        if (
            cfg.zero_endpoint_pcgrad_train
            and iteration > cfg.n_iterations_train_only_value
        ):
            anchor_observations = {
                key: value[:, 0].to(device) for key, value in b_obs_images.items()
            }
            anchor_observations["observation.state"] = b_obs_state[:, 0].to(device)
            normalized_anchor_observations = actor_module.normalize_inputs(
                copy.deepcopy(anchor_observations)
            )
            with torch.no_grad():
                endpoint_anchor_conditioning = endpoint_anchor_model.encode_observations(
                    normalized_anchor_observations
                ).detach()
                endpoint_anchor_targets = actor_zero_source_endpoint(
                    endpoint_anchor_conditioning,
                    policy_model=endpoint_anchor_model,
                ).detach()

        behavior_x1_predictions = None
        adaptive_lr_events = []
        actor_lr_at_policy_update_start = float(
            optimizer_actor.param_groups[0]["lr"]
        )
        if cfg.adaptive_actor_lr and iteration > cfg.n_iterations_train_only_value:
            if cfg.loss_mode != "fpo":
                raise ValueError("adaptive actor LR requires loss_mode='fpo'")
            behavior_x1_predictions = torch.empty(
                local_batch_size,
                n_action_steps,
                n_action_samples,
                action_dim,
                dtype=b_actions.dtype,
            )
            actor_module.eval()
            with torch.no_grad():
                for prediction_start in range(0, local_batch_size, minibatch_size):
                    prediction_end = min(
                        prediction_start + minibatch_size, local_batch_size
                    )
                    prediction_slice = slice(prediction_start, prediction_end)
                    prediction_obs = {
                        key: value[prediction_slice, 0].to(device)
                        for key, value in b_obs_images.items()
                    }
                    prediction_obs["observation.state"] = b_obs_state[
                        prediction_slice, 0
                    ].to(device)
                    prediction_obs["action"] = b_actions[prediction_slice].to(device)
                    prediction_times = b_cfm_loss_ts[prediction_slice].to(device)
                    prediction_times = prediction_times.permute(0, 2, 1).reshape(
                        -1, n_action_steps, 1
                    )
                    if not (
                        prediction_times[:, 0, 0]
                        == prediction_times[:, -1, 0]
                    ).all():
                        raise RuntimeError(
                            "adaptive LR received inconsistent CFM times within a chunk"
                        )
                    prediction_times = prediction_times[:, 0:1, :]
                    prediction_noises = b_cfm_loss_epsilons[
                        prediction_slice
                    ].to(device)
                    prediction_noises = prediction_noises.permute(
                        0, 2, 1, 3
                    ).reshape(-1, n_action_steps, action_dim)
                    _, _, _, behavior_x1 = get_cfm_values(
                        actor,
                        prediction_obs,
                        n_action_samples,
                        prediction_times,
                        prediction_noises,
                        return_predictions=True,
                    )
                    behavior_x1_predictions[prediction_slice] = (
                        behavior_x1.permute(1, 0, 2, 3).cpu()
                    )
            actor_module.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()

        # ---------- Policy update ----------
        b_inds = np.arange(local_batch_size)
        clipfracs = []
        endpoint_pcgrad_step_metrics = []
        rollback_active_elements = 0
        rollback_total_elements = 0
        actor.train(); critic.train()
        if cfg.freeze_vision_encoder:
            actor_module.model.vision_encoder.eval()

        # Initialize gradient norm tracking
        actor_grad_norm_before = torch.tensor(0.0)
        actor_grad_norm_after = torch.tensor(0.0)
        critic_grad_norm_before = torch.tensor(0.0)
        critic_grad_norm_after = torch.tensor(0.0)

        for epoch in trange(cfg.update_epochs, desc=f"[Rank {rank}] Policy update", disable=(rank != 0)):
            early_stop = False
            adaptive_epoch_event_start = len(adaptive_lr_events)
            np.random.shuffle(b_inds)
            accumulation_counter = 0
            optimizer_actor.zero_grad(set_to_none=True)
            optimizer_critic.zero_grad(set_to_none=True)

            for start in range(0, local_batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_actions = b_actions[mb_inds].to(device)
                mb_cfm_losses = b_cfm_losses[mb_inds].to(device)
                mb_cfm_loss_ts = b_cfm_loss_ts[mb_inds].to(device)
                mb_cfm_loss_epsilons = b_cfm_loss_epsilons[mb_inds].to(device)
                mb_dppo_log_probs = b_dppo_log_probs[mb_inds].to(device)
                mb_mdp_x_t_paths = b_mdp_x_t_paths[mb_inds].to(device)
                mb_cfm_value_invalid = b_cfm_value_invalid[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)
                mb_mc_returns = b_mc_returns[mb_inds].to(device)
                mb_mc_valid = b_mc_valid[mb_inds].to(device)
                mb_obs_images = {k: b_obs_images[k][mb_inds].to(device) for k in b_obs_images.keys()}
                mb_obs_state = b_obs_state[mb_inds].to(device)

                valid_idx_mask_in_chunk = 1.0 - mb_cfm_value_invalid

                # Critic forward
                obs_chunk = {k: mb_obs_images[k].reshape(-1, 3, img_h, img_w) for k in mb_obs_images.keys()}
                obs_chunk["observation.state"] = mb_obs_state.reshape(-1, joint_pos_dim)
                obs_chunk = actor_module.normalize_inputs(obs_chunk)
                obs_chunk_cond = actor_module.model.encode_observations(obs_chunk)
                newvalue = critic(obs_chunk_cond)
                newvalue = newvalue.reshape(mb_returns.shape[0], -1)

                if success_critic is not None:
                    optimizer_success_critic.zero_grad(set_to_none=True)
                    candidate_output = success_critic(obs_chunk_cond.detach()).reshape(
                        mb_returns.shape[0], -1
                    )
                    candidate_valid = valid_idx_mask_in_chunk.bool()
                    if cfg.discounted_success_critic_audit:
                        candidate_valid = candidate_valid & mb_mc_valid
                    if candidate_valid.any():
                        if cfg.discounted_success_critic_audit:
                            candidate_value_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                candidate_output[candidate_valid],
                                mb_mc_returns[candidate_valid],
                            )
                        else:
                            candidate_value_loss = 0.5 * torch.mean(
                                (
                                    candidate_output[candidate_valid]
                                    - mb_returns[candidate_valid]
                                ).square()
                            )
                        candidate_value_loss.backward()
                        nn.utils.clip_grad_norm_(success_critic.parameters(), cfg.max_grad_norm)
                        optimizer_success_critic.step()

                if cfg.loss_mode == "fpo":
                    # CFM losses
                    obs_chunk2 = {k: mb_obs_images[k][:, 0] for k in mb_obs_images.keys()}
                    obs_chunk2["observation.state"] = mb_obs_state[:, 0]
                    obs_chunk2["action"] = mb_actions

                    old_cfm_loss_ts = mb_cfm_loss_ts.permute(0, 2, 1).reshape(-1, n_action_steps, 1)
                    assert (old_cfm_loss_ts[:, 0, 0] == old_cfm_loss_ts[:, -1, 0]).all()
                    old_cfm_loss_ts = old_cfm_loss_ts[:, 0:1, :]
                    old_cfm_loss_epsilons = mb_cfm_loss_epsilons.permute(0, 2, 1, 3).reshape(-1, n_action_steps, action_dim)

                    if behavior_x1_predictions is None:
                        curr_cfm_loss, _, _ = get_cfm_values(
                            actor, obs_chunk2, n_action_samples, old_cfm_loss_ts, old_cfm_loss_epsilons
                        )
                    else:
                        curr_cfm_loss, _, _, curr_x1_predictions = get_cfm_values(
                            actor,
                            obs_chunk2,
                            n_action_samples,
                            old_cfm_loss_ts,
                            old_cfm_loss_epsilons,
                            return_predictions=True,
                        )
                        current_x1 = curr_x1_predictions.permute(1, 0, 2, 3)
                        behavior_x1 = behavior_x1_predictions[mb_inds].to(device)
                        kl_mask = valid_idx_mask_in_chunk[:, :, None, None]
                        kl_sum = ((current_x1.detach() - behavior_x1).square() * kl_mask).sum()
                        kl_count = kl_mask.sum() * n_action_samples * action_dim
                        if is_ddp:
                            dist.all_reduce(kl_sum, op=dist.ReduceOp.SUM)
                            dist.all_reduce(kl_count, op=dist.ReduceOp.SUM)
                        kl_proxy = kl_sum / kl_count.clamp_min(1.0)
                        previous_lr = optimizer_actor.param_groups[0]["lr"]
                        lr_decision = adapt_learning_rate_from_kl(
                            previous_lr,
                            float(kl_proxy.item()),
                            desired_kl=cfg.adaptive_actor_lr_target_kl,
                            minimum=cfg.learning_rate_actor * 0.1,
                            maximum=cfg.learning_rate_actor * 100.0,
                        )
                        for parameter_group in optimizer_actor.param_groups:
                            parameter_group["lr"] = lr_decision.learning_rate
                        adaptive_lr_events.append(
                            {
                                "epoch": epoch + 1,
                                "minibatch": len(adaptive_lr_events) + 1,
                                "kl_proxy": float(kl_proxy.item()),
                                "previous_lr": float(previous_lr),
                                "learning_rate": float(lr_decision.learning_rate),
                                "action": lr_decision.action,
                            }
                        )
                    curr_cfm_loss = curr_cfm_loss.permute(1, 0, 2)

                    old_cfm_loss = mb_cfm_losses.reshape(mb_cfm_losses.shape[0], -1, n_groups, group_size)
                    curr_cfm_loss = curr_cfm_loss.reshape(curr_cfm_loss.shape[0], -1, n_groups, group_size)

                    if cfg.clamp_old_cfm_loss is not None:
                        # old_cfm_loss = torch.clamp(old_cfm_loss, max=cfg.clamp_old_cfm_loss)
                        old_cfm_loss = clamp_ste(old_cfm_loss, max=cfg.clamp_old_cfm_loss)

                    if cfg.do_chunk_level_ppo:
                        if cfg.do_average_cfm_loss_in_chunk:
                            denom = valid_idx_mask_in_chunk.sum(dim=1).unsqueeze(-1).unsqueeze(-1).clamp_min(1.0)
                            old_cfm_loss = (old_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / denom
                            curr_cfm_loss = (curr_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / denom
                        else:
                            old_cfm_loss = (old_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                            curr_cfm_loss = (curr_cfm_loss * valid_idx_mask_in_chunk.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

                        old_cfm_loss = old_cfm_loss.mean(dim=-1)
                        curr_cfm_loss = curr_cfm_loss.mean(dim=-1)
                        logratio = old_cfm_loss - curr_cfm_loss
                        if cfg.clamp_logratio is not None:
                            # logratio = torch.clamp(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)
                            logratio = clamp_ste(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)

                        ratio = logratio.exp()
                        mb_advantages = mb_advantages[:, 0:1]
                    else:
                        old_cfm_loss = old_cfm_loss.mean(dim=-1) # (B, T, n_groups)
                        curr_cfm_loss = curr_cfm_loss.mean(dim=-1) # (B, T, n_groups)
                        logratio = old_cfm_loss - curr_cfm_loss
                        if cfg.clamp_logratio is not None:
                            # logratio = torch.clamp(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)
                            logratio = clamp_ste(logratio, min=-cfg.clamp_logratio, max=cfg.clamp_logratio)

                        ratio = logratio.exp()
                        mb_advantages = mb_advantages.unsqueeze(-1) # (B, T, 1)

                    # TODO
                    entropy = 0.0

                elif cfg.loss_mode == "dppo":
                    obs_chunk2 = {k: mb_obs_images[k][:, 0] for k in mb_obs_images.keys()}
                    obs_chunk2["observation.state"] = mb_obs_state[:, 0]
                    obs_chunk2["action"] = mb_actions

                    assert mb_mdp_x_t_paths.shape == (len(mb_inds), n_action_steps, actor_module.config.sampling_steps, action_dim)
                    obs_chunk2["mdp_x_t_path"] = mb_mdp_x_t_paths

                    log_prob, entropy, sde_sigma = get_log_prob_and_entropy(actor, obs_chunk2)
                    assert log_prob.shape == (len(mb_inds), n_action_steps, actor_module.config.sampling_steps)

                    # Get valid log probs
                    log_prob_valid = log_prob * valid_idx_mask_in_chunk.unsqueeze(-1)
                    mb_dppo_log_probs_valid = mb_dppo_log_probs * valid_idx_mask_in_chunk.unsqueeze(-1) 

                    # action chunk log prob
                    log_prob_chunk = log_prob_valid.sum(dim=1)
                    mb_dppo_log_probs_chunk = mb_dppo_log_probs_valid.sum(dim=1)                    

                    if cfg.average_logprob_over_denoising_steps:
                        # Average log probabilities over the flow steps
                        log_prob_chunk = log_prob_chunk.sum(dim=-1)
                        mb_dppo_log_probs_chunk = mb_dppo_log_probs_chunk.sum(dim=-1)

                        log_ratio = log_prob_chunk - mb_dppo_log_probs_chunk 
                        log_ratio = log_ratio.unsqueeze(-1)
                        assert log_ratio.shape == (len(mb_inds), 1), "log_ratio shape should be (len(mb_inds), 1)"

                        # log_ratio = log_ratio / (cfg.dppo_norm_factor * actor_module.config.sampling_steps * actor_module.config.horizon)
                        log_ratio = log_ratio / (cfg.dppo_norm_factor * actor_module.config.horizon)
                        ratio = log_ratio.exp()

                    else:
                        log_ratio = log_prob_chunk - mb_dppo_log_probs_chunk 

                        log_ratio = log_ratio / (cfg.dppo_norm_factor * actor_module.config.horizon)
                        ratio = log_ratio.exp()

                    # Do chunk level PPO
                    mb_advantages = mb_advantages[:, 0:1]

                else:
                    raise ValueError(f"Invalid loss mode: {cfg.loss_mode}")

                if cfg.norm_adv and cfg.advantage_normalization_scope == "minibatch":
                    if is_ddp:
                        # Compute local statistics
                        adv_mean = mb_advantages.mean()
                        adv_std = mb_advantages.std()
                        # if adv_std is nan, set it to 0
                        if torch.isnan(adv_std):
                            adv_std = torch.tensor(0.0, device=device)
                        # Sync across ranks
                        stats = torch.tensor([adv_mean, adv_std, mb_advantages.numel()], device=device)
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                        global_mean = stats[0] / world_size
                        global_std = stats[1] / world_size
                        mb_advantages = (mb_advantages - global_mean) / (global_std + 1e-8)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                advantage_weight_temperature = torch.tensor(float("nan"), device=device)
                advantage_weight_ess_fraction = torch.tensor(1.0, device=device)
                if cfg.advantage_weighting == "ess_softmax":
                    mirror_weights, advantage_weight_temperature, advantage_weight_ess_fraction = (
                        ess_softmax_weights(mb_advantages, cfg.advantage_weight_ess_fraction)
                    )
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]
                    pg_loss = clipped_mirror_ratio_loss(
                        ratio, mirror_weights, cfg.clip_coef
                    )
                elif cfg.trust_region_mode == "spo":
                    clipfracs += [0.0]
                    spo_obj = mb_advantages * ratio - mb_advantages.abs() / (2.0 * cfg.spo_clip_coef) * (ratio - 1.0) ** 2
                    pg_loss = (-spo_obj).mean()
                elif cfg.trust_region_mode == "aspo":
                    clipfracs += [0.0]
                    spo_obj = mb_advantages * ratio - mb_advantages.abs() / (2.0 * cfg.spo_clip_coef) * (ratio - 1.0) ** 2
                    spo_obj = -spo_obj
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    ppo_obj = torch.max(pg_loss1, pg_loss2).mean()
                    positive_mask = mb_advantages > 0
                    if cfg.do_chunk_level_ppo:
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef)[positive_mask[:, 0], :].float().mean().item()]
                    else:
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef)[positive_mask].float().mean().item()]
                    pg_loss = torch.where(mb_advantages > 0, ppo_obj, spo_obj).mean()
                elif cfg.trust_region_mode == "rollback":
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]
                    pg_loss, rollback_active = rollback_clipped_ratio_loss(
                        ratio,
                        mb_advantages,
                        cfg.clip_coef,
                        cfg.rollback_alpha,
                    )
                    if iteration > cfg.n_iterations_train_only_value:
                        rollback_active_elements += int(rollback_active.sum().item())
                        rollback_total_elements += rollback_active.numel()
                else:
                    clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -cfg.clip_coef, cfg.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
                v_loss = v_loss[valid_idx_mask_in_chunk == 1].mean()

                # Total loss
                entropy_loss = -entropy.mean() if cfg.learn_sde_sigma and isinstance(entropy, torch.Tensor) else 0.0
                policy_loss = pg_loss + cfg.entropy_loss_coef * entropy_loss if iteration > cfg.n_iterations_train_only_value else 0.0
                loss = (policy_loss + v_loss * cfg.vf_coef) / cfg.gradient_accumulation_steps
                loss.backward()
                if (
                    cfg.zero_endpoint_pcgrad_train
                    and iteration > cfg.n_iterations_train_only_value
                ):
                    endpoint_indices = torch.as_tensor(
                        mb_inds, device=device, dtype=torch.long
                    )
                    current_endpoints = actor_zero_source_endpoint(
                        endpoint_anchor_conditioning[endpoint_indices]
                    )
                    endpoint_anchor_loss = (
                        current_endpoints - endpoint_anchor_targets[endpoint_indices]
                    ).square().mean()
                    actor_parameters = [
                        parameter
                        for parameter in actor_module.parameters()
                        if parameter.requires_grad
                    ]
                    anchor_gradients = list(
                        torch.autograd.grad(
                            endpoint_anchor_loss,
                            actor_parameters,
                            allow_unused=True,
                        )
                    )
                    rl_gradients = [
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in actor_parameters
                    ]
                    projected_gradients, gradient_dot_value, projection_active = (
                        project_conflicting_gradient(rl_gradients, anchor_gradients)
                    )
                    rl_gradient_norm = float(gradient_norm(rl_gradients).item())
                    anchor_gradient_norm = float(gradient_norm(anchor_gradients).item())
                    projected_gradient_norm = float(
                        gradient_norm(projected_gradients).item()
                    )
                    gradient_cosine_value = (
                        float(
                            list_gradient_cosine(
                                rl_gradients, anchor_gradients
                            ).item()
                        )
                        if min(rl_gradient_norm, anchor_gradient_norm) > 1e-12
                        else 0.0
                    )
                    for parameter, projected_gradient in zip(
                        actor_parameters, projected_gradients, strict=True
                    ):
                        if projected_gradient is not None:
                            parameter.grad.copy_(projected_gradient)
                    endpoint_pcgrad_step_metrics.append(
                        {
                            "anchor_mse": float(endpoint_anchor_loss.detach().item()),
                            "projection_active": projection_active,
                            "gradient_dot": float(gradient_dot_value.item()),
                            "gradient_cosine": gradient_cosine_value,
                            "rl_gradient_norm": rl_gradient_norm,
                            "anchor_gradient_norm": anchor_gradient_norm,
                            "projected_gradient_norm": projected_gradient_norm,
                            "gradient_norm_retention": projected_gradient_norm
                            / max(rl_gradient_norm, 1e-12),
                        }
                    )
                accumulation_counter += 1

                if accumulation_counter % cfg.gradient_accumulation_steps == 0:
                    # Log gradient norms before clipping
                    actor_grad_norm_before = nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    critic_grad_norm_before = nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                    # Compute gradient norms after clipping
                    actor_grad_norm_after = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
                    critic_grad_norm_after = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
                    optimizer_actor.step()
                    optimizer_critic.step()
                    if hasattr(actor_module, "step_ema"):
                        actor_module.step_ema()
                    optimizer_actor.zero_grad(set_to_none=True)
                    optimizer_critic.zero_grad(set_to_none=True)

            if accumulation_counter % cfg.gradient_accumulation_steps != 0:
                # Log gradient norms before clipping
                actor_grad_norm_before = nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                critic_grad_norm_before = nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                # Compute gradient norms after clipping
                actor_grad_norm_after = torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'))
                critic_grad_norm_after = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
                optimizer_actor.step()
                optimizer_critic.step()
                if hasattr(actor_module, "step_ema"):
                    actor_module.step_ema()
                optimizer_actor.zero_grad(set_to_none=True)
                optimizer_critic.zero_grad(set_to_none=True)

            if early_stop_audit_tensors is not None:
                (
                    early_stop_local_indices,
                    early_stop_weights,
                    early_stop_actions,
                    early_stop_losses,
                    early_stop_times_stored,
                    early_stop_noises_stored,
                    early_stop_invalid,
                    early_stop_images,
                    early_stop_states,
                ) = early_stop_audit_tensors
                actor_module.eval()
                epoch_batch_results = []
                epoch_ratios = []
                audit_batch_size = early_stop_local_indices.numel() // 2
                for audit_batch in range(2):
                    batch_slice = slice(
                        audit_batch * audit_batch_size,
                        (audit_batch + 1) * audit_batch_size,
                    )
                    batch_indices = early_stop_local_indices[batch_slice]
                    batch_weights = early_stop_weights[batch_slice]
                    post_ratios, post_gradient = replay_audit_gradient(
                        batch_indices,
                        batch_weights,
                        b_actions=early_stop_actions,
                        b_cfm_losses=early_stop_losses,
                        b_cfm_loss_ts=early_stop_times_stored,
                        b_cfm_loss_epsilons=early_stop_noises_stored,
                        b_cfm_value_invalid=early_stop_invalid,
                        b_obs_images=early_stop_images,
                        b_obs_state=early_stop_states,
                    )
                    surrogate = float(
                        (batch_weights[:, None] * post_ratios).mean().item()
                    )
                    gradient_cosine = torch.nn.functional.cosine_similarity(
                        early_stop_pre_gradients[audit_batch].unsqueeze(0),
                        post_gradient.unsqueeze(0),
                        dim=1,
                    ).item()
                    epoch_batch_results.append(
                        {
                            "batch": audit_batch,
                            "active_positive_ratio_fraction": float(
                                (post_ratios <= 1 + cfg.clip_coef).float().mean().item()
                            ),
                            "gradient_cosine_to_preupdate": gradient_cosine,
                            "gradient_norm": float(post_gradient.norm().item()),
                            "ratio_mean": float(post_ratios.mean().item()),
                            "surrogate": surrogate,
                            "surrogate_gain": (
                                surrogate
                                - early_stop_pre_surrogates[audit_batch]
                            ),
                        }
                    )
                    epoch_ratios.append(post_ratios)
                pooled_ratios = torch.cat(epoch_ratios)
                epoch_result = {
                        "epoch": epoch + 1,
                        "pooled_active_positive_ratio_fraction": float(
                            (pooled_ratios <= 1 + cfg.clip_coef).float().mean().item()
                        ),
                        "batches": epoch_batch_results,
                    }
                epoch_adaptive_events = adaptive_lr_events[adaptive_epoch_event_start:]
                if epoch_adaptive_events:
                    epoch_result["adaptive_actor_lr"] = {
                        "kl_mean": float(
                            np.mean([event["kl_proxy"] for event in epoch_adaptive_events])
                        ),
                        "kl_max": float(
                            max(event["kl_proxy"] for event in epoch_adaptive_events)
                        ),
                        "lr_min": float(
                            min(event["learning_rate"] for event in epoch_adaptive_events)
                        ),
                        "lr_max": float(
                            max(event["learning_rate"] for event in epoch_adaptive_events)
                        ),
                        "lr_final": float(epoch_adaptive_events[-1]["learning_rate"]),
                        "increase_count": sum(
                            event["action"] == "increase" for event in epoch_adaptive_events
                        ),
                        "decrease_count": sum(
                            event["action"] == "decrease" for event in epoch_adaptive_events
                        ),
                        "hold_count": sum(
                            event["action"] == "hold" for event in epoch_adaptive_events
                        ),
                    }
                early_stop_epoch_results.append(epoch_result)
                actor_module.train()
                if cfg.freeze_vision_encoder:
                    actor_module.model.vision_encoder.eval()

            if early_stop:
                break

        if cfg.zero_endpoint_pcgrad_train and endpoint_pcgrad_step_metrics:
            projection_active_fraction = float(
                np.mean(
                    [
                        metric["projection_active"]
                        for metric in endpoint_pcgrad_step_metrics
                    ]
                )
            )
            endpoint_pcgrad_iteration_result = {
                "iteration": iteration,
                "num_actor_steps": len(endpoint_pcgrad_step_metrics),
                "projection_active_fraction": projection_active_fraction,
                "mean_anchor_mse": float(
                    np.mean(
                        [metric["anchor_mse"] for metric in endpoint_pcgrad_step_metrics]
                    )
                ),
                "mean_gradient_cosine": float(
                    np.mean(
                        [
                            metric["gradient_cosine"]
                            for metric in endpoint_pcgrad_step_metrics
                        ]
                    )
                ),
                "mean_gradient_norm_retention": float(
                    np.mean(
                        [
                            metric["gradient_norm_retention"]
                            for metric in endpoint_pcgrad_step_metrics
                        ]
                    )
                ),
                "steps": endpoint_pcgrad_step_metrics,
            }
            endpoint_pcgrad_training_history.append(endpoint_pcgrad_iteration_result)
            endpoint_pcgrad_output_path = run_dir / "zero_endpoint_pcgrad_training.json"
            endpoint_pcgrad_output_path.write_text(
                json.dumps(
                    {"iterations": endpoint_pcgrad_training_history},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            logger.info(
                "Zero-endpoint PCGrad training: iteration=%d active=%.2f%% "
                "mean_cosine=%.6f mean_norm_retention=%.2f%%",
                iteration,
                100 * projection_active_fraction,
                endpoint_pcgrad_iteration_result["mean_gradient_cosine"],
                100
                * endpoint_pcgrad_iteration_result[
                    "mean_gradient_norm_retention"
                ],
            )

        if (
            direct_advantage_head is not None
            and iteration == cfg.direct_advantage_audit_iteration - 1
        ):
            actor_module.eval()
            critic.eval()
            direct_advantage_head.train()
            (
                direct_train_observations,
                direct_train_behavior_actions,
                direct_train_center_actions,
            ) = direct_advantage_inputs(
                b_actions=b_actions,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
                seed_offset=iteration * 7919 + 1,
            )
            chunks_per_rollout = steps_per_iteration // n_action_steps
            with torch.no_grad():
                fixed_baseline_values = critic(direct_train_observations).reshape(
                    chunks_per_rollout, num_envs_per_process
                )
                final_observations = actor_module.normalize_inputs(copy.deepcopy(next_obs))
                final_observation_cond = actor_module.model.encode_observations(
                    final_observations
                )
                final_baseline_values = critic(final_observation_cond).reshape(-1)
                macro_rewards, macro_terminals = discounted_macro_rewards(
                    rewards_stored.to(device),
                    terminals_stored.to(device),
                    n_action_steps,
                    cfg.discount,
                )
                valid_chunks = (
                    b_cfm_value_invalid.sum(dim=1) == 0
                ).reshape(chunks_per_rollout, num_envs_per_process).to(device)
            direct_advantage_training_losses = []
            for _ in range(10):
                optimizer_direct_advantage.zero_grad(set_to_none=True)
                direct_train_advantages = direct_advantage_head(
                    direct_train_observations,
                    direct_train_behavior_actions,
                    direct_train_center_actions,
                ).reshape(chunks_per_rollout, num_envs_per_process)
                direct_residuals = direct_advantage_residuals(
                    direct_train_advantages,
                    macro_rewards,
                    macro_terminals,
                    fixed_baseline_values,
                    final_baseline_values,
                    valid_chunks,
                    cfg.discount**n_action_steps,
                    cfg.direct_advantage_horizon_chunks,
                )
                direct_loss = direct_residuals.square().mean()
                if not torch.isfinite(direct_loss):
                    raise RuntimeError("direct-advantage training produced a non-finite loss")
                direct_loss.backward()
                nn.utils.clip_grad_norm_(direct_advantage_head.parameters(), cfg.max_grad_norm)
                optimizer_direct_advantage.step()
                direct_advantage_training_losses.append(float(direct_loss.item()))
            logger.info(
                "Direct-advantage side training: windows=%d first_loss=%.6f final_loss=%.6f",
                direct_residuals.numel(),
                direct_advantage_training_losses[0],
                direct_advantage_training_losses[-1],
            )
            actor_module.train()
            critic.train()
            if cfg.freeze_vision_encoder:
                actor_module.model.vision_encoder.eval()

        if early_stop_audit_tensors is not None:
            active_fractions = [
                result["pooled_active_positive_ratio_fraction"]
                for result in early_stop_epoch_results
            ]
            selected_epoch = select_epoch_before_ratio_violation(
                active_fractions,
                cfg.heldout_ratio_early_stop_threshold,
            )
            early_stop_result = {
                "iteration": iteration,
                "num_positive_chunks_available": early_stop_positive_chunks_available,
                "num_chunks_audited": int(early_stop_audit_tensors[0].numel()),
                "num_batches": 2,
                "cfm_samples": n_action_samples,
                "active_ratio_threshold": cfg.heldout_ratio_early_stop_threshold,
                "trust_region_mode": cfg.trust_region_mode,
                "rollback_alpha": cfg.rollback_alpha,
                "rollback_active_elements": rollback_active_elements,
                "rollback_total_elements": rollback_total_elements,
                "rollback_active_fraction": (
                    rollback_active_elements / rollback_total_elements
                    if rollback_total_elements
                    else 0.0
                ),
                "adaptive_actor_lr": cfg.adaptive_actor_lr,
                "adaptive_actor_lr_target_kl": cfg.adaptive_actor_lr_target_kl,
                "actor_lr_at_policy_update_start": actor_lr_at_policy_update_start,
                "adaptive_actor_lr_events": adaptive_lr_events,
                "preupdate_batches": early_stop_pre_results,
                "selected_epoch": selected_epoch,
                "epochs": early_stop_epoch_results,
            }
            early_stop_output_path = (
                Path(cfg.heldout_ratio_early_stop_audit_output_json)
                if cfg.heldout_ratio_early_stop_audit_output_json is not None
                else run_dir / "heldout_ratio_early_stop_audit.json"
            )
            early_stop_output_path.parent.mkdir(parents=True, exist_ok=True)
            early_stop_output_path.write_text(
                json.dumps(early_stop_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "Held-out ratio early-stop audit: %s",
                json.dumps(early_stop_result, sort_keys=True),
            )

        if replay_audit_indices is not None:
            actor_module.eval()
            post_ratios, post_gradient = replay_audit_gradient(
                replay_audit_indices,
                torch.ones(replay_audit_indices.numel()),
                b_actions=b_actions,
                b_cfm_losses=b_cfm_losses,
                b_cfm_loss_ts=b_cfm_loss_ts,
                b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                b_cfm_value_invalid=b_cfm_value_invalid,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
            )
            replay_gradient_cosine = torch.nn.functional.cosine_similarity(
                replay_audit_pre_gradient.unsqueeze(0), post_gradient.unsqueeze(0), dim=1
            ).item()
            replay_result = {
                "iteration": iteration,
                "num_successful_chunks_available": int(success_indices.numel()),
                "num_successful_chunks_audited": int(replay_audit_indices.numel()),
                "pre_ratio_mean": float(pre_ratios.mean().item()),
                "pre_ratio_std": float(pre_ratios.std(unbiased=False).item()),
                "pre_ratio_min": float(pre_ratios.min().item()),
                "pre_ratio_max": float(pre_ratios.max().item()),
                "pre_replay_gradient_norm": float(replay_audit_pre_gradient.norm().item()),
                "fresh_positive_gradient_norm": replay_audit_positive_gradient_norm,
                "post_replay_gradient_norm": float(post_gradient.norm().item()),
                "active_positive_ratio_fraction": float(
                    (post_ratios <= 1 + cfg.clip_coef).float().mean().item()
                ),
                "ratio_ess_fraction": float(effective_sample_fraction(post_ratios).item()),
                "ratio_mean": float(post_ratios.mean().item()),
                "ratio_std": float(post_ratios.std(unbiased=False).item()),
                "ratio_min": float(post_ratios.min().item()),
                "ratio_max": float(post_ratios.max().item()),
                "pre_post_replay_gradient_cosine": replay_gradient_cosine,
                "success_fresh_positive_gradient_cosine": replay_audit_selection_cosine,
            }
            replay_output_path = (
                Path(cfg.success_replay_audit_output_json)
                if cfg.success_replay_audit_output_json is not None
                else run_dir / "success_replay_audit.json"
            )
            replay_output_path.parent.mkdir(parents=True, exist_ok=True)
            replay_output_path.write_text(json.dumps(replay_result, indent=2, sort_keys=True) + "\n")
            logger.info("Success replay audit: %s", json.dumps(replay_result, sort_keys=True))
            actor.train()

        if ratio_audit_indices is not None:
            actor_module.eval()
            unit_weights = torch.ones(ratio_audit_indices.numel())
            stored_post_ratios, stored_post_gradient = replay_audit_gradient(
                ratio_audit_indices,
                unit_weights,
                b_actions=b_actions,
                b_cfm_losses=b_cfm_losses,
                b_cfm_loss_ts=b_cfm_loss_ts,
                b_cfm_loss_epsilons=b_cfm_loss_epsilons,
                b_cfm_value_invalid=b_cfm_value_invalid,
                b_obs_images=b_obs_images,
                b_obs_state=b_obs_state,
            )
            (
                heldout_local_indices,
                heldout_actions,
                heldout_losses,
                heldout_times_stored,
                heldout_noises_stored,
                heldout_invalid,
                heldout_images,
                heldout_states,
            ) = ratio_audit_heldout_tensors
            heldout_post_ratios, heldout_post_gradient = replay_audit_gradient(
                heldout_local_indices,
                unit_weights,
                b_actions=heldout_actions,
                b_cfm_losses=heldout_losses,
                b_cfm_loss_ts=heldout_times_stored,
                b_cfm_loss_epsilons=heldout_noises_stored,
                b_cfm_value_invalid=heldout_invalid,
                b_obs_images=heldout_images,
                b_obs_state=heldout_states,
            )
            stored_active = (stored_post_ratios <= 1 + cfg.clip_coef).float().mean().item()
            heldout_active = (heldout_post_ratios <= 1 + cfg.clip_coef).float().mean().item()
            stored_median_abs_logratio = stored_post_ratios.log().abs().median().item()
            heldout_median_abs_logratio = heldout_post_ratios.log().abs().median().item()
            heldout_gradient_cosine = torch.nn.functional.cosine_similarity(
                ratio_audit_heldout_pre_gradient.unsqueeze(0),
                heldout_post_gradient.unsqueeze(0),
                dim=1,
            ).item()
            ratio_audit_result = {
                "iteration": iteration,
                "num_positive_chunks_available": int(positive_indices.numel()),
                "num_chunks_audited": int(ratio_audit_indices.numel()),
                "stored_active_positive_ratio_fraction": stored_active,
                "heldout_active_positive_ratio_fraction": heldout_active,
                "heldout_active_fraction_gain": heldout_active - stored_active,
                "stored_median_absolute_logratio": stored_median_abs_logratio,
                "heldout_median_absolute_logratio": heldout_median_abs_logratio,
                "heldout_to_stored_median_absolute_logratio": (
                    heldout_median_abs_logratio / max(stored_median_abs_logratio, 1e-12)
                ),
                "heldout_ratio_ess_fraction": float(
                    effective_sample_fraction(heldout_post_ratios).item()
                ),
                "stored_pre_gradient_norm": float(
                    ratio_audit_stored_pre_gradient.norm().item()
                ),
                "stored_post_gradient_norm": float(stored_post_gradient.norm().item()),
                "heldout_pre_gradient_norm": float(
                    ratio_audit_heldout_pre_gradient.norm().item()
                ),
                "heldout_post_gradient_norm": float(heldout_post_gradient.norm().item()),
                "heldout_pre_post_gradient_cosine": heldout_gradient_cosine,
                "stored_ratio_mean": float(stored_post_ratios.mean().item()),
                "heldout_ratio_mean": float(heldout_post_ratios.mean().item()),
            }
            ratio_output_path = (
                Path(cfg.cfm_ratio_generalization_audit_output_json)
                if cfg.cfm_ratio_generalization_audit_output_json is not None
                else run_dir / "cfm_ratio_generalization_audit.json"
            )
            ratio_output_path.parent.mkdir(parents=True, exist_ok=True)
            ratio_output_path.write_text(
                json.dumps(ratio_audit_result, indent=2, sort_keys=True) + "\n"
            )
            logger.info(
                "CFM ratio generalization audit: %s",
                json.dumps(ratio_audit_result, sort_keys=True),
            )
            actor.train()

        # Metrics (rank local)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        # Track CFM/DPPO metrics for logging (use last minibatch values as representative)
        if cfg.loss_mode == "fpo":
            cfm_metrics = {
                "cfm/old_cfm_loss_mean": float(old_cfm_loss.mean().item()),
                "cfm/old_cfm_loss_std": float(old_cfm_loss.std().item()),
                "cfm/curr_cfm_loss_mean": float(curr_cfm_loss.mean().item()),
                "cfm/curr_cfm_loss_std": float(curr_cfm_loss.std().item()),
                "cfm/logratio_mean": float(logratio.mean().item()),
                "cfm/logratio_std": float(logratio.std().item()),
                "cfm/logratio_min": float(logratio.min().item()),
                "cfm/logratio_max": float(logratio.max().item()),
                "cfm/ratio_mean": float(ratio.mean().item()),
                "cfm/ratio_std": float(ratio.std().item()),
                "cfm/ratio_min": float(ratio.min().item()),
                "cfm/ratio_max": float(ratio.max().item()),
                "cfm/old_cfm_loss_hist": wandb.Histogram(old_cfm_loss.detach().cpu().numpy().flatten()),
                "cfm/curr_cfm_loss_hist": wandb.Histogram(curr_cfm_loss.detach().cpu().numpy().flatten()),
            }
            if cfg.advantage_weighting == "ess_softmax":
                cfm_metrics.update({
                    "cfm/advantage_weight_temperature": float(advantage_weight_temperature.item()),
                    "cfm/advantage_weight_ess_fraction": float(advantage_weight_ess_fraction.item()),
                    "cfm/advantage_weight_min": float(mirror_weights.min().item()),
                    "cfm/advantage_weight_max": float(mirror_weights.max().item()),
                })
        else:  # dppo
            cfm_metrics = {
                "dppo/log_prob_chunk_mean": float(log_prob_chunk.mean().item()),
                "dppo/log_prob_chunk_std": float(log_prob_chunk.std().item()),
                "dppo/old_log_prob_chunk_mean": float(mb_dppo_log_probs_chunk.mean().item()),
                "dppo/old_log_prob_chunk_std": float(mb_dppo_log_probs_chunk.std().item()),
                "dppo/log_ratio_mean": float(log_ratio.mean().item()),
                "dppo/log_ratio_std": float(log_ratio.std().item()),
                "dppo/log_ratio_min": float(log_ratio.min().item()),
                "dppo/log_ratio_max": float(log_ratio.max().item()),
                "dppo/ratio_mean": float(ratio.mean().item()),
                "dppo/ratio_std": float(ratio.std().item()),
                "dppo/ratio_min": float(ratio.min().item()),
                "dppo/ratio_max": float(ratio.max().item()),
                "dppo/sde_sigma_mean": float(sde_sigma.mean().item()),
                "dppo/sde_sigma_std": float(sde_sigma.std().item()),
                "dppo/sde_sigma_min": float(sde_sigma.min().item()),
                "dppo/sde_sigma_max": float(sde_sigma.max().item()),
            }
            # Add entropy metrics (handles both learned and fixed sigma cases)
            if cfg.learn_sde_sigma and isinstance(entropy, torch.Tensor):
                # Learned sigma: entropy has shape (B, T)
                cfm_metrics["dppo/entropy_mean"] = float(entropy.mean().item())
                cfm_metrics["dppo/entropy_std"] = float(entropy.std().item())
            else:
                # Fixed sigma: entropy is scalar 0.0
                cfm_metrics["dppo/entropy_mean"] = float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)

        training_cum_time += time.time() - iteration_start_time
        sps_local_cum = int((steps_per_iteration * num_envs_per_process) / (training_cum_time / max(1, iteration)))

        # Log (rank 0 only)
        if rank == 0 and iteration % cfg.log_freq == 0:
            msg = (
                f"[iter {iteration:>6d}/{num_iterations}]"
                f" SR: {success_rate_global:.2%}"
                f" | SPS_local_est: {sps_local_cum}"
                f" | pg_loss: {float(pg_loss):.4f}"
                f" | v_loss: {float(v_loss):.4f}"
                f" | total_loss: {float(loss):.4f}"
                f" | valid_cfm: {valid_cfm_action_fraction:.2%}"
            )
            if cfg.advantage_weighting == "ess_softmax":
                msg += (
                    f" | adv_tau: {float(advantage_weight_temperature):.4f}"
                    f" | adv_ess: {float(advantage_weight_ess_fraction):.2%}"
                )
            logger.info(colored(msg, "green"))
            if cfg.wandb_enable:
                log_dict = {
                    "training/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                    "training/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                    "training/SPS_local_est": sps_local_cum,
                    "training/actor_grad_norm_before_clip": float(actor_grad_norm_before),
                    "training/actor_grad_norm_after_clip": float(actor_grad_norm_after),
                    "training/critic_grad_norm_before_clip": float(critic_grad_norm_before),
                    "training/critic_grad_norm_after_clip": float(critic_grad_norm_after),
                    "charts/rewards": b_rewards.sum().item(),
                    "charts/success_rate": success_rate_global,
                    "charts/success_rate_guided_source": guided_success_rate_global,
                    "charts/success_rate_random_source": random_success_rate_global,
                    "charts/valid_cfm_action_fraction": valid_cfm_action_fraction,
                    "charts/action_norm_mean": action_norms.mean(),
                    "charts/action_norm_std": action_norms.std(),
                    "values/advantages": b_advantages.mean().item(),
                    "values/returns": b_returns.mean().item(),
                    "values/values": b_values.mean().item(),
                    "losses/value_loss": float(v_loss),
                    "losses/policy_loss": float(pg_loss),
                    "losses/total_loss": float(loss),
                    "losses/clipfrac": float(np.mean(clipfracs)) if len(clipfracs) else 0.0,
                    "losses/explained_variance": explained_var,
                }
                # Add CFM/DPPO specific metrics
                log_dict.update(cfm_metrics)
                wandb.log(log_dict, step=global_step)

        # Step schedulers
        if not (
            cfg.adaptive_actor_lr
            and iteration > cfg.n_iterations_train_only_value
        ):
            lr_scheduler_actor.step()
        lr_scheduler_critic.step()
        if lr_scheduler_success_critic is not None:
            lr_scheduler_success_critic.step()

        # Checkpointing (rank 0)
        if rank == 0 and cfg.save_freq > 0 and iteration % cfg.save_freq == 0:
            model_to_save = actor.module if is_ddp else actor
            ckpt_dir = (run_dir / "checkpoints") / f"step_{global_step}"
            save_checkpoint(ckpt_dir, global_step, model_to_save, optimizer_actor)

            latest_dir = (run_dir / "checkpoints") / "latest"
            if latest_dir.exists():
                import shutil
                shutil.rmtree(latest_dir)
            save_checkpoint(latest_dir, global_step, model_to_save, optimizer_actor)

            torch.save(
                {
                    "critic_state_dict": (critic.module.state_dict() if is_ddp else critic.state_dict()),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": vars(cfg),
                    "success_rate": success_rate_global,
                    "training_cum_time": training_cum_time,
                },
                latest_dir / "ppo_state.pt",
            )
            logger.info(colored(f"Checkpoint saved @ {ckpt_dir}", "magenta"))

            if cfg.wandb_enable and wandb.run is not None:
                try:
                    checkpoint_artifact = wandb.Artifact(
                        name=f"checkpoint_step_{global_step}",
                        type="model",
                        description=f"Model checkpoint at global_step {global_step}",
                        metadata={"step": global_step, "loss": float(loss)},
                    )
                    checkpoint_artifact.add_dir(str(ckpt_dir))
                    wandb.log_artifact(checkpoint_artifact)

                    latest_artifact = wandb.Artifact(
                        name="checkpoint_latest",
                        type="model",
                        description=f"Latest model checkpoint (global_step {global_step})",
                        metadata={"step": global_step, "loss": float(loss)},
                    )
                    latest_artifact.add_dir(str(latest_dir))
                    wandb.log_artifact(latest_artifact, aliases=["latest"])
                    logger.info(colored("Checkpoints uploaded to W&B", "magenta"))
                except Exception as e:
                    logger.warning(colored(f"Failed to upload checkpoint to W&B: {e}", "yellow"))

        # Evaluation (all ranks compute; rank 0 logs)
        if (
            cfg.rollout_freq is not None
            and cfg.eval_env is not None
            and (iteration % cfg.rollout_freq == 0 or iteration == num_iterations or iteration == 1)
        ):
            if is_ddp:
                dist.barrier()  # make sure training step is aligned

            t0 = time.perf_counter()
            results = eval_all_ranks(
                env=env,
                num_envs_per_process=num_envs_per_process,
                actor_ddp_or_single=actor,
                world_size=world_size,
                rank=rank,
                is_ddp=is_ddp,
                device_str=device_str,
                device=device,
                run_dir=run_dir,
                cfg=cfg,
                create_env_fn=create_vectorized_env,
                global_step=global_step,
            )
            if is_ddp:
                dist.barrier()

            if rank == 0:
                rollout_ms = (time.perf_counter() - t0) * 1000
                eval_sr = results["success_rate"]
                eval_fps = results["fps"]
                video_path = results["video_path"]
                episodes = results["episodes"]

                eval_sr_non_zero = results["success_rate_non_zero"]
                eval_fps_non_zero = results["fps_non_zero"]

                logger.info(colored(
                    f"[iteration {iteration:>6d}] global eval SR: {eval_sr*100:.1f}% | global episodes: {episodes} | "
                    f"rollout: {rollout_ms/1000:.2f}s | ~fps: {eval_fps:.1f}",
                    "cyan"
                ))

                if cfg.wandb_enable:
                    log_data = {
                        "eval/success_rate_zero_sampling": eval_sr,
                        "eval/fps_zero_sampling": eval_fps,
                        "eval/success_rate_random_sampling": eval_sr_non_zero,
                        "eval/fps_random_sampling": eval_fps_non_zero,
                        "eval/episodes_total": results["episodes"],
                        "time/rollout_ms": rollout_ms,
                    }
                    if video_path is not None and video_path.exists():
                        try:
                            video_artifact = wandb.Artifact(
                                name=f"eval_video_step_{global_step}",
                                type="video",
                                description=f"Evaluation video at global_step {global_step}",
                                metadata={"step": global_step, "success_rate": eval_sr},
                            )
                            video_artifact.add_file(str(video_path))
                            wandb.log_artifact(video_artifact)
                            log_data["eval/rollout_video"] = wandb.Video(str(video_path), format="mp4")
                        except Exception as e:
                            logger.warning(colored(f"Failed to upload video to W&B: {e}", "yellow"))
                    wandb.log(log_data, step=global_step)

            # Save best
            if rank == 0 and eval_sr > best_eval_success_rate:
                best_eval_success_rate = eval_sr
                logger.info(colored(f"New best success-rate! Saving checkpoint at global step {global_step}", "magenta"))
                model_to_save = actor.module if is_ddp else actor
                best_dir = (run_dir / "checkpoints") / "best"
                if best_dir.exists():
                    import shutil
                    shutil.rmtree(best_dir)
                save_checkpoint(best_dir, global_step, model_to_save, optimizer_actor)
                torch.save(
                    {
                        "critic_state_dict": (critic.module.state_dict() if is_ddp else critic.state_dict()),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": vars(cfg),
                        "success_rate": eval_sr,
                        "training_cum_time": training_cum_time,
                    },
                    best_dir / "ppo_state.pt",
                )
                if cfg.wandb_enable and wandb.run is not None:
                    try:
                        best_artifact = wandb.Artifact(
                            name="checkpoint_best",
                            type="model",
                            description=f"Best model checkpoint (global_step {global_step}, SR: {eval_sr * 100:.1f}%)",
                            metadata={"step": global_step, "success_rate": eval_sr},
                        )
                        best_artifact.add_dir(str(best_dir))
                        wandb.log_artifact(best_artifact, aliases=["best"])
                        logger.info(colored("Best checkpoint uploaded to W&B", "magenta"))
                    except Exception as e:
                        logger.warning(colored(f"Failed to upload best checkpoint to W&B: {e}", "yellow"))

        if is_ddp:
            dist.barrier()

    logger.info(colored("Training finished!", "green", attrs=["bold"]))
    if rank == 0:
        if cfg.wandb_enable:
            wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    args_cli = tyro.cli(FlowPPOConfig, config=(tyro.conf.FlagConversionOff,))
    main(args_cli)
