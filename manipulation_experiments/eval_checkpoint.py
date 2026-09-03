#!/usr/bin/env python

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
import imageio
import numpy as np
import torch
import tyro
import wandb
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from termcolor import colored

from src.dexmg_env import VectorizedEnvWrapper, create_vectorized_env
from src.flow_model import FlowMatchingPolicy
from src.flow_model_config import FlowMatchingConfig

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# -------------------------------------------------
# Setup logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalCheckpointConfig:
    """Configuration for evaluating checkpoints."""

    # Checkpoint loading options
    wandb_run_id: Optional[str] = None
    """WandB run ID to download checkpoint from."""
    wandb_project: Optional[str] = None
    """WandB project name (required when using wandb_run_id)."""
    wandb_entity: str = "far-wandb"
    """WandB entity name."""
    checkpoint_step: Optional[str] = "latest"
    """Checkpoint step to evaluate. Can be 'latest', 'best', or a specific step number (e.g., 'step_3000')."""
    local_checkpoint_path: Optional[str] = None
    """Local path to checkpoint directory (alternative to wandb_run_id)."""
    load_ema: bool = False
    """If True, load EMA weights from checkpoint (if available) instead of regular weights."""

    # Evaluation environment configuration
    eval_env: str = "Lift"    
    """Environment name for evaluation."""
    zero_sampling: bool = True
    """If True, use zero sampling for evaluation."""
    # image_observation_keys: Optional[str] = None # "agentview_image robot0_eye_in_hand_image"
    # """Image observation keys to use for policy input (e.g., --image_observation_keys "robot0_eye_in_hand_image shouldercamera1_image"."""
    eval_num_episodes: int = 50
    """Number of episodes for evaluation."""
    eval_num_envs: int = 2
    """Number of parallel environments for evaluation."""
    eval_camera_size: int = 84
    """Camera size for evaluation."""
    camera_name: str = "agentview"
    """Camera name for video rendering."""

    # System configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for evaluation."""
    debug: bool = False
    """Enable debug mode (uses synchronous vectorized environments)."""
    seed: Optional[int] = None
    """Random seed for reproducibility."""

    # Output configuration
    output_dir: Optional[str] = None
    """Output directory for evaluation results. If None, will auto-generate."""
    save_video: bool = True
    """Whether to save evaluation video."""
    render_size: tuple[int, int] = (240, 320)
    """Resolution (height, width) for rendered rollout frames. This is different from eval_camera_size which controls policy input."""
    annotate_video: bool = True
    """Whether to annotate video frames with episode info (env idx, episode number, step, success/fail status)."""

    # Logging configuration
    wandb_enable: bool = True
    """Enable Weights & Biases logging for evaluation results."""
    experiment: str = "eval_checkpoint"
    """Experiment name for logging."""


# -----------------------------------------------------------------------------
# Evaluation helpers (reused from pretrain_flow_bc.py)
# -----------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    env_idx: int,
    episode_num: int,
    total_episodes: int,
    episode_step: int,
    is_success: bool,
    font=None,
) -> np.ndarray:
    """Annotate a single frame with episode information."""
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)

    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)

    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: FlowMatchingPolicy,
    env: VectorizedEnvWrapper,
    save_dir: Path,
    step: str,
    num_episodes: int,
    task: str,
    save_video: bool = True,
    zero_sampling: bool = False,
    annotate_video: bool = True,
):
    """Run *num_episodes* episodes with *policy* in vectorized *env* and compute success-rate."""
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_was_training = policy.training
    policy.eval()

    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")

    successes_list = [[] for _ in range(num_parallel_envs)]
    dones_list = [[] for _ in range(num_parallel_envs)]
    done_episodes = sum(sum(dones_list[env_idx]) for env_idx in range(num_parallel_envs))
    total_steps = 0
    start_time = time.perf_counter()

    logger.info(f"Running rollouts with environment: {env_name}")
    logger.info(f"Starting evaluation: {num_episodes} episodes using {num_parallel_envs} parallel environments")

    # Try to load a font for text annotations
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Create video writer if saving video
    video_writer = None
    video_path = None
    if save_video:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = save_dir / f"eval_step_{step}_{now}.mp4"
        video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    obs, _ = env.reset()
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs
    episode_returns = [0.0] * num_parallel_envs
    all_episode_returns = []
    all_episode_lengths = []
    all_episode_successes = []

    policy.reset()

    while done_episodes < num_episodes:
        with torch.inference_mode():
            action, _ = policy.select_action(obs, zero_sampling=zero_sampling)

        obs, reward, terminated, truncated, _ = env.step(action)

        # Track episode returns
        for env_idx in range(num_parallel_envs):
            episode_returns[env_idx] += reward[env_idx].item()
            episode_steps[env_idx] += 1

        if save_video:
            frames = env.render()
            for env_idx in range(num_parallel_envs):
                episode_frames[env_idx].append(frames[env_idx])

        total_steps += num_parallel_envs
        done = terminated | truncated

        finish_eval = False
        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]

            policy.reset(env_ids=terminated_envs)
    
            for env_idx_tensor in terminated_envs:
                env_idx = int(env_idx_tensor.item())
                is_success = env_idx_tensor in success_envs
                dones_list[env_idx].append(1)
                successes_list[env_idx].append(int(is_success))

                # Last render is the reset observation of the next episode; drop it.
                # Skip when video is off — frames are never appended in that path.
                if save_video and episode_frames[env_idx]:
                    episode_frames[env_idx].pop(-1)
                if episode_steps[env_idx] > 0:
                    episode_steps[env_idx] -= 1

                # Record per-episode stats
                all_episode_returns.append(episode_returns[env_idx])
                all_episode_lengths.append(episode_steps[env_idx])
                all_episode_successes.append(int(is_success))
                episode_returns[env_idx] = 0.0

                if save_video and video_writer is not None:
                    for step_idx, frame in enumerate(episode_frames[env_idx]):
                        if annotate_video:
                            frame = _annotate_frame(
                                frame=frame,
                                env_idx=env_idx,
                                episode_num=done_episodes,
                                total_episodes=num_episodes,
                                episode_step=step_idx + 1,
                                is_success=is_success,
                                font=font,
                            )
                        video_writer.append_data(frame)

                episode_frames[env_idx] = []
                episode_steps[env_idx] = 0

                done_episodes = sum(sum(dones_list[env_idx]) for env_idx in range(num_parallel_envs))
                if done_episodes >= num_episodes:
                    finish_eval = True
                    break
        
        if finish_eval:
            break

        done_episodes = sum(sum(dones_list[env_idx]) for env_idx in range(num_parallel_envs))
        successes = sum(sum(successes_list[env_idx]) for env_idx in range(num_parallel_envs))

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {done_episodes}, successes: {successes}, "
                f"FPS: {total_steps / (time.perf_counter() - start_time):.1f}"
            )

    if save_video and video_writer is not None:
        video_writer.close()

    done_episodes = sum(sum(dones_list[env_idx]) for env_idx in range(num_parallel_envs))
    successes = sum(sum(successes_list[env_idx]) for env_idx in range(num_parallel_envs))

    success_rate = successes / done_episodes if done_episodes > 0 else 0.0
    avg_return = np.mean(all_episode_returns) if all_episode_returns else 0.0
    std_return = np.std(all_episode_returns) if all_episode_returns else 0.0

    if policy_was_training:
        policy.train()

    total_elapsed_time = time.perf_counter() - start_time
    final_fps = total_steps / total_elapsed_time if total_elapsed_time > 0 else 0.0
    episodes_per_sec = done_episodes / total_elapsed_time if total_elapsed_time > 0 else 0.0

    # Compute mean and std of successes_list across environments
    successes_per_env = [sum(successes_list[env_idx]) for env_idx in range(num_parallel_envs)]
    done_episodes_per_env = [sum(dones_list[env_idx]) for env_idx in range(num_parallel_envs)]
    success_rates_per_env = [successes_per_env[env_idx] / done_episodes_per_env[env_idx] for env_idx in range(num_parallel_envs)]
    mean_success_rate = np.mean(success_rates_per_env)
    std_success_rate = np.std(success_rates_per_env)

    avg_episode_length = np.mean(all_episode_lengths) if all_episode_lengths else 0.0
    std_episode_length = np.std(all_episode_lengths) if all_episode_lengths else 0.0

    logger.info(f"Evaluation completed: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%) for task **{task}**")
    logger.info(f"Success stats across environments: mean={mean_success_rate:.2f} +/- {std_success_rate:.2f}")
    logger.info(f"Average return: {avg_return:.3f} +/- {std_return:.3f}")
    logger.info(f"Average episode length: {avg_episode_length:.1f} +/- {std_episode_length:.1f}")
    logger.info(f"Performance: {total_steps} total steps in {total_elapsed_time:.1f}s")
    logger.info(f"Average FPS: {final_fps:.1f} frames/sec | Episodes/sec: {episodes_per_sec:.2f}")
    if save_video and video_path:
        logger.info(f"Video saved: {video_path}")

    return mean_success_rate, std_success_rate, avg_return, std_return, video_path, final_fps, all_episode_returns, all_episode_lengths, all_episode_successes


def download_checkpoint_from_wandb(
    run_id: str,
    project: str,
    entity: str,
    artifact_alias: str = "latest",
    download_dir: Path = Path("./downloaded_checkpoints")
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


def load_policy(checkpoint_dir: Path, device: str = "cuda", load_ema: bool = False) -> FlowMatchingPolicy:
    """Load policy from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        device: Device to load the policy on
        load_ema: If True and EMA weights are available, load EMA weights into the model
    """
    policy_path = checkpoint_dir / "policy"

    if not policy_path.exists():
        # Maybe the checkpoint_dir is already the policy directory
        if (checkpoint_dir / "model.safetensors").exists():
            policy_path = checkpoint_dir
        else:
            raise ValueError(f"Policy path not found: {policy_path}")

    logger.info(colored(f"Loading policy from {policy_path}...", "cyan"))

    # Load the config from config.json
    config_path = policy_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    logger.info(colored(f"Loading config from {config_path}...", "cyan"))
    with open(config_path, 'r') as f:
        config_dict = json.load(f)  # type: ignore
        config_dict.pop('type')
        config_dict.pop('normalization_mapping')
        config = FlowMatchingConfig(**config_dict)

    # Set image_features and state_features from input_features (similar to pretrain_flow_bc.py)
    config.image_features = [k for k in config.input_features if "image" in k]
    config.state_features = [k for k in config.input_features if "state" in k or "pos" in k]
    logger.info(f"Image features: {config.image_features}")
    logger.info(f"State features: {config.state_features}")

    # Construct the policy using the config (without dataset_stats for now)
    logger.info(colored(f"Constructing FlowMatchingPolicy from config...", "cyan"))
    # Signal to create ema model if load_ema is True
    if load_ema:
        config.ema_power = 1.0
    policy = FlowMatchingPolicy(config, dataset_stats=None)

    # Load the model weights from model.safetensors
    weights_path = policy_path / "model.safetensors"
    if not weights_path.exists():
        raise ValueError(f"Model weights file not found: {weights_path}")

    logger.info(colored(f"Loading model weights from {weights_path}...", "cyan"))
    state_dict = load_file(weights_path, device=device)
    policy.load_state_dict(state_dict, strict=True)

    policy.to(device)

    # Load EMA weights if requested
    if load_ema and hasattr(policy, 'ema_model') and policy.ema_model is not None:
        # Try to load EMA state from checkpoint
        actual_checkpoint_dir = policy_path.parent if policy_path != checkpoint_dir else checkpoint_dir
        optimizer_path = actual_checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path, map_location='cpu')
            if 'ema_state_dict' in checkpoint:
                policy.ema_model.to(device)
                policy.ema_model.load_state_dict(checkpoint['ema_state_dict'])
                policy.ema_model.copy_to(policy.model.parameters())
                logger.info(colored("Loaded and applied EMA weights from checkpoint", "green"))
            else:
                logger.warning(colored("--load_ema flag set but no EMA weights found in checkpoint", "yellow"))
        else:
            logger.warning(colored("--load_ema flag set but optimizer.pt not found", "yellow"))

    logger.info(colored(f"Policy loaded successfully", "green"))
    logger.info(f"Policy config: horizon={policy.config.horizon}, n_action_steps={policy.config.n_action_steps}, sampling_steps={policy.config.sampling_steps}")

    return policy


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg: EvalCheckpointConfig):

    logger.info(colored("=" * 80, "cyan"))
    logger.info(colored("Checkpoint Evaluation", "cyan", attrs=["bold"]))
    logger.info(colored("=" * 80, "cyan"))

    # Set random seed if provided
    if cfg.seed is not None:
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # Set up output directory
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.output_dir is None:
        run_dir = Path("runs") / f"{cfg.experiment}_{run_start_time}"
    else:
        run_dir = Path(cfg.output_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(colored(f"Output directory: {run_dir}", "green"))

    # Load checkpoint
    checkpoint_dir = None

    if cfg.wandb_run_id is not None:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when using --wandb_run_id")

        checkpoint_dir = download_checkpoint_from_wandb(
            run_id=cfg.wandb_run_id,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            artifact_alias=cfg.checkpoint_step,
            download_dir=run_dir / "downloaded_checkpoints"
        )
    elif cfg.local_checkpoint_path is not None:
        checkpoint_dir = Path(cfg.local_checkpoint_path)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        logger.info(colored(f"Using local checkpoint: {checkpoint_dir}", "cyan"))
    else:
        raise ValueError("Must provide either --wandb_run_id or --local_checkpoint_path")

    # Load policy
    policy = load_policy(checkpoint_dir, device=cfg.device, load_ema=cfg.load_ema)

    # Create evaluation environment
    logger.info(colored(f"Creating evaluation environment: {cfg.eval_env}", "cyan"))
    device_str = "cpu" if cfg.device == "cpu" else "cuda"

    # Update cfg.image_observation_keys from config fnames
    cfg.image_observation_keys = [k.replace("observation.images.", "") for k in policy.config.image_features]
    logger.info(f"Using image features from checkpoint config: {policy.config.image_features}")

    env = create_vectorized_env(
        env_name=cfg.eval_env,
        num_envs=cfg.eval_num_envs,
        device=device_str,
        camera_size=cfg.eval_camera_size,
        render_size=cfg.render_size,
        video_key=cfg.camera_name,
        debug=cfg.debug,
        expected_image_keys=cfg.image_observation_keys,
        # expected_image_keys=["agentview_image"] # cfg.image_observation_keys,
    )

    # Initialize action buffers
    policy.init_action_buffers(cfg.eval_num_envs)

    # Initialize wandb if enabled
    if cfg.wandb_enable:
        wandb.init(
            project=cfg.wandb_project or "checkpoint-evaluation",
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name=f"{cfg.experiment}_{cfg.eval_env}_{cfg.checkpoint_step}",
            dir=str(run_dir),
            settings=wandb.Settings(),
        )
        logger.info(colored(f"W&B logging enabled", "blue"))

    # Run evaluation
    logger.info(colored("=" * 80, "yellow"))
    logger.info(colored("Starting Evaluation", "yellow", attrs=["bold"]))
    logger.info(colored("=" * 80, "yellow"))

    (mean_success_rate, std_success_rate, avg_return, std_return,
     video_path, final_fps,
     all_episode_returns, all_episode_lengths, all_episode_successes) = _run_rollouts(
        policy=policy,
        env=env,
        save_dir=run_dir / "videos",
        step=cfg.checkpoint_step,
        num_episodes=cfg.eval_num_episodes,
        task=cfg.eval_env,
        save_video=cfg.save_video,
        zero_sampling=cfg.zero_sampling,
        annotate_video=cfg.annotate_video,
    )

    # Print summary
    logger.info(colored("=" * 80, "green"))
    logger.info(colored("Evaluation Summary", "green", attrs=["bold"]))
    logger.info(colored("=" * 80, "green"))
    logger.info(f"Environment: {cfg.eval_env}")
    logger.info(f"Checkpoint: {cfg.checkpoint_step}")
    logger.info(f"Episodes: {cfg.eval_num_episodes}")
    logger.info(f"Success Rate: {mean_success_rate * 100:.2f}% +/- {std_success_rate * 100:.2f}%")
    logger.info(f"Average Return: {avg_return:.3f} +/- {std_return:.3f}")
    logger.info(f"FPS: {final_fps:.1f}")
    logger.info(colored("=" * 80, "green"))

    # Log to wandb
    if cfg.wandb_enable:
        log_data = {
            "eval/mean_success_rate": mean_success_rate,
            "eval/std_success_rate": std_success_rate,
            "eval/avg_return": avg_return,
            "eval/std_return": std_return,
            "eval/fps": final_fps,
            "eval/avg_episode_length": float(np.mean(all_episode_lengths)) if all_episode_lengths else 0.0,
        }

        if video_path is not None and video_path.exists():
            try:
                log_data["eval/rollout_video"] = wandb.Video(str(video_path), format="mp4")
                logger.info(colored(f"Video uploaded to W&B", "cyan"))
            except Exception as e:
                logger.warning(colored(f"Failed to upload video to W&B: {e}", "yellow"))

        # Log per-episode table for detailed analysis
        if all_episode_returns:
            episode_table = wandb.Table(
                columns=["episode", "return", "length", "success"],
                data=[
                    [i, ret, length, success]
                    for i, (ret, length, success) in enumerate(
                        zip(all_episode_returns, all_episode_lengths, all_episode_successes)
                    )
                ],
            )
            log_data["eval/per_episode"] = episode_table

        wandb.log(log_data)
        wandb.finish()

    # Save summary to file
    summary_path = run_dir / f"eval_summary_{cfg.eval_env}_{cfg.checkpoint_step}_zero_sampling_{cfg.zero_sampling}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Environment: {cfg.eval_env}\n")
        f.write(f"Checkpoint: {cfg.checkpoint_step}\n")
        f.write(f"Run ID: {cfg.wandb_run_id}\n")
        f.write(f"Episodes: {cfg.eval_num_episodes}\n")
        f.write(f"Success Rate: {mean_success_rate * 100:.2f}% +/- {std_success_rate * 100:.2f}%\n")
        f.write(f"Average Return: {avg_return:.3f} +/- {std_return:.3f}\n")
        f.write(f"FPS: {final_fps:.1f}\n")
        f.write(f"=" * 80 + "\n")

    logger.info(colored(f"Summary saved to: {summary_path}", "green"))

    # Clean up
    env.close()

    logger.info(colored("Evaluation completed successfully!", "green", attrs=["bold"]))


if __name__ == "__main__":
    args_cli = tyro.cli(EvalCheckpointConfig, config=(tyro.conf.FlagConversionOff,))
    main(args_cli)
