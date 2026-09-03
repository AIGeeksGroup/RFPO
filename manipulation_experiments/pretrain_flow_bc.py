#!/usr/bin/env python

from __future__ import annotations

import copy
import imageio
import json
import logging
import multiprocessing as mp
import os
import time
import tyro
import wandb

from typing import Annotated
from dataclasses import asdict, is_dataclass, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize
from safetensors.torch import load_file

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import set_seed
from termcolor import colored



from src.flow_model_config import FlowMatchingConfig
from src.dexmg_env import VectorizedEnvWrapper, create_vectorized_env
from src.flow_model import FlowMatchingPolicy

# Set multiprocessing start method for CUDA compatibility
# This must be done before any other multiprocessing operations
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set, which is fine
    pass


# A list type, but `tyro` will expect a JSON string from the CLI.
# I need this for Tyro/Wandb sweeping, but looks ugly.
JsonIntList = Annotated[
    list[int],
    tyro.constructors.PrimitiveConstructorSpec(
        # Number of arguments to consume.
        nargs=1,
        # Argument name in usage messages.
        metavar="LIST",
        # Convert a list of strings to an instance. The length of the list
        # should match `nargs`.
        instance_from_str=lambda args: json.loads(args[0]),
        # Check if an instance is of the expected type. This is only used for
        # helptext formatting in the presence of union types.
        is_instance=lambda instance: isinstance(instance, list[int]),
        # Convert an instance to a list of strings. This is used for handling
        # default values that are set in Python. The length of the list should
        # match `nargs`.
        str_from_instance=lambda instance: [json.dumps(instance)],
    ),
]


@dataclass
class TrainFlowBCConfig:
    """Configuration for offline training on a HF Hub dataset with LeRobot policies."""
    
    # Required args
    dataset: str = field(
        metadata={"help": "HF Hub dataset repo-id e.g. `ankile/franka-lift-dataset`"}
    )
    max_num_episodes: Optional[int] = None
    """Maximum number of episodes to load from dataset. If None, loads all episodes. Use this to reduce memory usage or for quick experiments."""
    
    # Policy selection
    policy: Literal[
        "flowmatching"
    ] = "flowmatching"
    """Which policy architecture to train."""
    
    # Training hyper-parameters
    steps: int = 3_000 # training epochs
    """Total optimization steps."""
    batch_size: int = 256
    learning_rate: float = 1e-4
    """Learning rate for myflow."""
    lr_backbone: float = 1e-5
    """Learning rate for vision backbone."""
    weight_decay: float = 1e-6
    """Weight decay for myflow."""
    is_ddp: bool = False
    """Use DDP."""
    grad_clip_norm: float = 10.0
    """Gradient clip norm."""
    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""
    num_workers: int = 4
    """Number of workers for data loading."""
    
    # Reproducibility / device
    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging & checkpoints
    output_dir: Optional[str] = None
    """Output directory for this run. If None, will auto-generate under runs/{experiment}_{timestamp}/"""
    log_freq: int = 100
    """How often to print & log to W&B (in steps)."""
    save_freq: int = 1_000
    """How often to save checkpoints (in steps)."""
    
    # WandB
    wandb_enable: bool = True
    """Enable Weights & Biases logging."""
    wandb_project: Optional[str] = None
    """W&B project name (required when --wandb_enable)."""
    wandb_entity: Optional[str] = "far-wandb"
    """W&B entity name."""
    experiment: Optional[str] = "train_flow_bc"
    """Experiment name."""
    
    # Resume
    resume_ckpt: Optional[str] = None
    """Path to a local checkpoint directory to resume from."""
    resume_run_id: Optional[str] = None
    """WandB run ID to download checkpoint from (downloads checkpoint but starts new wandb run)."""
    checkpoint_step: Optional[str] = "latest"
    """Checkpoint step to load. Can be 'latest', 'best', or a specific step number (e.g., 'step_3000')."""
    resume_wandb_run: bool = False
    """Recommend to be False so that wandb logs are refreshed. If True, resume the actual wandb run (requires resume_run_id). If False, only download checkpoint but start fresh wandb run."""
    load_ema: bool = False
    """If True, load EMA weights from checkpoint (if available) instead of regular weights."""
    
    # Model parameters
    vision_backbone: str = "resnet18" # clip
    """Vision backbone model."""
    enable_geometric_augmentations: bool = False
    """Enable geometric augmentations (rotation, translation, scale)."""
    
    # n_obs_steps: int = 2 # it's just 1
    """Number of observation steps for flow matching."""
    horizon: int = 16
    """Prediction horizon for flow matching."""
    n_action_steps: int = 8
    """Number of action steps to execute for flow matching."""
    sampling_steps: int = 10
    """Number of sampling steps for flow model."""
    ema_power: float = 0.995
    """EMA power for flow matching (0.0 means no EMA)."""
    grad_clip_norm: float = 1.0
    """Gradient clip norm."""

    network_architecture: Literal["unet", "mlp", "residual_mlp"] = "mlp"
    """Network architecture: 'unet' for 1D-CNN based, 'mlp' for MLP based."""
    mlp_dims: JsonIntList = field(default_factory=lambda: [512, 512, 512])
    """Hidden dimensions for MLP layers (only used if network_architecture='mlp'). (e.g., --mlp-dims "[512, 512, 512]") If None, uses [512, 512, 512]."""

    flow_network_output_param: Literal["u", "x0"] = "u" # "u" for velocity, "x0" for data
    """Flow network output: 'u' for velocity, 'x0' for data."""
    cfm_loss_mode: Literal["u", "x0", "eps"] = "u" # "u" for velocity, "x0" for data, "eps" for epsilon
    """CFM loss mode: 'u' for velocity, 'x0' for data, 'eps' for epsilon."""
    transported_clip_value: Optional[float] = None
    """Clip transported predictions (x0 or u) to [-value, value]. None means no clipping."""
    cfm_loss_use_huber: bool = False
    """Use Huber loss instead of MSE loss."""
    cfm_loss_huber_delta: float = 0.5
    """Huber loss delta."""


    # Evaluation rollouts
    rollout_freq: Optional[int] = 100
    """Frequency (in optimization steps) at which to run rollouts in a DexMimicGen environment to compute the success-rate of the current policy. Disabled when not set."""
    eval_env: Optional[str] = None
    """DexMimicGen environment name used for evaluation rollouts."""
    eval_num_envs: int = 2
    """Number of parallel environments for evaluation."""
    eval_num_episodes: int = 10
    """Total number of episodes to run during evaluation."""
    eval_camera_size: int = 84
    """Camera image size for evaluation rollouts (should match dataset)."""
    debug: bool = False
    """Enable debug mode (uses synchronous vectorized environments instead of async multiprocessing for easier debugging)."""
    image_observation_keys: Optional[str] = None # "agentview_image robot0_eye_in_hand_image"
    """Image observation keys to use for policy input (e.g., --image_observation_keys "robot0_eye_in_hand_image shouldercamera1_image"."""


# -------------------------------------------------
# Setup logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

# Create a named logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Evaluation helpers
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

    # Add text annotation to the frame
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Prepare episode status text
    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)

    # Add text annotations
    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)

    # Convert back to numpy array
    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: PreTrainedPolicy,
    env,
    save_dir: Path,
    step: int,
    num_episodes: int,
    task: str,
):
    """Run *num_episodes* episodes with *policy* in vectorized *env* and compute success-rate.

    Captures a video, writes it to *save_dir*/`eval_step_<step>.mp4`, and returns `(success_rate, video_path)`.
    """

    assert isinstance(env, VectorizedEnvWrapper)

    save_dir.mkdir(parents=True, exist_ok=True)

    policy_was_training = policy.training
    policy.eval()

    # Get environment info
    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")

    successes = 0
    done_episodes = 0
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

    # Create video writer at the beginning
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use the provided save_dir for videos
    save_dir.mkdir(parents=True, exist_ok=True)
    video_path = save_dir / f"eval_step_{step}_{now}.mp4"

    video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    obs, _ = env.reset()
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    # obs["task"] = task  # type: ignore

    # Reset policy for all environments
    policy.reset()

    while done_episodes < num_episodes:
        # Run episodes in parallel until we complete the required number

        with torch.inference_mode():
            # Convert numpy observations to PyTorch tensors for the policy
            action, _ = policy.select_action(obs)
            # print("action: ", action)

        obs, reward, terminated, truncated, info = env.step(action)
        # obs["task"] = task  # type: ignore
 
        frames = env.render()
        for env_idx in range(num_parallel_envs):
            episode_frames[env_idx].append(frames[env_idx])
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs

        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]

            # Reset policy hidden state for the terminated envs.
            policy.reset(env_ids=terminated_envs)
            # Annotate and write frames for completed episodes immediately
            for env_idx_tensor in terminated_envs:
                env_idx = int(env_idx_tensor.item())
                is_success = env_idx_tensor in success_envs
                done_episodes += 1
                successes += int(is_success)

                # Annotate each frame in this episode and write to video
                for step_idx, frame in enumerate(episode_frames[env_idx]):
                    annotated_frame = _annotate_frame(
                        frame=frame,
                        env_idx=env_idx,
                        episode_num=done_episodes,
                        total_episodes=num_episodes,
                        episode_step=step_idx + 1,
                        is_success=is_success,
                        font=font,
                    )
                    # Write frame immediately to video
                    video_writer.append_data(annotated_frame)

                # Reset for next episode
                episode_frames[env_idx] = []
                episode_steps[env_idx] = 0

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {done_episodes}, successes: {successes}, "
                f"FPS: {total_steps / (time.perf_counter() - start_time):.1f}"
            )

    video_writer.close()

    success_rate = successes / done_episodes if done_episodes > 0 else 0.0

    if policy_was_training:
        policy.train()

    # Calculate final performance metrics
    total_elapsed_time = time.perf_counter() - start_time
    final_fps = total_steps / total_elapsed_time if total_elapsed_time > 0 else 0.0
    episodes_per_sec = done_episodes / total_elapsed_time if total_elapsed_time > 0 else 0.0

    logger.info(f"Evaluation completed: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%) for task **{task}**")
    logger.info(f"Performance: {total_steps} total steps in {total_elapsed_time:.1f}s")
    logger.info(f"Average FPS: {final_fps:.1f} frames/sec | Episodes/sec: {episodes_per_sec:.2f}")
    logger.info(
        f"Parallel efficiency: {num_parallel_envs} environments, {final_fps / num_parallel_envs:.1f} frames/sec per environment"
    )
    logger.info(f"Video saved with annotated frames: {video_path}")

    return success_rate, video_path, final_fps


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def save_checkpoint(checkpoint_dir: Path, step: int, policy: PreTrainedPolicy, optimizer):
    """Save checkpoint with model and optimizer state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save policy
    policy.save_pretrained(str(checkpoint_dir / "policy"))

    # Save optimizer state
    checkpoint_data = {
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Save EMA state if available
    if hasattr(policy, 'ema_model') and policy.ema_model is not None:
        checkpoint_data['ema_state_dict'] = policy.ema_model.state_dict()

    torch.save(checkpoint_data, checkpoint_dir / "optimizer.pt")


def load_checkpoint(checkpoint_dir: Path, policy: PreTrainedPolicy, optimizer, load_ema: bool = False, device: str = "cuda"):
    """Load checkpoint with model and optimizer state.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        policy: Policy to load weights into
        optimizer: Optimizer to load state into
        load_ema: If True and EMA weights are available, load EMA weights into the model
        device: Device to load optimizer state to
    """
    # Load policy
    policy_path = checkpoint_dir / "policy"
    if policy_path.exists():
        # The policy's from_pretrained method will handle loading
        state_dict = load_file(policy_path / "model.safetensors", device='cpu')
        policy.load_state_dict(state_dict)

    # Load optimizer state and EMA if available
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        checkpoint = torch.load(optimizer_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state to the correct device
        # This is crucial to avoid device mismatch errors when resuming training
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # Load EMA state if available and policy has EMA
        if hasattr(policy, 'ema_model') and policy.ema_model is not None:
            if 'ema_state_dict' in checkpoint:
                policy.ema_model.load_state_dict(checkpoint['ema_state_dict'])
                logger.info("Loaded EMA state from checkpoint")

                # If load_ema flag is set, copy EMA weights to model
                if load_ema:
                    policy.ema_model.copy_to(policy.model.parameters())
                    logger.info("Copied EMA weights to model for inference")

        return checkpoint['step'] + 1, policy, optimizer

    return 0, policy, optimizer


def download_checkpoint_from_wandb(run_id: str, project: str, entity: str, artifact_alias: str = "latest", download_dir: Path = Path("./downloaded_checkpoints")) -> Path:
    """Download a checkpoint artifact from wandb.

    Args:
        run_id: WandB run ID to download from
        project: WandB project name
        entity: WandB entity name
        artifact_alias: Artifact alias to download (e.g., "latest", "best")
        download_dir: Local directory to download checkpoint to

    Returns:
        Path to the downloaded checkpoint directory
    """
    logger.info(colored(f"Downloading checkpoint from W&B run {run_id} (artifact: {artifact_alias})...", "cyan"))

    # Get the wandb API
    api = wandb.Api()

    # Find the run
    run = api.run(f"{entity}/{project}/{run_id}")

    logger.info(colored(f"Fetching artifacts from run {run_id}...", "cyan"))

    # IMPORTANT: Get artifacts FROM THE SPECIFIC RUN, not from the project level
    # This ensures we don't accidentally download checkpoints from a different run
    artifacts = run.logged_artifacts()

    if not artifacts:
        raise ValueError(f"No artifacts found for run {run_id}")

    logger.info(colored(f"Found {len(artifacts)} total artifacts in run {run_id}", "cyan"))

    # Filter for checkpoint artifacts only
    checkpoint_artifacts = [a for a in artifacts if "checkpoint" in a.name]

    if not checkpoint_artifacts:
        raise ValueError(f"No checkpoint artifacts found for run {run_id}")

    logger.info(colored(f"Found {len(checkpoint_artifacts)} checkpoint artifacts", "cyan"))

    # Helper function to extract step number from artifact
    def get_step_from_artifact(artifact):
        """Extract step number from artifact metadata or name."""
        # Try metadata first
        if hasattr(artifact, 'metadata') and artifact.metadata and 'step' in artifact.metadata:
            return artifact.metadata['step']
        # Try to extract from name (e.g., "checkpoint_step_3000")
        import re
        match = re.search(r'step_(\d+)', artifact.name)
        if match:
            return int(match.group(1))
        return 0  # Default to 0 if we can't find step number

    # Strategy to find the right artifact based on alias
    artifact = None

    if artifact_alias == "latest":
        # For "latest", get the checkpoint with the highest step number
        checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
        artifact = checkpoint_artifacts[0]
        logger.info(colored(f"Looking for latest checkpoint...", "cyan"))

    elif artifact_alias == "best":
        # For "best", look for artifacts with "best" in the name or alias
        best_artifacts = [a for a in checkpoint_artifacts if "best" in a.name.lower()]
        if best_artifacts:
            artifact = best_artifacts[0]
            logger.info(colored(f"Found 'best' checkpoint", "cyan"))
        else:
            logger.warning(colored(f"No 'best' checkpoint found, falling back to latest", "yellow"))
            checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
            artifact = checkpoint_artifacts[0]
    else:
        # For other aliases, try to find by name first, then fall back to latest
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

    # Show all available checkpoints for debugging
    checkpoint_artifacts.sort(key=get_step_from_artifact, reverse=True)
    for i, a in enumerate(checkpoint_artifacts[:10]):  # Show top 10
        step = get_step_from_artifact(a)
        selected_marker = "✓" if a == artifact else " "
        logger.info(colored(f"  {selected_marker} {i+1}. {a.name} (step {step})", "cyan"))

    # Download the artifact
    download_path = download_dir / f"{run_id}_{artifact_alias}"
    artifact_dir = artifact.download(root=str(download_path))

    logger.info(colored(f"Checkpoint downloaded to: {artifact_dir}", "green"))
    return Path(artifact_dir)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(cfg: TrainFlowBCConfig):

    # Quite hacky, but idk to use Tyro
    cfg.image_observation_keys = cfg.image_observation_keys.split(" ") if cfg.image_observation_keys is not None else None # Make a list of strings

    print("MLP dims: ", cfg.mlp_dims, "its type: ", type(cfg.mlp_dims))

    # ---------------------------------------------------------------------
    # DDP Setup
    # ---------------------------------------------------------------------
    is_ddp = False  # cfg.is_ddp
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", rank=local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[{rank}/{world_size}] Using device: {device}")
        # Suppress logging on non-master nodes
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
    else:
        rank = 0
        world_size = 1
        device = torch.device(cfg.device)


    # ---------------------------------------------------------------------
    # Logging / device setup
    # ---------------------------------------------------------------------
    logger.info(colored(f"[{rank}/{world_size}][rank/world_size] Using device: {device}", "green"))

    # Create run directory structure to organize all outputs
    from datetime import datetime

    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up output directory - organize everything under a single run directory
    if cfg.output_dir is None:
        # Auto-generate run directory
        run_dir = Path("runs") / f"{cfg.experiment}_{run_start_time}"
    else:
        run_dir = Path(cfg.output_dir)

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(colored(f"Run directory: {run_dir}", "green"))

    if cfg.seed is not None:
        set_seed(cfg.seed + rank)
        if rank == 0:
            logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))

    # ---------------------------------------------------------------------
    # Dataset (metadata first, then actual dataset with resolved timestamps)
    # ---------------------------------------------------------------------
    # In DDP, only rank 0 should download. Others should wait for it to finish.
    if is_ddp and rank != 0:
        dist.barrier()

    if rank == 0:
        logger.info("Fetching dataset metadata from the Hub…")
    ds_meta = LeRobotDatasetMetadata(cfg.dataset)

    # ---------------------------------------------------------------------
    # Build the policy configuration, applying any CLI-specified overrides
    # ---------------------------------------------------------------------

    # Build the policy config and set the target device.  When resuming from
    # checkpoints the device string can include an explicit index (e.g.
    # "cuda:0").  The underlying config validator only understands the bare
    # device types ("cuda", "cpu", "mps"), so we drop any optional suffix
    # before passing it on.
    

    if cfg.policy == "flowmatching":

        horizon = getattr(cfg, 'horizon', 16)
        n_action_steps = getattr(cfg, 'n_action_steps', 8)
        # Make sure n_action_steps is less than or equal to horizon.
        # assert n_action_steps <= horizon, f"n_action_steps ({n_action_steps}) must be less than or equal to horizon ({horizon})"
        if n_action_steps > horizon:
            logger.warning(f"n_action_steps ({n_action_steps}) is greater than horizon ({horizon}), setting n_action_steps to horizon ({horizon})")
            n_action_steps = horizon
        sampling_steps = getattr(cfg, 'sampling_steps', 10)
        vision_backbone = getattr(cfg, 'vision_backbone', "resnet18")
        pretrained_backbone_weights = f"ResNet{vision_backbone.replace('resnet', '')}_Weights.IMAGENET1K_V1"
        ema_power = getattr(cfg, 'ema_power', 0.0)
        learning_rate = getattr(cfg, 'learning_rate', 0.0001)
        lr_backbone = getattr(cfg, 'lr_backbone', 1e-05)
        weight_decay = getattr(cfg, 'weight_decay', 1e-06)
        flow_network_output_param = getattr(cfg, 'flow_network_output_param', "u")  # or "u" for velocity
        cfm_loss_mode = getattr(cfg, 'cfm_loss_mode', "u")  # or "u" or "eps"
        transported_clip_value = getattr(cfg, 'transported_clip_value', None)
        cfm_loss_use_huber = getattr(cfg, 'cfm_loss_use_huber', False)
        cfm_loss_huber_delta = getattr(cfg, 'cfm_loss_huber_delta', 0.5)
        network_architecture = getattr(cfg, 'network_architecture', "unet")
        mlp_dims = getattr(cfg, 'mlp_dims', None)

        # Create model configuration
        policy_cfg = FlowMatchingConfig(
            horizon=horizon,
            n_action_steps=n_action_steps,
            sampling_steps=sampling_steps,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            ema_power=ema_power,
            optimizer_lr=learning_rate,
            optimizer_lr_backbone=lr_backbone,
            optimizer_weight_decay=weight_decay,
            flow_network_output_param=flow_network_output_param,
            cfm_loss_mode=cfm_loss_mode,
            transported_clip_value=transported_clip_value,
            cfm_loss_use_huber=cfm_loss_use_huber,
            cfm_loss_huber_delta=cfm_loss_huber_delta,
            network_architecture=network_architecture,
            mlp_dims=mlp_dims if mlp_dims is not None else [512, 512, 512],
        )

    else:
        raise ValueError(f"Invalid policy: {cfg.policy}")

    if isinstance(cfg.device, str):
        # e.g. "cuda:0" -> "cuda"
        policy_cfg.device = cfg.device.split(":", 1)[0]
    else:
        # Fall back to original value if somehow not a string
        policy_cfg.device = cfg.device

    # Determine delta-timestamps from policy indices & dataset fps
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    logger.info("Building LeRobotDataset with inferred delta-timestamps…")

    # If using CLIP as vision backbone, add resizing to 224x224 on top of default transforms
    image_transforms_config = ImageTransformsConfig(enable=True)

    # Enable geometric augmentations (rotation, translation, scale)
    # Set weight to 1.0 to enable, 0.0 to disable
    # Note: Geometric augmentations are disabled by default
    # You can also customize the parameters in transforms.py or directly in pretrain_flow_bc.py:
    # image_transforms_config.tfs["affine"].kwargs["degrees"] = (-20, 20)  # More rotation
    # image_transforms_config.tfs["affine"].kwargs["scale"] = (0.8, 1.2)  # More scale variation
    if getattr(cfg, 'enable_geometric_augmentations', False):
        image_transforms_config.tfs["rotation"].weight = 1.0
        image_transforms_config.tfs["affine"].weight = 1.0  # Combined rotation, translation, scale
        image_transforms_config.tfs["perspective"].weight = 1.0
    else:
        image_transforms_config.tfs["rotation"].weight = 0.0
        image_transforms_config.tfs["affine"].weight = 0.0  # Combined rotation, translation, scale
        image_transforms_config.tfs["perspective"].weight = 0.0

    # Add resize to 224x224 if using CLIP as vision backbone
    if getattr(policy_cfg, "vision_backbone", None) == "clip":
        # Compose resize with the default transforms
        image_transforms = Compose(
            [
                Resize((224, 224)),
                ImageTransforms(image_transforms_config),
            ]
        )
    else:
        image_transforms = ImageTransforms(image_transforms_config)

    # Determine which episodes to load
    episodes_to_load = None
    if cfg.max_num_episodes is not None:
        # Load only first N episodes
        episodes_to_load = list(range(cfg.max_num_episodes))
        if rank == 0:
            logger.info(colored(f"Loading first {cfg.max_num_episodes} episodes only (max_num_episodes set)", "yellow"))

    dataset = LeRobotDataset(
        cfg.dataset,
        delta_timestamps=delta_timestamps,
        download_videos=True,
        image_transforms=image_transforms,
        episodes=episodes_to_load,  # Filter episodes
        # video_backend='pyav'
    )

    # Log dataset info
    if rank == 0:
        logger.info(colored(f"Dataset loaded: {dataset.num_episodes} episodes, {len(dataset)} frames", "green"))
        if cfg.max_num_episodes is not None and dataset.num_episodes < cfg.max_num_episodes:
            logger.warning(colored(f"Requested {cfg.max_num_episodes} episodes but only {dataset.num_episodes} available", "yellow"))

    # In DDP, rank 0 signals that the download is complete, and other ranks can proceed.
    if is_ddp and rank == 0:
        dist.barrier()

    # ---------------------------------------------------------------------
    # Dataloader
    # ---------------------------------------------------------------------
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True) if is_ddp else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    dl_iter = cycle(dataloader)

    # ---------------------------------------------------------------------
    # Determine if we're resuming from a checkpoint and load config if so
    # ---------------------------------------------------------------------
    resume_checkpoint_path = None

    # Check if we're resuming from wandb or local checkpoint
    if cfg.resume_run_id is not None:
        # Download checkpoint from wandb artifact
        if rank == 0:
            if cfg.resume_wandb_run:
                logger.info(colored(f"Downloading checkpoint from W&B run {cfg.resume_run_id} (will resume wandb run)", "cyan"))
            else:
                logger.info(colored(f"Downloading checkpoint from W&B run {cfg.resume_run_id} (will start NEW wandb run)", "cyan"))

            try:
                # Validate that project and entity are provided
                if cfg.wandb_project is None:
                    raise ValueError("--wandb_project is required when resuming from W&B")

                # Download checkpoint from wandb before initializing wandb
                resume_checkpoint_path = download_checkpoint_from_wandb(
                    run_id=cfg.resume_run_id,
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    artifact_alias=cfg.checkpoint_step,
                    download_dir=run_dir / "downloaded_checkpoints"
                )
                logger.info(colored(f"✓ Downloaded checkpoint from W&B", "green"))
            except Exception as e:
                logger.error(colored(f"Failed to download checkpoint from W&B: {e}", "red"))
                logger.warning(colored("Starting from scratch", "yellow"))
                resume_checkpoint_path = None

    elif cfg.resume_ckpt is not None:
        # Resume from local checkpoint path
        if rank == 0:
            logger.info(colored(f"Resuming from local checkpoint: {cfg.resume_ckpt}", "cyan"))

        checkpoint_path = Path(cfg.resume_ckpt)
        if checkpoint_path.exists():
            resume_checkpoint_path = checkpoint_path
            logger.info(colored(f"Found local checkpoint", "cyan"))
        else:
            if rank == 0:
                logger.warning(colored(f"Checkpoint not found at {cfg.resume_ckpt}, starting from scratch", "yellow"))
            resume_checkpoint_path = None

    # ---------------------------------------------------------------------
    # Policy configuration: Load from checkpoint or create new
    # ---------------------------------------------------------------------
    if cfg.policy == "flowmatching":
        # If resuming from checkpoint, load the config from checkpoint
        if resume_checkpoint_path is not None:
            logger.info(colored("Loading policy config from checkpoint...", "cyan"))

            config_path = resume_checkpoint_path / "policy" / "config.json"
            if not config_path.exists():
                raise ValueError(f"Config file not found in checkpoint: {config_path}")

            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                config_dict.pop('type', None)
                config_dict.pop('normalization_mapping', None)
                policy_cfg = FlowMatchingConfig(**config_dict)

            logger.info(colored("Loaded policy config from checkpoint", "green"))

            # Use image features from loaded config
            if not hasattr(policy_cfg, '_image_features') or not policy_cfg._image_features:
                # Set from input_features if not already set
                policy_cfg.image_features = [k for k in policy_cfg.input_features if "image" in k]

            # Update the config's image observation keys using the loaded config's image features
            policy_cfg.image_observation_keys = [k.replace("observation.images.", "") for k in policy_cfg.image_features]

            logger.info(f"Using image features from checkpoint config: {policy_cfg.image_features}")

            # Set state features if not already set
            if not hasattr(policy_cfg, '_state_features') or not policy_cfg._state_features:
                policy_cfg.state_features = [k for k in policy_cfg.input_features if "state" in k or "pos" in k]
            logger.info(f"State features: {policy_cfg.state_features}")

        else:
            # Create new config from scratch
            logger.info(colored("Creating new policy config...", "cyan"))

            # Configure input/output features
            policy_cfg.input_features = list(dataset.features.keys())
            policy_cfg.output_features = ["action"]
            policy_cfg.input_shapes = ds_meta.shapes
            policy_cfg.output_shapes = {"action": ds_meta.shapes["action"]}

            # Determine image and state features
            if cfg.image_observation_keys is not None:
                # Use custom image observation keys (convert to policy format: observation.images.{key})
                policy_cfg.image_features = [
                    f"observation.images.{key.replace('_image', '')}" for key in cfg.image_observation_keys
                ]
                print(f"policy_cfg.input_features: {policy_cfg.input_features}")
                print(f"cfg.image_observation_keys: {cfg.image_observation_keys}")
                # Pop up image features that is not in use from policy_cfg.input_features
                new_input_features = []
                for policy_input_feature in policy_cfg.input_features:
                    if "images" not in policy_input_feature:
                        new_input_features.append(policy_input_feature)
                    elif "images" in policy_input_feature and policy_input_feature in policy_cfg.image_features:
                        new_input_features.append(policy_input_feature)
                policy_cfg.input_features = new_input_features
                logger.info(f"Using custom image observation keys: {policy_cfg.image_features}")
            else:
                # Use all image features from dataset
                policy_cfg.image_features = [k for k in policy_cfg.input_features if "image" in k]
                logger.info(f"Using all image features from dataset: {policy_cfg.image_features}")

            policy_cfg.state_features = [k for k in policy_cfg.input_features if "state" in k or "pos" in k]

        # Create model
        logger.info("Creating FlowMatching model...")
        policy = FlowMatchingPolicy(policy_cfg, dataset_stats=ds_meta.stats)

    else:
        raise ValueError(f"Invalid policy: {cfg.policy}")

    # Learning-rate & weight-decay fallbacks
    lr_default = getattr(policy_cfg, "optimizer_lr", 1e-4)
    wd_default = getattr(policy_cfg, "optimizer_weight_decay", 0.0)

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=lr_default, weight_decay=wd_default)

    # ---------------------------------------------------------------------
    # Load checkpoint weights if resuming
    # ---------------------------------------------------------------------
    start_step = 0

    if resume_checkpoint_path is not None:
        logger.info(colored(f"Loading checkpoint weights from {resume_checkpoint_path}...", "cyan"))
        start_step, policy, optimizer = load_checkpoint(resume_checkpoint_path, policy, optimizer, load_ema=cfg.load_ema, device=device)
        logger.info(colored(f"✓ Loaded checkpoint from step {start_step}", "green"))

    policy.to(device)
    if getattr(policy_cfg, "ema_power", 0.0) > 0:
        policy.ema_model.to(device)

    if is_ddp:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)
        # When using gradient checkpointing with DDP, we need to specify that
        # the model graph is static, otherwise we will get an error that a
        # variable has been marked as ready twice. This is a workaround for a
        # known issue in PyTorch.
        # See: https://github.com/pytorch/pytorch/issues/43259
        if cfg.policy in ["pi0", "pi0fast", "mypi0fast", "mypi0fast_chunked"]:
            policy._set_static_graph()

    policy.train()

    # ---------------------------------------------------------------------
    # Optional WandB
    # ---------------------------------------------------------------------
    if cfg.wandb_enable and rank == 0:
        if cfg.wandb_project is None:
            raise ValueError("--wandb_project is required when --wandb_enable is set")

        # Determine if we should resume the wandb run itself
        # Only resume wandb run if explicitly requested AND we have a run_id
        wandb_run_id = None
        wandb_resume_mode = None

        if cfg.resume_wandb_run and cfg.resume_run_id:
            wandb_run_id = cfg.resume_run_id
            wandb_resume_mode = "must"
            logger.info(colored(f"Resuming wandb run: {wandb_run_id}", "cyan"))
        else:
            logger.info(colored("Starting new wandb run", "cyan"))

        # Simple config logging - just the essentials
        wandb_config = {
            **vars(cfg),
            "policy_config": asdict(policy_cfg) if is_dataclass(policy_cfg) else policy_cfg.__dict__,
            "dataset": cfg.dataset,
            "total_episodes": ds_meta.total_episodes,
            "total_frames": ds_meta.total_frames,
        }

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=wandb_config,
            name=f"{cfg.experiment}_{cfg.policy}_{Path(cfg.dataset).name}",
            id=wandb_run_id,
            resume=wandb_resume_mode,
            dir=str(run_dir),  # Save wandb logs in the run directory
            settings=wandb.Settings(),
        )
        logger.info(colored(f"W&B logging enabled (logs saved to {run_dir / 'wandb'})", "blue"))

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    # Checkpoints will be saved under run_dir/checkpoints
    checkpoints_dir = run_dir / "checkpoints" if rank == 0 else None
    if rank == 0:
        assert checkpoints_dir is not None
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    step = start_step
    eval_env = None  # type: ignore  # will hold the evaluation environment if created

    # Track the best evaluation success-rate achieved so far (used for early-stopping style checkpointing)
    best_success_rate = 0.0

    if cfg.rollout_freq and cfg.eval_env:
        # ------------------------------------------------------------------
        # Create the DexMimicGen evaluation environment
        # ------------------------------------------------------------------
        device_str = "cpu" if cfg.device == "cpu" else "cuda"

        # Check if this is a DexMimicGen environment
        dexmimicgen_envs = [
            "TwoArmCoffee",
            "TwoArmThreading",
            "TwoArmThreePieceAssembly",
            "TwoArmTransport",
            "TwoArmLiftTray",
            "TwoArmBoxCleanup",
            "TwoArmDrawerCleanup",
            "TwoArmPouring",
            "TwoArmCanSortRandom",
        ]
        robomimic_envs = [
            "Lift",
            "Can",
            "Square",
        ]
        # alias_map = {
        #     # Robomimic papers / datasets refer to these tasks without the
        #     # full robosuite class name. We translate them here so that
        #     # callers can simply pass "Lift", "Can", "Square", or
        #     # "Transport" and things will work out of the box.
        #     "Can": "PickPlaceCan",
        #     "Square": "NutAssemblySquare",
        #     "Transport": "TwoArmTransport",
        # }

        envs = dexmimicgen_envs + robomimic_envs

        if cfg.eval_env in envs:
            # Create DexMimicGen environment (vectorized or single based on debug flag)
            if rank == 0:

                logger.info(f"Creating DexMimicGen evaluation environment: {cfg.eval_env}")

                if cfg.debug:
                    logger.info(f"Expected image keys: {cfg.image_observation_keys}, its type: {type(cfg.image_observation_keys)}")
                    # Debug mode: use synchronous vectorized environment for easier debugging
                    logger.info("Debug mode enabled: using synchronous vectorized environment")
                    eval_env = create_vectorized_env(
                        env_name=cfg.eval_env,
                        num_envs=cfg.eval_num_envs,
                        device=device_str,
                        camera_size=cfg.eval_camera_size,
                        # video_key="robot0_eye_in_hand",
                        video_key="agentview",
                        debug=True,
                        expected_image_keys=cfg.image_observation_keys,
                    )
                else:
                    logger.info(f"Expected image keys: {cfg.image_observation_keys}, its type: {type(cfg.image_observation_keys)}")
                    # Production mode: use asynchronous multiprocessing environment for speed
                    logger.info("Production mode: using asynchronous multiprocessing environment")
                    eval_env = create_vectorized_env(
                        env_name=cfg.eval_env,
                        num_envs=cfg.eval_num_envs,
                        device=device_str,
                        camera_size=cfg.eval_camera_size,
                        # video_key="robot0_eye_in_hand",
                        video_key="agentview",
                        debug=False,
                        expected_image_keys=cfg.image_observation_keys,
                    )
        elif rank == 0:
            raise ValueError(
                f"Unknown environment: {cfg.eval_env}. This script only supports DexMimicGen environments."
            )

    while step < cfg.steps:
        # Accumulate loss over all micro-batches for logging
        total_loss_for_logging = torch.tensor(0.0, device=device)
        iter_start_t = time.perf_counter()

        # Inner loop for gradient accumulation
        for i in range(cfg.gradient_accumulation_steps):
            if is_ddp:
                # Set epoch to ensure different shuffling for each micro-batch
                if sampler is not None:
                    sampler.set_epoch(step * cfg.gradient_accumulation_steps + i)

            data_t0 = time.perf_counter()
            batch: dict[str, Any] = next(dl_iter)
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device, non_blocking=True)

            data_load_ms = (time.perf_counter() - data_t0) * 1000

            # HACK: Rename 'state' to 'observation.state' for LIBERO compatibility
            if "state" in batch and "observation.state" not in batch:
                batch["observation.state"] = batch.pop("state")
            # HACK: Rename 'actions' to 'action' for LIBERO compatibility
            if "actions" in batch and "action" not in batch:
                batch["action"] = batch.pop("actions")

            # Save sample images for inspection (only once at step 0)
            if cfg.debug and step == 0 and rank == 0:
                save_dir = run_dir / "debug_images"
                save_dir.mkdir(exist_ok=True, parents=True)

                # You can do the same for other cameras
                for key in batch.keys():
                    if "observation.images" in key:
                        imgs = batch[key]
                        logger.info(f"{key} shape: {imgs.shape}, dtype: {imgs.dtype}, range: [{imgs.min():.3f}, {imgs.max():.3f}]")

                        # Save first 8 images from the batch as a grid
                        num_to_save = min(8, imgs.shape[0])
                        # Assuming images are in [B, C, H, W] format and normalized to [0, 1] or [-1, 1]
                        # If they're in [0, 255], normalize them first
                        if imgs.max() > 1.0:
                            imgs_norm = imgs[:num_to_save] / 255.0
                        else:
                            imgs_norm = imgs[:num_to_save]

                        grid = vutils.make_grid(imgs_norm, nrow=4, normalize=True, scale_each=True)
                        vutils.save_image(grid, save_dir / f"{key.replace('observation.images.', '')}_batch.png")
                        logger.info(f"Saved sample images to {save_dir}" + f"/{key.replace('observation.images.', '')}_batch.png")

            # DDP: only sync gradients on the last accumulation step
            is_last_accumulation_step = (i + 1) == cfg.gradient_accumulation_steps
            if is_ddp and not is_last_accumulation_step:
                with policy.no_sync():  # type: ignore
                    loss, loss_dict = policy.get_cfm_loss(batch)  # type: ignore

                    # Average loss over accumulation steps
                    loss = loss / cfg.gradient_accumulation_steps
                    loss.backward()
            else:
                loss, loss_dict = policy.get_cfm_loss(batch)  # type: ignore
                # Average loss over accumulation steps
                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()

            # Accumulate loss for logging, detached from graph
            total_loss_for_logging += loss.detach()

        # Clip gradients and step optimizer
        if is_ddp:
            dist.all_reduce(total_loss_for_logging, op=dist.ReduceOp.AVG)

        # Compute gradient norm before clipping (for logging)
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update EMA if enabled
        model_to_update_ema = policy.module if is_ddp else policy
        if isinstance(model_to_update_ema, PreTrainedPolicy):
            model_to_update_ema.step_ema()

        iter_ms = (time.perf_counter() - iter_start_t) * 1000

        # ------------------------------------------------------------------
        # Logging ----------------------------------------------------------
        # ------------------------------------------------------------------
        loss_val = total_loss_for_logging.item()

        # Compute fractional epoch (how many full passes over the dataset)
        samples_seen = (step + 1) * cfg.batch_size * cfg.gradient_accumulation_steps
        fractional_epoch = samples_seen / len(dataset) if len(dataset) > 0 else 0.0

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        if rank == 0 and step % cfg.log_freq == 0:
            msg = (
                f"[step {step:>6d}/{cfg.steps}]"
                f" loss: {loss_val:.4f}"
                f" | lr: {current_lr:.2e}"
                f" | grad_norm: {grad_norm_before_clip:.4f}"
                f" | epoch: {fractional_epoch:.2f}"
                f" | data: {data_load_ms:.1f} ms"
                f" | iter: {iter_ms:.1f} ms"
            )
            logger.info(msg)
            if cfg.wandb_enable:
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/loss_dict": loss_dict,
                        "train/learning_rate": current_lr,
                        "train/grad_norm_before_clip": grad_norm_before_clip.item() if torch.is_tensor(grad_norm_before_clip) else grad_norm_before_clip,
                        "train/epoch": fractional_epoch,
                        "time/data_load_ms": data_load_ms,
                        "time/iter_ms": iter_ms,
                    },
                    step=step,
                )

        # Checkpointing ----------------------------------------------
        if (step % cfg.save_freq == 0 and step != start_step) or step + 1 == cfg.steps:
            if rank == 0:
                assert checkpoints_dir is not None
                model_to_save = policy.module if is_ddp else policy
                assert isinstance(model_to_save, PreTrainedPolicy)

                # Save checkpoint to checkpoints directory
                ckpt_dir = checkpoints_dir / f"step_{step}"
                save_checkpoint(ckpt_dir, step, model_to_save, optimizer)

                # Also save a "latest" checkpoint for easy resuming
                latest_dir = checkpoints_dir / "latest"
                if latest_dir.exists():
                    import shutil
                    shutil.rmtree(latest_dir)
                save_checkpoint(latest_dir, step, model_to_save, optimizer)

                logger.info(colored(f"Checkpoint saved @ {ckpt_dir}", "magenta"))

                # If wandb is enabled, upload checkpoints using artifacts
                if cfg.wandb_enable and wandb.run is not None:
                    try:
                        # Create and log checkpoint artifact
                        checkpoint_artifact = wandb.Artifact(
                            name=f"checkpoint_step_{step}",
                            type="model",
                            description=f"Model checkpoint at training step {step}",
                            metadata={
                                "step": step,
                                "loss": loss_val,
                            }
                        )
                        checkpoint_artifact.add_dir(str(ckpt_dir))
                        wandb.log_artifact(checkpoint_artifact)

                        # Also create a "latest" artifact for easy access
                        latest_artifact = wandb.Artifact(
                            name="checkpoint_latest",
                            type="model",
                            description=f"Latest model checkpoint (step {step})",
                            metadata={
                                "step": step,
                                "loss": loss_val,
                            }
                        )
                        latest_artifact.add_dir(str(latest_dir))
                        wandb.log_artifact(latest_artifact, aliases=["latest"])

                        logger.info(colored(f"Checkpoints uploaded to W&B as artifacts", "magenta"))
                    except Exception as e:
                        logger.warning(colored(f"Failed to upload checkpoint artifacts to W&B: {e}", "yellow"))

        step += 1

        # ------------------------------------------------------------------
        # Rollout evaluation -----------------------------------------------
        # ------------------------------------------------------------------
        if (
            rank == 0
            and cfg.rollout_freq is not None
            and cfg.eval_env is not None
            and (step % cfg.rollout_freq == 0 or step == cfg.steps or step == 1)
        ):
            rollout_t0 = time.perf_counter()

            model_to_eval = policy.module if is_ddp else policy
            # copy the model
            model_to_eval = copy.deepcopy(model_to_eval)

            # Enable EMA for evaluation if configured
            if isinstance(model_to_eval, PreTrainedPolicy):
                model_to_eval.enable_ema()

            model_to_eval.init_action_buffers(cfg.eval_num_envs)

            assert isinstance(model_to_eval, PreTrainedPolicy)
            assert eval_env is not None
            success_rate, video_path, final_fps = _run_rollouts(
                policy=model_to_eval,
                env=eval_env,
                save_dir=run_dir / "videos",
                step=step,
                num_episodes=cfg.eval_num_episodes,
                task=cfg.eval_env,
            )

            # Disable EMA after evaluation (not needed since we deepcopied, but good practice)
            if isinstance(model_to_eval, PreTrainedPolicy):
                model_to_eval.disable_ema()

            optimizer.zero_grad()


            rollout_ms = (time.perf_counter() - rollout_t0) * 1000

            logger.info(
                colored(
                    f"[step {step:>6d}] eval success-rate: {success_rate * 100:.1f}% | rollout: {rollout_ms / 1000:.2f} s | {final_fps:.1f} fps",
                    "cyan",
                )
            )

            if cfg.wandb_enable:
                # Prepare log data
                log_data = {
                    "eval/success_rate": success_rate,
                    "time/rollout_ms": rollout_ms,
                }

                # Upload video as artifact and add to log data
                if video_path is not None and video_path.exists():
                    try:
                        # Upload video as artifact for versioned storage
                        video_artifact = wandb.Artifact(
                            name=f"eval_video_step_{step}",
                            type="video",
                            description=f"Evaluation rollout video at step {step} (success rate: {success_rate * 100:.1f}%)",
                            metadata={
                                "step": step,
                                "success_rate": success_rate,
                                "num_episodes": cfg.eval_num_episodes,
                                "env_name": cfg.eval_env,
                                "fps": final_fps,
                            }
                        )
                        video_artifact.add_file(str(video_path))
                        wandb.log_artifact(video_artifact)

                        # Add video for UI visualization to the same log call
                        log_data["eval/rollout_video"] = wandb.Video(str(video_path), format="mp4")

                        logger.info(colored(f"Video uploaded to W&B (artifact + UI)", "cyan"))
                    except Exception as e:
                        logger.warning(colored(f"Failed to upload video to W&B: {e}", "yellow"))

                # Single wandb.log() call to avoid duplicate step logging
                wandb.log(log_data, step=step)

            # -------------------------------------------------------------
            # Checkpoint the model whenever we obtain a new best success-rate
            # -------------------------------------------------------------
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                if rank == 0:
                    logger.info(colored(f"New best success-rate! Saving checkpoint at step {step}", "magenta"))

                    assert checkpoints_dir is not None
                    model_to_save = policy.module if is_ddp else policy
                    assert isinstance(model_to_save, PreTrainedPolicy)

                    # Save best checkpoint locally
                    best_dir = checkpoints_dir / "best"
                    if best_dir.exists():
                        import shutil
                        shutil.rmtree(best_dir)
                    save_checkpoint(best_dir, step, model_to_save, optimizer)

                    # If wandb is enabled, upload best checkpoint using artifacts
                    if cfg.wandb_enable and wandb.run is not None:
                        try:
                            best_artifact = wandb.Artifact(
                                name="checkpoint_best",
                                type="model",
                                description=f"Best model checkpoint (step {step}, success rate: {success_rate * 100:.1f}%)",
                                metadata={
                                    "step": step,
                                    "success_rate": success_rate,
                                }
                            )
                            best_artifact.add_dir(str(best_dir))
                            wandb.log_artifact(best_artifact, aliases=["best"])
                            logger.info(colored(f"Best checkpoint uploaded to W&B as artifact", "magenta"))
                        except Exception as e:
                            logger.warning(colored(f"Failed to upload best checkpoint artifact to W&B: {e}", "yellow"))

        # All ranks must wait for rank 0 to finish evaluation before proceeding
        # to the next training step. Otherwise, ranks 1..N will rush ahead and
        # deadlock at the next collective operation while rank 0 is busy here.
        if is_ddp:
            dist.barrier()

    if rank == 0:
        logger.info(colored("Training finished!", "green", attrs=["bold"]))
        if cfg.wandb_enable:
            wandb.finish()

        if eval_env is not None:
            eval_env.close()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Parse arguments using tyro
    args_cli = tyro.cli(TrainFlowBCConfig, config=(tyro.conf.FlagConversionOff,))
    main(args_cli)
