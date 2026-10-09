#!/usr/bin/env python3
"""
Modified FPO play script with Viser visualization support.

This script extends the standard play.py to include real-time visualization
using Viser for headless operation.

Example usage:
    # First extract assets
    python isaac_asset_extractor.py --task Isaac-Velocity-Flat-Unitree-Go2-v0

    # Then play with Viser (with local checkpoint)
    python play_with_viser.py \
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
        --checkpoint /path/to/checkpoint \
        --headless \
        --viser \
        --asset-dir output/isaac_velocity_flat_unitree_go2_v0

    # Or download from W&B
    python play_with_viser.py \
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
        --wandb-run-path entity/project/run_id \
        --wandb-checkpoint model_7500.pt \
        --headless \
        --viser
"""

"""Launch Isaac Sim Simulator first."""

# HACK: import websockets before isaaclab stuff happens. This is necessary.
import viser

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with a trained FPO agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# Viser-specific arguments
parser.add_argument("--viser", action="store_true", default=False, help="Enable Viser visualization.")
parser.add_argument("--viser-port", type=int, default=8080, help="Port for Viser web server.")
parser.add_argument("--asset-dir", type=str, default=None, help="Directory containing pre-extracted assets.")
parser.add_argument("--viser-update-freq", type=int, default=1, help="Update Viser every N steps.")
parser.add_argument("--viser-env-spacing", type=float, default=1.5, help="Spacing between environments in regular grid visualization (default: 1.5m).")
parser.add_argument("--viser-fps", type=int, default=60, help="Target frame rate for Viser visualization (default: 60).")
parser.add_argument("--viser-random-grid-size", type=float, default=0.0, help="Size of grid for random robot offsets. Set to 0.0 to disable random offsets (default: 3x3).")

# Flow matching specific arguments
parser.add_argument("--flow-sampling-steps", type=int, default=None, help="Number of sampling steps for flow matching inference (default: use training value).")
parser.add_argument("--training-sampling-steps", type=int, default=None, help="Number of sampling steps for training (default: use training value).")

# W&B checkpoint download arguments
parser.add_argument("--wandb-run-path", type=str, default=None, help="W&B run path (e.g., entity/project/run_id)")
parser.add_argument("--wandb-checkpoint", type=str, default=None, help="W&B checkpoint file name to download (e.g., model_7500.pt)")

# append FPO cli arguments
cli_args.add_fpo_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import sys
import time
import torch
import wandb

from isaaclab_fpo.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_pickle
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_fpo.viser import ViserIsaacLab

from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401 — registers motion tracking envs
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def download_wandb_checkpoint(run_path, checkpoint_name, download_dir="checkpoints"):
    """Download a checkpoint from W&B.

    Args:
        run_path: W&B run path (e.g., "entity/project/run_id")
        checkpoint_name: Name of the checkpoint file to download (e.g., "model_7500.pt")
        download_dir: Directory to download to (default: "checkpoints")

    Returns:
        Path to the downloaded checkpoint file
    """
    # Create download directory if it doesn't exist
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)

    # Create a subdirectory for this specific run
    run_id = run_path.split("/")[-1]
    run_dir = download_path / run_id
    run_dir.mkdir(exist_ok=True)

    # Full path for the checkpoint
    checkpoint_path = run_dir / checkpoint_name

    # Download from W&B
    print(f"[INFO] Downloading checkpoint from W&B...")
    print(f"       Run path: {run_path}")
    print(f"       Checkpoint: {checkpoint_name}")
    print(f"       Destination: {checkpoint_path}")

    # Initialize W&B API (respects WANDB_BASE_URL env var for custom servers)
    api = wandb.Api()

    # Get the run
    run = api.run(run_path)

    # Download the checkpoint file
    file = run.file(checkpoint_name)
    file.download(root=str(run_dir), replace=True)

    print(f"[INFO] Downloaded checkpoint to: {checkpoint_path}")

    # Also download params files if they exist
    params_dir = run_dir / "params"
    params_dir.mkdir(exist_ok=True)

    try:
        # Try to download agent.pkl
        agent_file = run.file("params/agent.pkl")
        agent_file.download(root=str(run_dir), replace=True)
        print(f"[INFO] Downloaded agent config to: {params_dir / 'agent.pkl'}")
    except Exception as e:
        print(f"[WARNING] Could not download agent.pkl: {e}")

    try:
        # Try to download env.pkl
        env_file = run.file("params/env.pkl")
        env_file.download(root=str(run_dir), replace=True)
        print(f"[INFO] Downloaded environment config to: {params_dir / 'env.pkl'}")
    except Exception as e:
        print(f"[WARNING] Could not download env.pkl: {e}")

    return str(checkpoint_path)


def main():
    """Play with FPO agent."""
    task_name = args_cli.task.split(":")[-1]
    # First determine the checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("isaaclab_fpo", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.wandb_run_path and args_cli.wandb_checkpoint:
        resume_path = download_wandb_checkpoint(
            args_cli.wandb_run_path,
            args_cli.wandb_checkpoint
        )
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # Parse default config to get experiment name
        agent_cfg_temp = cli_args.parse_fpo_cfg(task_name, args_cli)
        log_root_path = os.path.join("logs", "isaaclab_fpo", agent_cfg_temp.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg_temp.load_run, agent_cfg_temp.load_checkpoint)

    # Try to load configs from saved params files
    log_dir = os.path.dirname(resume_path)
    agent_pkl_path = os.path.join(log_dir, "params", "agent.pkl")
    env_pkl_path = os.path.join(log_dir, "params", "env.pkl")

    # Load agent config
    if os.path.exists(agent_pkl_path):
        print(f"[INFO] Loading agent config from: {agent_pkl_path}")
        agent_cfg = load_pickle(agent_pkl_path)
    else:
        print(f"[WARNING] No saved agent config found at {agent_pkl_path}, using default config")
        agent_cfg: FpoRslRlOnPolicyRunnerCfg = cli_args.parse_fpo_cfg(task_name, args_cli)

    # Load environment config
    if os.path.exists(env_pkl_path):
        print(f"[INFO] Loading environment config from: {env_pkl_path}")
        env_cfg = load_pickle(env_pkl_path)
        print(f"[INFO] Loaded config has num_envs: {env_cfg.scene.num_envs}")
        # Override some settings for playback
        if args_cli.num_envs is not None:
            print(f"[INFO] Overriding num_envs from {env_cfg.scene.num_envs} to {args_cli.num_envs}")
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
    else:
        print(f"[WARNING] No saved environment config found at {env_pkl_path}, using default config")
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )

    print(f"[INFO] Loading experiment from directory: {log_dir}")
    print(f"[INFO] Using policy network hidden dims: {agent_cfg.policy.actor_hidden_dims}")
    print(f"[INFO] Final num_envs for simulation: {env_cfg.scene.num_envs}")

    # Override the training sampling steps if specified
    if args_cli.training_sampling_steps is not None and hasattr(agent_cfg.policy, 'training_sampling_steps'):
        print(f"[INFO] Overriding training sampling steps to {args_cli.training_sampling_steps}")
        agent_cfg.policy.training_sampling_steps = args_cli.training_sampling_steps
    elif hasattr(agent_cfg.policy, 'training_sampling_steps'):
        print(f"[INFO] Using training sampling steps: {agent_cfg.policy.training_sampling_steps}")

    # Override flow sampling steps if specified
    if args_cli.flow_sampling_steps is not None and hasattr(agent_cfg.policy, 'sampling_steps'):
        print(f"[INFO] Overriding flow sampling steps from {agent_cfg.policy.sampling_steps} to {args_cli.flow_sampling_steps}")
        agent_cfg.policy.sampling_steps = args_cli.flow_sampling_steps
    elif hasattr(agent_cfg.policy, 'sampling_steps'):
        print(f"[INFO] Using flow sampling steps: {agent_cfg.policy.sampling_steps}")

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Save reference to base environment for Viser
    base_env = env.unwrapped

    # Initialize Viser visualization
    viser_viz = None
    if args_cli.viser:
        print("\n[INFO] Initializing Viser visualization...")

        # Determine asset directory
        if args_cli.asset_dir:
            asset_dir = Path(args_cli.asset_dir)
        else:
            # Try to auto-detect based on task name
            task_clean = args_cli.task.lower().replace(":", "_").replace("-", "_")
            asset_dir = Path(__file__).resolve().parents[1] / "viser_assets" / task_clean

        if not asset_dir.exists():
            print(f"[ERROR] Asset directory not found: {asset_dir}")
            print("Please run isaac_asset_extractor.py first to extract the assets.")
            return

        try:
            # Get actual number of environments
            # You can adjust this cap based on your system's performance
            num_envs_to_viz = min(env_cfg.scene.num_envs, 256)  # Cap at 256 for performance
            if num_envs_to_viz < env_cfg.scene.num_envs:
                print(f"[WARNING] Visualizing only {num_envs_to_viz} out of {env_cfg.scene.num_envs} environments for performance")

            # Generate random offsets if grid size > 0
            random_offsets = None
            if args_cli.viser_random_grid_size > 0:
                # Generate random positions within the grid using single call
                half_size = args_cli.viser_random_grid_size / 2
                random_offsets = np.random.uniform(
                    low=[-half_size, -half_size, 0],
                    high=[half_size, half_size, 0],
                    size=(num_envs_to_viz, 3)
                )
                print(f"[INFO] Generated random offsets within {args_cli.viser_random_grid_size}x{args_cli.viser_random_grid_size} grid")
                # Pass random offsets to ViserIsaacLab
                env_spacing = 0.0  # No regular grid spacing when using random offsets
            else:
                env_spacing = args_cli.viser_env_spacing

            viser_viz = ViserIsaacLab(
                    asset_dir=asset_dir,
                    port=args_cli.viser_port,
                    update_freq=args_cli.viser_update_freq,
                    num_envs=num_envs_to_viz,
                    env_spacing=env_spacing,
                    fps=args_cli.viser_fps,
                    random_offsets=random_offsets,
                )

            # Load mapping from base environment
            viser_viz.load_from_env(base_env)

            print(f"[INFO] Viser server running at http://localhost:{args_cli.viser_port}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize Viser: {e}")
            import traceback
            traceback.print_exc()
            viser_viz = None

    # wrap around environment for FPO
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Apply the sampling steps override to the loaded policy if it was changed
    if args_cli.flow_sampling_steps is not None and hasattr(ppo_runner.alg.policy, 'sampling_steps'):
        ppo_runner.alg.policy.sampling_steps = args_cli.flow_sampling_steps

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        print(f"[INFO] Exported policy to JIT format: {os.path.join(export_model_dir, 'policy.pt')}")
    except Exception as e:
        print(f"[WARNING] Failed to export policy to JIT: {e}")

    # Try ONNX export but don't fail if it doesn't work (e.g., for flow matching policies)
    try:
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
        print(f"[INFO] Exported policy to ONNX format: {os.path.join(export_model_dir, 'policy.onnx')}")
    except Exception as e:
        print(f"[WARNING] Failed to export policy to ONNX (this is expected for flow matching policies): {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Performance tracking for Viser
    viser_update_time = 0.0
    viser_update_count = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # Handle reset request with proper mode management
        reset_requested = viser_viz is not None and viser_viz.check_reset_request()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            if reset_requested:
                print(f"[INFO] Resetting all {env.unwrapped.num_envs} environments...")
                # Use the wrapper's reset method which returns the proper observation format
                obs, _ = env.reset()
                print(f"[INFO] Reset complete")

        # Update Viser visualization
        if viser_viz is not None:

            viser_start = time.time()
            try:
                # Use the saved base environment reference and pass rewards and actions
                viser_viz.update_from_env(base_env, rewards=rewards, actions=actions)
            except Exception as e:
                print(f"[WARNING] Viser update failed: {e}")
            viser_update_time += time.time() - viser_start
            viser_update_count += 1

            # Print performance stats every 1000 steps
            if viser_update_count % 1000 == 0:
                avg_viser_time = viser_update_time / viser_update_count * 1000  # ms
                print(f"[VISER] Average update time: {avg_viser_time:.2f} ms")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # Close Viser
    if viser_viz is not None:
        viser_viz.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
