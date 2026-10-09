# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from FPO."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# local imports
from isaaclab_fpo import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Plot Flow Field and Action Correlation for a FPO++ policy.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and algorithm. If not specified, uses default from config.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

# Evaluation arguments for Motion Tracking task
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--zero_noise_sampling", action="store_true", default=False, help="Use zero noise sampling instead of random noise for flow model (overrides config)")
parser.add_argument("--training_sampling_steps", type=int, default=None, help="Number of sampling steps for training (default: use training value).")
parser.add_argument("--flow_sampling_steps", type=int, default=None, help="Number of sampling steps for flow matching inference (default: use training value).")

# W&B checkpoint download arguments
parser.add_argument("--wandb-run-path", type=str, default=None, help="W&B run path (e.g., entity/project/run_id)")
parser.add_argument("--wandb-checkpoint", type=str, default=None, help="W&B checkpoint file name to download (e.g., model_7500.pt)")

# Plotting arguments
parser.add_argument("--num_noise_samples", type=int, default=100, help="Number of noise samples to generate.")
parser.add_argument("--plot_interval", type=int, default=10, help="Interval for plotting flow field and action correlation.")
parser.add_argument(
    "--marginal_flow_minibatch_size",
    type=int,
    default=1000,
    help="Mini-batch size to use when sampling marginal flow trajectories.",
)
parser.add_argument(
    "--action_dims",
    type=int,
    nargs="+",
    default=[20],
    help="List of action dimension indices to visualize (space-separated).",
)
parser.add_argument(
    "--action_scale", type=float, default=1.0, help="Scalar to scale the final action values for visualization."
)

# append FPO cli arguments
cli_args.add_fpo_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import wandb
from pathlib import Path

from isaaclab_fpo.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_pickle

import isaaclab_tasks  # noqa: F401
import whole_body_tracking  # noqa: F401 — registers motion tracking envs
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_fpo import FpoRslRlOnPolicyRunnerCfg, FpoRslRlVecEnvWrapper

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

    print(f"[INFO] ✅ Downloaded checkpoint to: {checkpoint_path}")

    # Also download params files if they exist
    params_dir = run_dir / "params"
    params_dir.mkdir(exist_ok=True)

    try:
        # Try to download agent.pkl
        agent_file = run.file("params/agent.pkl")
        agent_file.download(root=str(run_dir), replace=True)
        print(f"[INFO] ✅ Downloaded agent config to: {params_dir / 'agent.pkl'}")
    except Exception as e:
        print(f"[WARNING] Could not download agent.pkl: {e}")

    try:
        # Try to download env.pkl
        env_file = run.file("params/env.pkl")
        env_file.download(root=str(run_dir), replace=True)
        print(f"[INFO] ✅ Downloaded environment config to: {params_dir / 'env.pkl'}")
    except Exception as e:
        print(f"[WARNING] Could not download env.pkl: {e}")

    return str(checkpoint_path)


################## Plotting functions ################## 
def visualize_action_correlation(trajectories, save_path, checkpoint_name, env_steps, action_scale=1.0):
    """
    Calculates and plots the Pearson correlation matrix of the final scaled actions.
    """
    # 1. Extract the final action tensor from each trajectory
    final_actions_list = [traj[-1][1].flatten() for traj in trajectories]
    final_actions = np.array(final_actions_list)  # Shape: (num_samples, num_actions)

    # 2. Scale the final actions (important for consistency, though correlation is scale-invariant)
    scaled_final_actions = final_actions * action_scale
    num_actions = scaled_final_actions.shape[1]

    # 3. Compute the Pearson correlation matrix
    # 'rowvar=False' treats each column as a separate variable (action dimension)
    corr_matrix = np.corrcoef(scaled_final_actions, rowvar=False)

    # 4. Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Correlation ranges from -1 to 1, so set vmin/vmax accordingly
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # 5. Add a colorbar and labels
    fig.colorbar(im, ax=ax) #, label="Pearson Correlation Coefficient")
    ax.set_xticks(np.arange(num_actions))
    ax.set_yticks(np.arange(num_actions))
    # ax.set_xlabel("Action Dimension", fontsize=12)
    # ax.set_ylabel("Action Dimension", fontsize=12)
    
    # Optional: Add grid lines for better cell separation
    ax.set_xticks(np.arange(num_actions + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_actions + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False) # Hide minor ticks from axes

    # 6. Finalize the plot and save it
    title = f"Action Correlation Matrix (Scale: {action_scale:.2f}, Checkpoint: {checkpoint_name}, Env Steps: {env_steps})"
    # ax.set_title(title, fontsize=14, pad=20)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Action correlation matrix saved to: {save_path}")


def visualize_flow_field(trajectories, save_path, checkpoint_name, env_steps, action_scale=1.0, action_dim=20):
    """
    Visualize the flow field as a heatmap with marginal distributions and temporal scaling.
    - Center: Heatmap of probability density over time.
    - Left: Gaussian prior distribution at t=0.
    - Right: Histogram and fitted Gaussian of the final data distribution at t=1.
    """
    # 1. Setup figure and axes using GridSpec for a clean layout
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 5, 1], wspace=0.05)
    ax_left = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharey=ax_left)
    ax_right = fig.add_subplot(gs[2], sharey=ax_left)

    # 2. Extract, process, and scale trajectory data
    # Use the specified action dimension to visualize
    
    # The time values go from 1.0 down to 0.0, but for plotting we want 0.0 to 1.0
    times = np.array([1.0 - step[0] for step in trajectories[0]])
    values = np.array(
        [[step[1].flatten()[action_dim] for step in traj] for traj in trajectories]
    )
    
    # Interpolate the scale factor over time (from 1.0 at t=0 to action_scale at t=1)
    # scale_factors = 1.0 * (1 - times) + action_scale * times
    scale_factors = 1.0
    
    # Apply the interpolated scaling to the values using broadcasting
    scaled_values = values * scale_factors
    final_action_values = scaled_values[:, -1]

    # 3. Define plot range and bins for the y-axis dynamically based on scaled data
    y_min, y_max = -3.5, 3.5 #scaled_values.min() - 0.5, scaled_values.max() + 0.5
    num_y_bins = 128
    y_bins = np.linspace(y_min, y_max, num_y_bins)
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    # 4. Plot left side: Standard Normal distribution (always N(0,1) at t=0)
    # Always show standard normal distribution regardless of data scaling
    prior_pdf = norm.pdf(y_centers, 0, 1)
    ax_left.plot(prior_pdf, y_centers, color="gray", linewidth=2)
    ax_left.fill_betweenx(y_centers, prior_pdf, color="gray", alpha=0.2)
    ax_left.set_xlim(ax_left.get_xlim()[::-1])  # Flip to have the peak point inwards
    # ax_left.set_ylabel("Scaled Action Value", fontsize=12)
    for spine in ["top", "right", "bottom", "left"]:
        ax_left.spines[spine].set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # 5. Plot right side: Final data distribution (at t=1)
    ax_right.hist(
        final_action_values, bins=y_bins, orientation="horizontal", density=True, color="gray", alpha=0.7
    )
    mu, std = norm.fit(final_action_values)
    fit_pdf = norm.pdf(y_centers, mu, std)
    ax_right.plot(fit_pdf, y_centers, "gray", linewidth=2) #, "r-",label=f"Fit: $\\mu={mu:.2f}, \\sigma={std:.2f}$")
    for spine in ["top", "right", "bottom", "left"]:
        ax_right.spines[spine].set_visible(False)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # 6. Create and plot the central heatmap from scaled data
    num_steps = scaled_values.shape[1]
    heatmap_data = np.zeros((num_y_bins - 1, num_steps))
    for i in range(num_steps):
        counts, _ = np.histogram(scaled_values[:, i], bins=y_bins, density=True)
        heatmap_data[:, i] = counts

    # hide the left and right spines
    ax_main.spines['left'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    ax_main.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        extent=[times.min(), times.max(), y_min, y_max],
        cmap="viridis",
    )
    ax_main.set_xlabel("Time ($t$)", fontsize=12)
    plt.setp(ax_main.get_yticklabels(), visible=False)

    # 7. Finalize and save the plot
    title = f"Flow Field Density (Scale: {action_scale:.2f}, Checkpoint: {checkpoint_name}, Env Steps: {env_steps})"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Flow field visualization saved to: {save_path}")

def get_marginal_flow_trajectories(policy, observation, sample_noises, device=None, mini_batch_size=1000):
    """Sample trajectories through the flow matching denoising process (batched version, with mini-batching).
    
    Args:
        policy: The policy network.
        observation: The observation tensor (1, obs_dim) or (obs_dim,).
        sample_noises: (num_samples, action_dim) tensor of initial noise samples.
        device: torch device.
        mini_batch_size: Number of samples per mini-batch.
    Returns:
        trajectories: list of length num_samples, each is [(t0, x0), (t1, x1), ...]
    """
    x_t_all = sample_noises.clone()  # (num_samples, action_dim)
    num_samples = x_t_all.shape[0]
    action_dim = x_t_all.shape[1]
    trajectories = [[] for _ in range(num_samples)]

    def _flow_schedule(flow_steps: int):
        full_t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device)
        return full_t_path[:-1], full_t_path[1:]

    t_current, t_next = _flow_schedule(policy.sampling_steps)

    obs = observation[0:1].expand(num_samples, -1)  # (N, obs_dim)

    pure_array_trajectories = [[] for _ in range(num_samples)]
    for i in range(num_samples):
        pure_array_trajectories[i].append(x_t_all[i:i+1].detach().cpu().numpy())

    # Initial state at t=1.0
    for i in range(num_samples):
        trajectories[i].append((1.0, x_t_all[i:i+1].detach().cpu().numpy()))

    # Batched flow steps with mini-batching
    for t_curr, t_nxt in zip(t_current, t_next):
        dt = t_nxt - t_curr
        for start in range(0, num_samples, mini_batch_size):
            end = min(start + mini_batch_size, num_samples)
            batch_idx = slice(start, end)
            x_t = x_t_all[batch_idx]  # (B, action_dim)
            obs_batch = obs[batch_idx]  # (B, obs_dim)
            embedded_t = policy._embed_timestep(torch.tensor([[t_curr]], device=device)).expand(end - start, -1)
            mlp_input = torch.cat([obs_batch, embedded_t, x_t], dim=-1)  # (B, obs+embed+action)
            velocity = policy.actor(mlp_input)  # (B, action_dim)
            x_t_new = x_t + velocity * dt
            x_t_all[batch_idx] = x_t_new
            # Store trajectory for each sample in the batch
            for i, idx in enumerate(range(start, end)):
                trajectories[idx].append((t_nxt.item(), x_t_new[i:i+1].detach().cpu().numpy()))
                pure_array_trajectories[idx].append(x_t_new[i:i+1].detach().cpu().numpy())
            
    return trajectories, pure_array_trajectories


################## Main function ################## 
def main():
    """Plot Flow Field and Action Correlation for FPO agent."""

    ################## Load and override config ##################
    task_name = args_cli.task.split(":")[-1]
    # First determine the checkpoint path
    # Handle W&B checkpoint download if specified
    if args_cli.wandb_run_path and args_cli.wandb_checkpoint:
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
        print(f"[INFO] Loaded config has num_envs: {env_cfg.scene.num_envs}. Overriding to {args_cli.num_envs}.")
        # Override some settings for playback
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
    else:
        print(f"[WARNING] No saved environment config found at {env_pkl_path}, using default config")
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        print(f"[INFO] Created environment config with num_envs: {env_cfg.scene.num_envs}")
    # Don't move this. This should be before the environment creation.
    print(f"[INFO] Using seed from command line: {args_cli.seed} for environment config.")
    env_cfg.seed = args_cli.seed
    
    print(f"[INFO] Loading experiment from directory: {log_dir}")
    print(f"[INFO] Using policy network hidden dims: {agent_cfg.policy.actor_hidden_dims}")
    
    
    # Override zero_noise_sampling if provided via command line
    if args_cli.zero_noise_sampling and hasattr(agent_cfg.policy, 'zero_noise_sampling'):
        agent_cfg.policy.zero_noise_sampling = True
        print(f"[INFO] Overriding config: zero_noise_sampling = True")

    # Override the training sample steps if it is not None
    if args_cli.training_sampling_steps is not None and hasattr(agent_cfg.policy, 'training_sampling_steps'):
        print(f"[INFO] Overriding training sampling steps from {ppo_runner.alg.policy.training_sampling_steps} to {args_cli.training_sampling_steps}")
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


    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)


    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    checkpoint_name = os.path.basename(resume_path).replace(".pt", "")

    vis_dir = os.path.join(log_dir, f"flow_field_and_action_correlation_{checkpoint_name}")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[INFO] Visualization directory: {vis_dir}")

    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    import random
    import math

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    math.seed = SEED

    num_samples = args_cli.num_noise_samples
    max_env_steps = 100
    sample_noises = torch.randn(max_env_steps, num_samples, policy_nn.num_actions, device=env.unwrapped.device)

    obs, _ = env.get_observations()
    timestep = 0
    env_steps = 0
    joint_angle_first = env.unwrapped.scene["robot"].data.joint_pos.detach().cpu().numpy()
    root_pose_first = env.unwrapped.scene["robot"].data.root_pose_w.detach().cpu().numpy()

    # Data collection containers
    joint_angle_history = [joint_angle_first]
    root_pose_history = [root_pose_first]
    observations_history = [obs.detach().cpu().numpy()]
    actions_history = []
    trajectories_history = []
    trajectories_env_step_history = []
    
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # Record observation before passing to policy
            actions = policy(obs)
            # Record action output from the policy
            obs, _, _, _ = env.step(actions)
            env_steps += 1
            print("env_steps", env_steps)
            if env_steps == 1 or env_steps % args_cli.plot_interval == 0:
                trajectories, pure_array_trajectories = get_marginal_flow_trajectories(
                    policy_nn,
                    obs,
                    sample_noises[env_steps - 1],
                    device=env.unwrapped.device,
                    mini_batch_size=args_cli.marginal_flow_minibatch_size,
                )
                # trajectories[0][0]      
                # (1.0, tensor([[ 0.1940,  2.1614, -0.1721,  0.8491, -1.9244,  0.6530, -0.6494, -0.8175,
                # 0.5280, -1.2753, -1.6621, -0.3033, -0.0926,  0.1992, -1.1204,  1.8577,
                #   -0.7145,  0.6881,  0.7968]], device='cuda:0'))

                observations_history.append(obs.detach().cpu().numpy())
                actions_history.append(actions.detach().cpu().numpy())
                trajectories_history.append(pure_array_trajectories)
                trajectories_env_step_history.append(env_steps)
                joint_angle_history.append(env.unwrapped.scene["robot"].data.joint_pos.detach().cpu().numpy())
                root_pose_history.append(env.unwrapped.scene["robot"].data.root_pose_w.detach().cpu().numpy())

                skip_vis = False
                if not skip_vis:
                    for action_dim in args_cli.action_dims:
                        save_path = os.path.join(
                            vis_dir,
                            f"{task_name}_FPO++_{checkpoint_name}_a{action_dim}_flow_field_env_steps_{env_steps}.pdf",
                        )
                        visualize_flow_field(
                            trajectories,
                            save_path,
                            checkpoint_name,
                            env_steps,
                            action_scale=args_cli.action_scale,
                            action_dim=action_dim,
                        )

                    corr_save_path = os.path.join(
                        vis_dir,
                        f"{task_name}_FPO++_{checkpoint_name}_correlation_matrix_env_steps_{env_steps}.pdf",
                    )
                    visualize_action_correlation(
                        trajectories,
                        corr_save_path,
                        checkpoint_name,
                        env_steps,
                        action_scale=1. #args_cli.action_scale
                    )

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        if env_steps > max_env_steps:
            break
    env.close()

    # Save observations, actions, trajectories, and seed as a single NPZ
    try:
        obs_array = np.stack(observations_history) if len(observations_history) > 0 else np.empty((0,))
        act_array = np.stack(actions_history) if len(actions_history) > 0 else np.empty((0,))
        traj_env_step_times = np.array(trajectories_env_step_history) if len(trajectories_env_step_history) > 0 else np.empty((0,))
        traj_values = np.stack(trajectories_history) if len(trajectories_history) > 0 else np.empty((0,))
        joint_angle_history = np.stack(joint_angle_history) if len(joint_angle_history) > 0 else np.empty((0,))
        root_pose_history = np.stack(root_pose_history) if len(root_pose_history) > 0 else np.empty((0,))

        npz_path = os.path.join(
            vis_dir,
            f"{task_name}_{checkpoint_name}_sim_data_final.npz",
        )
        np.savez_compressed(
            npz_path,
            seed=SEED,
            observations=obs_array,
            actions=act_array,
            env_steps=traj_env_step_times,
            flow_field_over_time=traj_values,
            joint_angle_history=joint_angle_history,
            root_pose_history=root_pose_history,
        )

        """
        actions: (num_logged_env_time_steps,1, action_dim)
        observations: (num_logged_env_time_steps+1, 1, obs_dim)
        env_steps: (num_logged_env_time_steps,)
        flow_field_over_time: (num_logged_env_time_steps, num_noise_samples, num_flow_steps + 1, 1, action_dim) 
        # noise -> clean
        joint_angle_history: (num_logged_env_time_steps+1, 19)
        root_pose_history: (num_logged_env_time_steps+1, 7)
        """
        print(f"[INFO] Saved simulation data NPZ to: {npz_path}")
    except Exception as e:
        print(f"[WARN] Failed to save simulation data NPZ: {e}")



if __name__ == "__main__":
    main()
    simulation_app.close()
