#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal ViViDex state-env bootstrap check.")
    parser.add_argument("--seq-name", required=True, help="Sequence name without .npz suffix")
    parser.add_argument("--norm-traj", action="store_true", help="Use norm_trajectories instead of trajectories")
    parser.add_argument("--robot-name", default="allegro_hand_ur5", help="Robot key registered in common_robot_utils")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of vectorized envs for the check")
    parser.add_argument("--multi-proc", action="store_true", help="Use SubprocVecEnv instead of DummyVecEnv")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "hand_imitation"))

    import numpy as np
    import yaml

    from hand_imitation.utils.util import make_env

    env_cfg_path = repo_root / "algos" / "rl" / "config" / "env" / "env.yaml"
    env_cfg = yaml.safe_load(env_cfg_path.read_text())
    env_cfg["name"] = args.seq_name
    env_cfg["norm_traj"] = bool(args.norm_traj)
    env_cfg["robot_name"] = args.robot_name
    env_cfg["n_envs"] = int(args.n_envs)
    env_cfg["vid_freq"] = None

    from hand_imitation.env.motion_paths import resolve_existing_motion_path

    seq_path = resolve_existing_motion_path(repo_root, args.seq_name, args.norm_traj, args.robot_name)

    env = make_env(
        multi_proc=bool(args.multi_proc),
        is_eval=False,
        **env_cfg,
    )

    obs = env.reset()
    sample_action = np.stack([env.action_space.sample() for _ in range(args.n_envs)], axis=0)
    next_obs, reward, done, info = env.step(sample_action)

    print("[vividex] env bootstrap ok")
    print(f"[vividex] seq_name={args.seq_name}")
    print(f"[vividex] robot_name={args.robot_name}")
    print(f"[vividex] obs_type={type(obs).__name__}")
    print(f"[vividex] reward_shape={np.asarray(reward).shape}")
    print(f"[vividex] done_shape={np.asarray(done).shape}")
    print(f"[vividex] info_len={len(info)}")

    env.close()


if __name__ == "__main__":
    main()
