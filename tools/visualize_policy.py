import os
import sys
import _init_paths
import yaml
import json

import torch
import argparse
import numpy as np
from hand_imitation.env.create_env import create_env
from hand_imitation.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
from stable_baselines3 import PPO
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf
from algos.rl.fpo_trainer import load_fpo_state_policy


def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action


def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping ** 2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos


def compute_control_command(env, action: np.ndarray):
    current_qpos = env.robot.get_qpos()
    action = np.clip(action, -1, 1)
    target_root_velocity = recover_action(action[:6], env.velocity_limit[:6])
    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(current_qpos[:env.arm_dof])
    arm_qvel = compute_inverse_kinematics(target_root_velocity, palm_jacobian)[:env.arm_dof]
    arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
    arm_qpos = arm_qvel * env.control_time_step + env.robot.get_qpos()[:env.arm_dof]

    hand_qpos = recover_action(action[6:], env.robot.get_qlimits()[env.arm_dof:])

    target_qpos = np.concatenate([arm_qpos, hand_qpos])
    target_qvel = np.zeros_like(target_qpos)
    target_qvel[:env.arm_dof] = arm_qvel

    return target_qpos, target_qvel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--checkpoint_path', type=str, required=True)
    parser.add_argument('-v', '--use_visual_obs', action='store_true')
    parser.add_argument('-r', '--robot_name', type=str, default="allegro_hand_ur5")
    parser.add_argument('-s', '--save_control', action='store_true')
    parser.add_argument('-n', '--norm_traj', action='store_true')
    parser.add_argument('-env', '--env_name', type=str, default="ycb-006_mustard_bottle-20200709-subject-01-20200709_143211")
    args = parser.parse_args()

    if args.use_visual_obs:
        from imitate_train import TrainWorkspace
        config_path = os.path.join(args.checkpoint_path, ".hydra", "config.yaml")
        cfg = OmegaConf.load(config_path)
        workspace = TrainWorkspace(cfg)
        workspace.load_checkpoint(path=os.path.join(args.checkpoint_path, 'checkpoints', 'latest.ckpt'))
        policy = workspace.model
        env_name = args.env_name
        task_kwargs = None

        env = create_env(name=env_name, task_kwargs=task_kwargs, use_gui=True, is_eval=True, is_vision=True, norm_traj=args.norm_traj, robot_name=args.robot_name)
        env._stage = 0
    else:
        config_path = os.path.join(args.checkpoint_path, "exp_config.yaml")
        config = yaml.safe_load(open(config_path, 'r'))
        agent_name = config.get('params', {}).get('agent', {}).get('name', 'PPO')
        if agent_name == 'FPO':
            checkpoint_path = os.path.join(args.checkpoint_path, "restore_checkpoint.pt")
            policy = load_fpo_state_policy(checkpoint_path)
        else:
            checkpoint_path = os.path.join(args.checkpoint_path, "restore_checkpoint.zip")
            policy = PPO.load(checkpoint_path)
        env_name = config['params']['env']['name']
        task_kwargs = config['params']['env']['task_kwargs']

        env_cfg = config.get('params', {}).get('env', {})
        env = create_env(name=env_name, task_kwargs=task_kwargs, use_gui=True, is_eval=True, norm_traj=args.norm_traj, robot_name=env_cfg.get('robot_name', args.robot_name))
        env._stage = 0

    fps = 30
    while True:
        obs = env.reset()
        tot_reward = 0
        data_control = dict()
        data_control['qpos_obs'] = []
        data_control['qpos'] = []

        for j in range(env.horizon):
            if isinstance(obs, dict):
                for key, value in obs.items():
                    obs[key] = value[np.newaxis, :]
            else:
                obs = obs[np.newaxis, :]
            
            if args.use_visual_obs:
                obs['agent_pos'] = torch.from_numpy(obs['agent_pos'].astype(np.float32)).cuda().unsqueeze(1)
                obs['point_cloud'] = torch.from_numpy(obs['point_cloud'].astype(np.float32)).cuda().unsqueeze(1)
                action = policy.predict_action(obs)
                action = action['action'][:, 0].detach().cpu().numpy()

                target_qpos, target_qvel = compute_control_command(env, action[0])
                data_control['qpos_obs'].append(obs['agent_pos'][0, 0].detach().cpu().numpy().tolist())
                data_control['qpos'].append(target_qpos.tolist())
            else:
                action = policy.predict(observation=obs, deterministic=True)[0]

                target_qpos, target_qvel = compute_control_command(env, action[0])
                data_control['qpos_obs'].append(obs[0][:env.robot.dof].tolist())
                data_control['qpos'].append(target_qpos.tolist())

            if len(action.shape) > 0:
                action = action[0]
            obs, reward, done, _ = env.step(action)

            for _ in range(60 // fps):
                env.render()

            if done:
                break
        tot_reward += reward
        print(f"reward = {tot_reward}")
        
        if args.save_control:
            with open('control.json', 'w') as f:
                json.dump(data_control, f, indent=2)
            break
