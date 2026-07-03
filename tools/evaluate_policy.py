import os
import argparse
import numpy as np
import yaml
import _init_paths
from tqdm import tqdm
from stable_baselines3 import PPO
from hand_imitation.env.create_env import create_env
from algos.rl.fpo_trainer import load_fpo_state_policy


def infer_algo_name(config):
    agent_name = config.get('params', {}).get('agent', {}).get('name', 'PPO')
    if agent_name == 'FPO':
        return 'fpo'
    return 'ppo'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--checkpoint_dir', type=str, required=True)
    parser.add_argument('-n', '--num_trajs', type=int, default=100)
    parser.add_argument('-r', '--real_robot', action='store_true')
    args = parser.parse_args()

    tot_count = 0
    state_arrays = []
    point_cloud_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    for checkpoint_dir in args.checkpoint_dir.split('--'):
        config_path = os.path.join(checkpoint_dir, "exp_config.yaml")
        config = yaml.safe_load(open(config_path, 'r'))
        algo_name = infer_algo_name(config)
        if algo_name == 'fpo':
            checkpoint_path = os.path.join(checkpoint_dir, "restore_checkpoint.pt")
            policy = load_fpo_state_policy(checkpoint_path)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "restore_checkpoint.zip")
            policy = PPO.load(checkpoint_path)
        env_name = config['params']['env']['name']
        env_cfg = config['params']['env']
        task_kwargs = env_cfg['task_kwargs']
        env = create_env(name=env_name, task_kwargs=task_kwargs, use_gui=False, is_eval=True, is_vision=False, is_demo_rollout=False, is_real_robot=args.real_robot, robot_name=env_cfg.get('robot_name', 'allegro_hand_ur5'))
        env._stage = 2

        SR3 = 0.0
        SR10 = 0.0
        for _ in tqdm(range(args.num_trajs)):
            obs = env.reset()
            for j in range(env.horizon):

                if isinstance(obs, dict):
                    for key, value in obs.items():
                        obs[key] = value[None, :]
                else:
                    obs = obs[None, :]

                try:
                    action = policy.predict(observation=obs, deterministic=True)[0]
                except:
                    fix_obs = np.zeros((1, 396), dtype=np.float32)
                    fix_obs[:, :367] = obs[:, :367]
                    fix_obs[:, 367:370] = obs[:, 364:367]
                    fix_obs[:, 370:] = obs[:, 367:]
                    action = policy.predict(observation=fix_obs, deterministic=True)[0]

                if len(action.shape) > 0:
                    action = action[0]

                obs, reward, done, info = env.step(action)

                if done:
                    break
            
            cur_SR3, cur_SR10 = env.is_success()
            SR3 += float(cur_SR3)
            SR10 += float(cur_SR10)
        
        avg_SR3 = SR3 / args.num_trajs
        avg_SR10 = SR10 / args.num_trajs
        print(f"avg SR3: {avg_SR3}")
        print(f"avg SR10: {avg_SR10}")
