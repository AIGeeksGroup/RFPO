import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from hand_imitation.env.create_env import create_env
from hand_imitation.env.gym_wrapper import GymWrapper
from algos.imitate.gym_util.multistep_wrapper import MultiStepWrapper
from algos.imitate.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from algos.imitate.policy.base_policy import BasePolicy
from algos.imitate.common.pytorch_util import dict_apply
from algos.imitate.env_runner.base_runner import BaseRunner
import algos.imitate.common.logger_util as logger_util


class SapienRunner(BaseRunner):
    def __init__(self, output_dir, n_train=10, max_steps=250, n_obs_steps=8, n_action_steps=8, fps=10, crf=22, tqdm_interval_sec=5.0, task_name=None, real_robot=False, noisy_points=False, point_cs="world"):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            if self.task_name == "mustard_bottle":
                self.env_name_list = ["ycb-006_mustard_bottle-20200709-subject-01-20200709_143211", "ycb-006_mustard_bottle-20200908-subject-05-20200908_144439", "ycb-006_mustard_bottle-20200928-subject-07-20200928_144226"]
            elif self.task_name == "tomato_soup_can":
                self.env_name_list = ["ycb-005_tomato_soup_can-20200709-subject-01-20200709_142853", "ycb-005_tomato_soup_can-20201015-subject-09-20201015_143403", "ycb-005_tomato_soup_can-20200709-subject-01-20200709_142926"]
            elif self.task_name == "sugar_box":
                self.env_name_list = ["ycb-004_sugar_box-20200918-subject-06-20200918_113441", "ycb-004_sugar_box-20200903-subject-04-20200903_104157", "ycb-004_sugar_box-20200908-subject-05-20200908_143931"]
            elif self.task_name == "large_clamp":
                self.env_name_list = ["ycb-052_extra_large_clamp-20200709-subject-01-20200709_152843", "ycb-052_extra_large_clamp-20200820-subject-03-20200820_144829", "ycb-052_extra_large_clamp-20201002-subject-08-20201002_112816"]
            elif self.task_name == "mug":
                self.env_name_list = ["ycb-025_mug-20200709-subject-01-20200709_150949", "ycb-025_mug-20200928-subject-07-20200928_154547", "ycb-025_mug-20200820-subject-03-20200820_143304"]
            elif self.task_name == "all":
                self.env_name_list = ["ycb-006_mustard_bottle-20200709-subject-01-20200709_143211", "ycb-006_mustard_bottle-20200908-subject-05-20200908_144439", "ycb-006_mustard_bottle-20200928-subject-07-20200928_144226", "ycb-005_tomato_soup_can-20200709-subject-01-20200709_142853", "ycb-005_tomato_soup_can-20201015-subject-09-20201015_143403", "ycb-005_tomato_soup_can-20200709-subject-01-20200709_142926", "ycb-004_sugar_box-20200918-subject-06-20200918_113441", "ycb-004_sugar_box-20200903-subject-04-20200903_104157", "ycb-004_sugar_box-20200908-subject-05-20200908_143931", "ycb-052_extra_large_clamp-20200709-subject-01-20200709_152843", "ycb-052_extra_large_clamp-20200820-subject-03-20200820_144829", "ycb-052_extra_large_clamp-20201002-subject-08-20201002_112816", "ycb-025_mug-20200709-subject-01-20200709_150949", "ycb-025_mug-20200928-subject-07-20200928_154547", "ycb-025_mug-20200820-subject-03-20200820_143304"]
            else:
                self.env_name_list = self.task_name.split('--')

            env_list = []
            for env_name in self.env_name_list:
                env = GymWrapper(create_env(name=env_name, is_eval=True, is_vision=True, is_real_robot=real_robot, pc_noise=noisy_points, point_cs=point_cs, norm_traj=True))
                env._base_env._stage = 2
                env_list.append(MultiStepWrapper(SimpleVideoRecordingWrapper(env), n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, max_episode_steps=max_steps, reward_agg_method='sum',))
            return env_list

        self.env_train_list = env_fn()
        self.episode_train = n_train

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train3 = logger_util.LargestKRecorder(K=5)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train_list = self.env_train_list

        ##############################
        # train env loop
        for idx, env_train in enumerate(env_train_list):
            all_returns_train = []
            all_success_rates_3 = []
            all_success_rates_10 = []
            all_sim_videos_train = []

            for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexSapien {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
                # start rollout
                obs = env_train.reset()

                policy.reset()

                done = False
                reward_sum = 0.
                for step_id in range(self.max_steps):
                    # create obs dict
                    np_obs_dict = dict(obs)
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                    # run policy
                    with torch.no_grad():
                        # add batch dim to match. (1,2,3,84,84)
                        # and multiply by 255, align with all envs
                        obs_dict_input = {}  # flush unused keys
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                        action_dict = policy.predict_action(obs_dict_input)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                    action = np_action_dict['action'].squeeze(0)

                    # step env
                    obs, reward, done, info = env_train.step(action)
                    reward_sum += reward
                    done = np.all(done)

                    if done:
                        break

                # videos_train = env_train.env.get_video()
                # all_sim_videos_train.append(wandb.Video(videos_train, fps=self.fps, format="mp4"))
                all_returns_train.append(reward_sum)
                SR3, SR10 = env_train.is_success()
                all_success_rates_3.append(SR3)
                all_success_rates_10.append(SR10)

            SR_mean_3 = np.mean(all_success_rates_3)
            SR_mean_10 = np.mean(all_success_rates_10)
            returns_mean_train = np.mean(all_returns_train)

            # log
            max_rewards = collections.defaultdict(list)
            log_data = dict()
            log_data['mean_success_rates_3'] = SR_mean_3
            log_data['mean_success_rates_10'] = SR_mean_10
            log_data['mean_returns_train'] = returns_mean_train
            log_data['test_mean_score'] = SR_mean_3
            # log_data['videos'] = all_sim_videos_train

            self.logger_util_train3.record(SR_mean_3)
            self.logger_util_train10.record(SR_mean_10)

            log_data['SR_train_L3'] = self.logger_util_train3.average_of_largest_K()
            log_data['SR_train_L10'] = self.logger_util_train10.average_of_largest_K()
        
            cur_env_name = self.env_name_list[idx]
            cprint(f"{cur_env_name} Mean SR 3: {SR_mean_3:.3f}", 'green')
            cprint(f"{cur_env_name} Mean SR 10: {SR_mean_10:.3f}", 'green')

            # clear out video buffer
            _ = env_train.reset()
            videos_train = None
            del env_train

        return log_data
