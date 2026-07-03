#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import time
import _init_paths
import copy
import os

import numpy as np
from tqdm import tqdm
from sapien.utils import Viewer
from hand_imitation.env.create_env import create_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--seq_name', type=str, default='ycb-006_mustard_bottle-20200709-subject-01-20200709_143211')
    parser.add_argument('-n', '--norm_traj', action='store_true')
    parser.add_argument('-v', '--vis_pregrasp', action='store_true')
    parser.add_argument('-c', '--check_state', action='store_true')
    parser.add_argument('-r', '--robot_name', type=str, default="allegro_hand_ur5")
    args = parser.parse_args()

    if args.check_state:
        if args.norm_traj:
            traj_dir = "../norm_trajectories"
        else:
            traj_dir = "../trajectories"

        object_name_list = ["master_chef_can", "tuna_fish_can", "pudding_box", "gelatin_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "potted_meat_can", "banana", "pitcher_base", "bleach_cleanser", "mug", "wood_block", "extra_large_clamp", "foam_brick"]
        for filename in tqdm(os.listdir(traj_dir)):
            if any(cur_obj in filename for cur_obj in object_name_list):
                seq_name = filename.split('.')[0]
                env = create_env(name=seq_name, norm_traj=args.norm_traj, use_gui=False, is_eval=True, robot_name=args.robot_name)
                env.reset()
                env.initial_object_height = env._motion_file['object_translation'][0, 2] + 0.003
                for i in range(env.horizon):
                    env.step(np.zeros(env.action_dim))
                init_object_height = env.manipulated_object.get_pose().p[2]
                init_object_quat = env._motion_file['object_orientation'][0].tolist()
                traj_data = np.load(os.path.join(traj_dir, filename))
                clean_data = dict()
                for key in traj_data.files:
                    clean_data[key] = traj_data[key].copy()
                clean_data['init_object_height'] = init_object_height
                clean_data['init_object_quat'] = init_object_quat
                np.savez(f'{traj_dir}/{filename}', **clean_data)
    else:
        env = create_env(name=args.seq_name, norm_traj=args.norm_traj, use_gui=True, is_eval=True, is_vision=True, robot_name=args.robot_name)
        env._stage = 2
        print(f"init object height: {env.init_object_height}")
        robot_dof = env.robot.dof
        env.seed(0)
        env.reset()
        # config the viewer
        viewer = Viewer(env.renderer)
        viewer.set_scene(env.scene)
        viewer.focus_camera(env.cameras['instance_1'])
        env.viewer = viewer

        fps = 20
        while True:
            env.reset(vis_pregrasp=args.vis_pregrasp)
            for i in range(env.horizon):
                # env.step(np.random.random(robot_dof))
                env.step(np.zeros(env.action_dim))
                for _ in range(60 // fps):
                    env.render()
        viewer.close()
