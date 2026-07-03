from functools import cached_property
from typing import Optional

import os
import copy
import trimesh
import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from hand_imitation.env.rl_env.base import BaseRLEnv, compute_inverse_kinematics
from hand_imitation.env.rl_env.reference_utils import (
    neutral_hand_qpos_for_qlimits,
    rm75_hand_qpos_for_qlimits,
    should_terminate_for_contact_loss,
    slice_hand_qpos,
)
from hand_imitation.utils.ycb_object_utils import INVERSE_YCB_CLASSES, YCB_ROOT, YCB_CLASSES, YCB_HEIGHT_UNSEEN, YCB_ORIENTATION_UNSEEN
from hand_imitation.env.sim_env.relocate_env import LabRelocateEnv, configured_ycb_object_scale
from hand_imitation.real_world import lab


RM75_INITIAL_HAND_QPOS = np.zeros(6, dtype=np.float32)
RM75_PREGRASP_HAND_QPOS = np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)


def to_quat(arr):
    if isinstance(arr, Quaternion):
        return arr.unit
    if len(arr.shape) == 2:
        return Quaternion(matrix=arr).unit
    elif len(arr.shape) == 1 and arr.shape[0] == 9:
        return Quaternion(matrix=arr.reshape((3,3))).unit
    return Quaternion(array=arr).unit


def rotation_distance(q1, q2):
    delta_quat = to_quat(q2) * to_quat(q1).inverse
    return np.abs(delta_quat.angle)


def get_relocate_robot_link_config(robot_name):
    if robot_name == "rm75_inspire_right":
        finger_tip_names = [
            "rh_thumb_tip",
            "rh_index_tip",
            "rh_middle_tip",
            "rh_ring_tip",
            "rh_pinky_tip",
        ]
        finger_contact_link_names = [
            "rh_thumb_tip",
            "rh_thumb_distal",
            "rh_thumb_intermediate",
            "rh_thumb_proximal",
            "rh_index_tip",
            "rh_index_intermediate",
            "rh_index_proximal",
            "rh_middle_tip",
            "rh_middle_intermediate",
            "rh_middle_proximal",
            "rh_ring_tip",
            "rh_ring_intermediate",
            "rh_ring_proximal",
            "rh_pinky_tip",
            "rh_pinky_intermediate",
            "rh_pinky_proximal",
        ]
        finger_contact_ids = np.array(
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
            dtype=np.int64,
        )
        robot_joint_link_names = [
            "link_1",
            "link_2",
            "link_3",
            "link_4",
            "link_5",
            "link_6",
            "link_7",
            "rh_hand_base_link",
            "rh_thumb_proximal_base",
            "rh_thumb_proximal",
            "rh_thumb_intermediate",
            "rh_thumb_distal",
            "rh_index_proximal",
            "rh_index_intermediate",
            "rh_middle_proximal",
            "rh_middle_intermediate",
            "rh_ring_proximal",
            "rh_ring_intermediate",
            "rh_pinky_proximal",
            "rh_pinky_intermediate",
        ]
        return finger_tip_names, finger_contact_link_names, finger_contact_ids, robot_joint_link_names

    if robot_name == "allegro_hand_ur5":
        finger_tip_names = [
            "right_gripper_link_15_tip",
            "right_gripper_link_03_tip",
            "right_gripper_link_07_tip",
            "right_gripper_link_11_tip",
        ]
        finger_contact_link_names = [
            "right_gripper_link_15_tip",
            "right_gripper_link_15",
            "right_gripper_link_14",
            "right_gripper_link_03_tip",
            "right_gripper_link_03",
            "right_gripper_link_02",
            "right_gripper_link_01",
            "right_gripper_link_07_tip",
            "right_gripper_link_07",
            "right_gripper_link_06",
            "right_gripper_link_05",
            "right_gripper_link_11_tip",
            "right_gripper_link_11",
            "right_gripper_link_10",
            "right_gripper_link_09",
        ]
        finger_contact_ids = np.array([0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4], dtype=np.int64)
        robot_joint_link_names = [
            "right_shoulder_link",
            "right_upper_arm_link",
            "right_forearm_link",
            "right_wrist_1_link",
            "right_wrist_2_link",
            "right_wrist_3_link",
            "right_gripper_link_00",
            "right_gripper_link_01",
            "right_gripper_link_02",
            "right_gripper_link_03",
            "right_gripper_link_04",
            "right_gripper_link_05",
            "right_gripper_link_06",
            "right_gripper_link_07",
            "right_gripper_link_08",
            "right_gripper_link_09",
            "right_gripper_link_10",
            "right_gripper_link_11",
            "right_gripper_link_12",
            "right_gripper_link_13",
            "right_gripper_link_14",
            "right_gripper_link_15",
        ]
        return finger_tip_names, finger_contact_link_names, finger_contact_ids, robot_joint_link_names

    raise NotImplementedError(f"No relocate link config for robot: {robot_name}")


class AllegroRelocateRLEnv(LabRelocateEnv, BaseRLEnv):
    def __init__(self, is_eval=False, is_vision=False, is_demo_rollout=False, is_real_robot=False, pc_noise=False, point_cs="world", norm_traj=False, task_kwargs=None, use_gui=False, frame_skip=10, motion_file=None, robot_name="allegro_hand_ur5", object_category="YCB", root_frame="robot", **renderer_kwargs):
        task_kwargs = task_kwargs or {}
        self.task_kwargs = task_kwargs
        try:
            object_name = '_'.join(motion_file['object_name'].tolist().split('_')[1:])
            self._motion_file = motion_file.copy()
            self._reference_motion = motion_file.copy()
            self.init_object_height = self._reference_motion['init_object_height']
            self.task_name = motion_file['task_name']
        except:
            object_name = motion_file.split('-')[0]
            pose_idx = int(motion_file.split('-')[-1])
            self.init_object_height = YCB_HEIGHT_UNSEEN[object_name][pose_idx]
            self.init_object_quat = YCB_ORIENTATION_UNSEEN[object_name][pose_idx]
            self.task_name = "relocate"
            if "master_chef_can" in object_name or "bleach_cleanser" in object_name:
                self.init_object_height *= 0.8

            if "pitcher_base" in object_name or "wood_block" in object_name:
                self.init_object_height *= 0.6

        self.object_scale = configured_ycb_object_scale(self.task_name, task_kwargs)
        self.init_object_height *= self.object_scale

        self.is_eval = is_eval
        self.is_vision = is_vision
        self.is_demo_rollout = is_demo_rollout
        self.is_real_robot = is_real_robot
        self.pc_noise = pc_noise
        self.point_cs = point_cs
        self.norm_traj = norm_traj
        super().__init__(use_gui, frame_skip, robot_name, object_category, object_name, **renderer_kwargs)

        # Base class
        self.setup(robot_name)

        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Base frame for observation
        self.root_frame = root_frame
        self.base_frame_pos = np.zeros(3)

        finger_tip_names, finger_contact_link_names, self.finger_contact_ids, robot_joint_names = get_relocate_robot_link_config(robot_name)
        robot_link_names = [link.get_name() for link in self.robot.get_links()]
        link_name_to_link = {link.get_name(): link for link in self.robot.get_links()}
        missing_names = [
            name for name in finger_tip_names + finger_contact_link_names + robot_joint_names + [self.palm_link_name]
            if name not in link_name_to_link
        ]
        if missing_names:
            raise RuntimeError(f"Robot {robot_name} misses required relocate links: {missing_names}")
        self.finger_tip_links = [link_name_to_link[name] for name in finger_tip_names]
        self.robot_joint_links = [link_name_to_link[name] for name in robot_joint_names]
        self.finger_contact_links = [link_name_to_link[name] for name in finger_contact_link_names]
        self.finger_tip_pos = np.zeros([len(finger_tip_names), 3])

        # Contact buffer
        self.robot_object_contact = np.zeros(self.num_contact_groups, dtype=np.float32)

        # Reference trajectory updates
        try:
            self._substeps = int(self._reference_motion['SIM_SUBSTEPS'])                
            self._data_substeps = self._reference_motion.get('DATA_SUBSTEPS', self._substeps)              
            self._step, self.traj_step = 0, 0
            pregrasp_step = self._reference_motion['pregrasp_step']

            self._reference_motion['robot_pregrasp_jpos'] = self._reference_motion['robot_jpos'][:pregrasp_step + 1].copy()
            self._reference_motion['robot_jpos'] = self._reference_motion['robot_jpos'][pregrasp_step:]
            self._reference_motion['object_translation'] = self._reference_motion['object_translation'][pregrasp_step:]
            self._reference_motion['object_orientation'] = self._reference_motion['object_orientation'][pregrasp_step:]
            self._reference_motion['length'] = self._reference_motion['length'] - pregrasp_step
            self.pregrasp_qpos = slice_hand_qpos(self._reference_motion['robot_qpos'][pregrasp_step], self.arm_dof)
            self.cur_reference_motion = copy.deepcopy(self._reference_motion)
            self.start_step = 0
        except:
            self._step, self.traj_step = 0, 0
            self.start_step = 0

        self._stage = 0
        self.init_x = 0.35
        self.init_y = 0.35
        self.target_object_pos = np.zeros(3)

    def _compute_rm75_pregrasp_arm_qpos(self, target_palm_pos):
        qpos = np.zeros(self.robot.dof, dtype=np.float32)
        qpos[:self.arm_dof] = self.robot_info.arm_init_qpos
        hand_qlimits = self.robot.get_qlimits()[self.arm_dof:]
        if bool(getattr(self, "rm75_native_hand_control", False)):
            qpos[self.arm_dof:] = neutral_hand_qpos_for_qlimits(hand_qlimits)
        else:
            qpos[self.arm_dof:] = rm75_hand_qpos_for_qlimits(RM75_PREGRASP_HAND_QPOS, hand_qlimits)

        qlimits = self.robot.get_qlimits()
        scene_state = self.scene.pack()
        try:
            orientation_weight = max(float(getattr(self, "rm75_pregrasp_ik_orientation_weight", 0.0)), 0.0)
            desired_ee_quat = None
            if orientation_weight > 0.0:
                self.robot.set_qpos(qpos)
                self.scene.step()
                desired_ee_quat = np.asarray(self.ee_link.get_pose().q, dtype=np.float32)
            for _ in range(120):
                self.robot.set_qpos(qpos)
                self.scene.step()
                palm_pos = self.palm_link.get_pose().p
                pos_error = np.asarray(target_palm_pos, dtype=np.float32) - palm_pos
                if np.linalg.norm(pos_error) < 0.015:
                    break

                delta_pose = np.zeros(6, dtype=np.float32)
                delta_pose[:3] = np.clip(pos_error, -0.04, 0.04)
                if desired_ee_quat is not None:
                    current_ee_quat = np.asarray(self.ee_link.get_pose().q, dtype=np.float32)
                    desired_rot = R.from_quat(desired_ee_quat[[1, 2, 3, 0]])
                    current_rot = R.from_quat(current_ee_quat[[1, 2, 3, 0]])
                    rotvec = (desired_rot * current_rot.inv()).as_rotvec()
                    delta_pose[3:] = np.clip(
                        orientation_weight * rotvec,
                        -0.08,
                        0.08,
                    )
                jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(qpos[:self.arm_dof])
                delta_qpos = compute_inverse_kinematics(delta_pose, jacobian, damping=0.08)[:self.arm_dof]
                qpos[:self.arm_dof] = np.clip(qpos[:self.arm_dof] + delta_qpos, qlimits[:self.arm_dof, 0], qlimits[:self.arm_dof, 1])
        finally:
            self.scene.unpack(scene_state)
            zero_qvel = np.zeros(self.robot.dof, dtype=np.float32)
            if hasattr(self.robot, "set_qvel"):
                self.robot.set_qvel(zero_qvel)
            if hasattr(self.robot, "set_drive_velocity_target"):
                self.robot.set_drive_velocity_target(zero_qvel)
            if hasattr(self.robot, "set_qf"):
                self.robot.set_qf(np.zeros(self.robot.dof, dtype=np.float32))
        return qpos

    def _rm75_pregrasp_qpos(self):
        qpos = np.zeros(self.robot.dof, dtype=np.float32)
        qpos[:self.arm_dof] = self.robot_info.arm_init_qpos
        hand_qlimits = self.robot.get_qlimits()[self.arm_dof:]
        if bool(getattr(self, "rm75_native_hand_control", False)):
            qpos[self.arm_dof:] = neutral_hand_qpos_for_qlimits(hand_qlimits)
        else:
            qpos[self.arm_dof:] = rm75_hand_qpos_for_qlimits(RM75_PREGRASP_HAND_QPOS, hand_qlimits)

        try:
            target_palm_pos = np.asarray(self.cur_reference_motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
            if bool(getattr(self, "rm75_scripted_action_prior_offset_pregrasp", False)):
                grasp_offset = np.asarray(
                    getattr(self, "rm75_scripted_action_prior_grasp_offset", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                )
                if grasp_offset.shape == (3,):
                    target_palm_pos = target_palm_pos + grasp_offset
        except Exception:
            return qpos
        robot_pose = self.robot.get_pose()
        cache_key = (
            tuple(np.round(target_palm_pos, 5).tolist()),
            tuple(np.round(np.asarray(robot_pose.p, dtype=np.float32), 5).tolist()),
            tuple(np.round(np.asarray(robot_pose.q, dtype=np.float32), 5).tolist()),
            bool(getattr(self, "rm75_native_hand_control", False)),
            int(self.robot.dof),
            int(self.arm_dof),
        )
        cache = getattr(self, "_rm75_pregrasp_qpos_cache", None)
        if cache is None:
            cache = {}
            self._rm75_pregrasp_qpos_cache = cache
        cached_qpos = cache.get(cache_key)
        if cached_qpos is not None:
            return cached_qpos.copy()
        qpos = self._compute_rm75_pregrasp_arm_qpos(target_palm_pos)
        cache[cache_key] = qpos.copy()
        return qpos

    def _apply_rm75_approach_pregrasp_reference(self):
        if self.robot_name != "rm75_inspire_right":
            return
        motion = getattr(self, "cur_reference_motion", None)
        if motion is None or "robot_jpos" not in motion or "robot_pregrasp_jpos" not in motion:
            return
        if bool(getattr(self, "rm75_native_hand_control", False)) and not bool(
            getattr(self, "rm75_native_apply_template_approach_pregrasp", False)
        ):
            return
        has_template_delta = "rm75_template_approach_delta" in motion
        follow_reference_delta = bool(getattr(self, "rm75_follow_reference_approach_delta", False))
        override_template_delta = bool(getattr(self, "rm75_override_template_approach_delta", False))
        if has_template_delta and follow_reference_delta:
            # ``robot_pregrasp_jpos`` has already been rotated/translated with the
            # episode trajectory during reset, while the stored template delta is
            # in canonical coordinates.  Preserve the transformed approach vector.
            # ``rm75_override_template_approach_delta`` only affects which
            # canonical delta is seeded before reset; it should not pin the
            # episode approach direction to world coordinates after reset.
            approach_delta = (
                np.asarray(motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
                - np.asarray(motion["robot_jpos"][0, 0], dtype=np.float32)
            )
        elif has_template_delta and not override_template_delta:
            approach_delta = np.asarray(motion["rm75_template_approach_delta"], dtype=np.float32)
        else:
            approach_delta = np.asarray(
                getattr(self, "rm75_default_approach_delta", [0.0, -0.05, 0.0]),
                dtype=np.float32,
            )
        if approach_delta.shape != (3,):
            return
        grasp_reference = np.asarray(motion["robot_jpos"][0], dtype=np.float32)
        motion["robot_pregrasp_jpos"][-1, : len(grasp_reference)] = grasp_reference + approach_delta[None, :]

    def _zero_manipulated_object_velocity(self):
        if hasattr(self.manipulated_object, "set_velocity"):
            self.manipulated_object.set_velocity(np.zeros(3, dtype=np.float32))
        if hasattr(self.manipulated_object, "set_angular_velocity"):
            self.manipulated_object.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def get_oracle_state(self):
        robot_state = self.get_robot_state()
        object_state = self.get_object_state()
        goal_state = self.get_goal_state()
        time_state = self.get_time_state()
        return np.concatenate([robot_state, object_state, goal_state, time_state])
    
    def get_object_state(self):
        object_pos = self.manipulated_object.get_pose().p
        object_quat = self.manipulated_object.get_pose().q
        object_lin_vel = self.manipulated_object.get_velocity()
        object_ang_vel = self.manipulated_object.get_angular_velocity()
        return np.concatenate([object_pos, object_quat, object_lin_vel, object_ang_vel])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        robot_qvel_vec = self.robot.get_qvel()
        robot_joint_pos = np.zeros([len(self.robot_joint_links), 3])
        robot_joint_quat = np.zeros([len(self.robot_joint_links), 4])
        robot_joint_lin_vel = np.zeros([len(self.robot_joint_links), 3])
        robot_joint_ang_vel = np.zeros([len(self.robot_joint_links), 3])
        for i, link in enumerate(self.robot_joint_links):
            robot_joint_pos[i] = self.robot_joint_links[i].get_pose().p
            robot_joint_quat[i] = self.robot_joint_links[i].get_pose().q
            robot_joint_lin_vel[i] = self.robot_joint_links[i].get_velocity()
            robot_joint_ang_vel[i] = self.robot_joint_links[i].get_angular_velocity()
        return np.concatenate([robot_qpos_vec, robot_qvel_vec, robot_joint_pos.reshape(-1), robot_joint_quat.reshape(-1), robot_joint_lin_vel.reshape(-1), robot_joint_ang_vel.reshape(-1)])
    
    def get_test_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        if self.is_vision:
            if self.norm_traj:
                robot_target_vec = np.array([self.init_x, self.init_y, 0.2])
            else:
                robot_target_vec = self.target_object_pos
        else:
            if self.norm_traj:
                robot_target_vec = np.array([self.init_x, self.init_y, 0.2])
            else:
                robot_target_vec = self.target_object.get_pose().p

        robot_mat = np.eye(4, dtype=np.float32)[None].repeat(len(self.finger_tip_links) + 1, 0)
        for i, link in enumerate(self.finger_tip_links):
            robot_mat[i] = self.finger_tip_links[i].get_pose().to_transformation_matrix()
        robot_mat[-1] = self.palm_link.get_pose().to_transformation_matrix()

        return np.concatenate([robot_qpos_vec, robot_target_vec, robot_mat.reshape(-1)])

    def _get_vision_obs_from_test_state(self, test_state):
        obs = dict()
        agent_dim = self.robot.dof
        target_slice = slice(agent_dim, agent_dim + 3)
        transform_start = agent_dim + 3
        point_cloud = self.get_camera_obs()['instance_1-point_cloud']
        trans_mat = test_state[transform_start:].reshape((-1, 4, 4))
        obs['agent_pos'] = test_state[:agent_dim]
        if self.point_cs == "target":
            point_cloud = np.concatenate([point_cloud, point_cloud - test_state[target_slice]], 1)
        elif self.point_cs == "hand":
            pcs_list = []
            pcs_list.append(point_cloud)
            pcs_list.append(point_cloud - test_state[target_slice])
            for idx in range(len(trans_mat)):
                trans_pc = point_cloud - trans_mat[idx, :3, 3]
                trans_pc = (trans_mat[idx, :3, :3].T @ trans_pc.transpose(1, 0)).transpose(1, 0)
                pcs_list.append(trans_pc)
            point_cloud = np.concatenate(pcs_list, 1)
        obs['point_cloud'] = point_cloud
        return obs

    @property
    def num_contact_groups(self):
        return int(np.max(self.finger_contact_ids)) + 1

    def _contact_groups(self):
        check_contact_links = self.finger_contact_links + [self.palm_link]
        contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.manipulated_object)
        contact_groups = np.bincount(
            self.finger_contact_ids,
            weights=contact_boolean,
            minlength=self.num_contact_groups,
        )
        return np.clip(contact_groups, 0, 1).astype(np.float32)

    def _reference_hand_jpos(self, robot_jpos):
        if robot_jpos.shape[0] == len(self.finger_tip_links) + 1:
            return robot_jpos[1:]
        if robot_jpos.shape[0] == len(self.finger_tip_links):
            return robot_jpos
        if robot_jpos.shape[0] - 1 < len(self.finger_tip_links):
            result = np.zeros((len(self.finger_tip_links), 3), dtype=robot_jpos.dtype)
            result[:robot_jpos.shape[0] - 1] = robot_jpos[1:]
            result[robot_jpos.shape[0] - 1:] = robot_jpos[-1]
            return result
        return robot_jpos[1:len(self.finger_tip_links) + 1]
    
    def get_goal_state(self):
        traj_goals = []
        for i in [1, 5, 10]:
            if self.traj_step + i <= self.pregrasp_steps:
                for k in ('object_orientation', 'object_translation'):
                    traj_goals.append(self.cur_reference_motion[k][0].flatten())
            else:
                i = min(self.traj_step + i, self.imitate_steps - 1) - self.pregrasp_steps
                for k in ('object_orientation', 'object_translation'):
                    traj_goals.append(self.cur_reference_motion[k][i].flatten())
        traj_goals = np.concatenate(traj_goals)
        
        hand_obj_diff = self.palm_link.get_pose().p - self.manipulated_object.get_pose().p
        finger_tip_pos = np.zeros([len(self.finger_tip_links), 3])
        for i, _ in enumerate(self.finger_tip_links):
            finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        hand_obj_dense_diff = finger_tip_pos - self.manipulated_object.get_pose().p
        if self.is_vision:
            hand_tgt_diff = self.palm_link.get_pose().p - self.target_object_pos
            obj_tgt_diff = self.manipulated_object.get_pose().p - self.target_object_pos
        else:
            hand_tgt_diff = self.palm_link.get_pose().p - self.target_object.get_pose().p
            obj_tgt_diff = self.manipulated_object.get_pose().p - self.target_object.get_pose().p
        return np.concatenate([traj_goals, hand_obj_diff.reshape(-1), hand_obj_dense_diff.reshape(-1), hand_tgt_diff.reshape(-1), obj_tgt_diff.reshape(-1)])
    
    def get_time_state(self):
        t = self.traj_step / self.imitate_steps
        t = np.array([1, 4, 6, 8]) * t
        time_state = np.concatenate((np.sin(t), np.cos(t)))
        return time_state

    def get_reward(self, action):
        if self.is_vision and not self.is_demo_rollout:
            return 0
        else:
            robot_hand_links = self.finger_tip_links
            robot_hand_pos = np.zeros([len(robot_hand_links) , 3])
            for i, link in enumerate(robot_hand_links):
                robot_hand_pos[i] = robot_hand_links[i].get_pose().p
            object_pos = self.manipulated_object.get_pose().p
            object_rot = self.manipulated_object.get_pose().q

            self.robot_object_contact[:] = self._contact_groups()
            self.is_contact = sum(self.robot_object_contact[:]) >= 1
            self.object_lift = max(object_pos[2] - self.init_object_height, 0)
        
            if self.current_step <= self.pregrasp_steps:
                tgt_robot_hand_pos = self._reference_hand_jpos(self.cur_reference_motion['robot_pregrasp_jpos'][-1])
                self.hand_jpos_err = np.mean(np.linalg.norm(robot_hand_pos - tgt_robot_hand_pos, axis=1))
                reward = 10 * np.exp(-10 * self.hand_jpos_err)
            else:
                reward = sum(self.robot_object_contact) * 0.5

                tgt_object_pos = self.cur_reference_motion['object_translation'][self.current_step - self.pregrasp_steps]
                tgt_object_rot = self.cur_reference_motion['object_orientation'][self.current_step - self.pregrasp_steps]
                tgt_robot_hand_pos = self._reference_hand_jpos(self.cur_reference_motion['robot_jpos'][self.current_step - self.pregrasp_steps])

                self.obj_com_err = np.linalg.norm(object_pos - tgt_object_pos)
                self.obj_rot_err = rotation_distance(object_rot, tgt_object_rot) / np.pi
                self.hand_mjpos_err = np.mean(np.linalg.norm(robot_hand_pos - tgt_robot_hand_pos, axis=1))

                reward += 10 * np.exp(-50 * (self.obj_com_err + 0.1 * self.obj_rot_err))
                reward += 4.0 * np.exp(-10 * self.hand_mjpos_err)

                if self.object_lift > 0.02:
                    reward += 2.5
        
            if not self.pregrasp_success and self.current_step == self.pregrasp_steps:
                if self.hand_jpos_err < 0.05:
                    self.pregrasp_success = True

            controller_penalty = (self.cartesian_error ** 2) * -1e3
            action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * -0.01

            return (reward + action_penalty + controller_penalty) / 10

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, vis_pregrasp = False):
        # Gym reset function
        if seed is not None:
            self.seed(seed)
        
        if self._stage == 0:
            traj_x = 0.35
            traj_y = 0.35
            traj_rot = 0.0
        elif self._stage == 1:
            traj_x = self.np_random.uniform(low=0.3, high=0.4)
            traj_y = self.np_random.uniform(low=0.3, high=0.4)
            traj_rot = 0.0
        elif self._stage == 2:
            traj_x = self.np_random.uniform(low=0.3, high=0.4)
            traj_y = self.np_random.uniform(low=0.3, high=0.4)
            traj_rot = self.np_random.uniform(low=-1/12, high=1/12) * np.pi

        self.init_x = traj_x
        self.init_y = traj_y

        try:
            self.cur_reference_motion = copy.deepcopy(self._reference_motion)
            # canonicalize the trajectory
            self.cur_reference_motion['robot_jpos'] -= self.cur_reference_motion['object_translation'][0]
            self.cur_reference_motion['object_translation'] -= self.cur_reference_motion['object_translation'][0]
            self.cur_reference_motion['robot_jpos'][:, :, 2] += self.init_object_height
            self.cur_reference_motion['object_translation'][:, 2] += self.init_object_height
            
            # Rotate the trajectory
            rot_matrix = np.array([[np.cos(traj_rot), -np.sin(traj_rot), 0], [np.sin(traj_rot), np.cos(traj_rot), 0], [0, 0, 1]])
            self.cur_reference_motion['object_translation'] = (rot_matrix @ self.cur_reference_motion['object_translation'].transpose(1, 0)).transpose(1, 0)
            for idx in range(self.cur_reference_motion['object_orientation'].shape[0]):
                self.cur_reference_motion['object_orientation'][idx] = Quaternion(matrix=(rot_matrix @ Quaternion(self.cur_reference_motion['object_orientation'][idx]).rotation_matrix)).elements
            self.cur_reference_motion['robot_jpos'] = (rot_matrix[None] @ self.cur_reference_motion['robot_jpos'].transpose(0, 2, 1)).transpose(0, 2, 1)
            self.cur_reference_motion['robot_pregrasp_jpos'] = (rot_matrix[None] @ self.cur_reference_motion['robot_pregrasp_jpos'].transpose(0, 2, 1)).transpose(0, 2, 1)

            # Translate the trajectory
            self.cur_reference_motion['object_translation'][:, :2] += np.array([traj_x, traj_y])
            self.cur_reference_motion['robot_jpos'][:, :, :2] += np.array([traj_x, traj_y])
            self.cur_reference_motion['robot_pregrasp_jpos'][:, :, :2] += np.array([traj_x, traj_y])
            self._apply_rm75_approach_pregrasp_reference()
            
            # Imitate until lifting by 0.1m
            if not self.norm_traj:
                keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.1)[0][0] + 1
                self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
                self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
                self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]
            else:
                if self.task_name == "pour":
                    target_pos = np.array([0.15, 0.35, 0.12])
                    target_orn = (R.from_rotvec(-2 * np.pi / 3 * np.array([0, 1, 0])) * R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

                    dist_vec = target_pos - self.cur_reference_motion['object_translation'][-1]
                    unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
                    relocate_step = 0.01
                    num_step = int(np.linalg.norm(dist_vec) // relocate_step)
                    init_rot = R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])
                    rotation_step = -2 * np.pi / 3 / num_step

                    syn_object_translation = []
                    syn_object_orientation = []
                    syn_robot_jpos = []
                    for idx in range(num_step):
                        step_size = (idx + 1) * relocate_step
                        syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                        rot_size = (idx + 1) * rotation_step
                        syn_object_orientation.append((R.from_rotvec(rot_size * np.array([0, 1, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])
                        cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                        rotmat = R.from_rotvec(rot_size * np.array([0, 1, 0])).as_matrix()
                        cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                        syn_robot_jpos.append(cur_joint)
                    
                    rot_size = -2 * np.pi / 3
                    syn_object_translation.append(target_pos)
                    syn_object_orientation.append(target_orn)
                    cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                    rotmat = R.from_rotvec(rot_size * np.array([0, 1, 0])).as_matrix()
                    cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                    syn_robot_jpos.append(cur_joint)

                    self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
                    self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
                    self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))
                elif self.task_name == "place":
                    keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.25)[0][0] + 1
                    self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
                    self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
                    self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]

                    target_pos = np.array([0.15, 0.35, 0.12])
                    target_orn = (R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])) * R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

                    dist_vec = target_pos - self.cur_reference_motion['object_translation'][-1]
                    unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
                    relocate_step = 0.01
                    num_step = int(np.linalg.norm(dist_vec) // relocate_step)
                    init_rot = R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])
                    rotation_step = -np.pi / 2 / num_step

                    syn_object_translation = []
                    syn_object_orientation = []
                    syn_robot_jpos = []
                    for idx in range(num_step):
                        step_size = (idx + 1) * relocate_step
                        syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                        rot_size = (idx + 1) * rotation_step
                        syn_object_orientation.append((R.from_rotvec(rot_size * np.array([1, 0, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])
                        cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                        rotmat = R.from_rotvec(rot_size * np.array([1, 0, 0])).as_matrix()
                        cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                        syn_robot_jpos.append(cur_joint)
                    
                    descent_vec = np.array([0, 0, -0.01])
                    for idx in range(2):
                        syn_object_translation.append(syn_object_translation[-1] + descent_vec)
                        syn_object_orientation.append(syn_object_orientation[-1])
                        syn_robot_jpos.append(syn_robot_jpos[-1] + descent_vec[None, :])

                    self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
                    self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
                    self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))

            self.traj_len = len(self.cur_reference_motion['object_translation'])
            self.traj_step = 0
            self._step = 0
            
            # Set the final goal
            self._final_goal = self.cur_reference_motion['object_translation'][-1]
        except:
            self.traj_step = 0
            self._step = 0
        
        self.reset_internal()
        init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        if self.robot_name == "rm75_inspire_right":
            base_offset = np.asarray(getattr(self, "rm75_robot_base_offset", np.zeros(3)), dtype=np.float32)
            base_rpy = np.asarray(getattr(self, "rm75_robot_base_rpy", np.zeros(3)), dtype=np.float32)
            init_pos = init_pos + base_offset
        else:
            base_rpy = np.zeros(3, dtype=np.float32)
        init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(*base_rpy))
        self.robot.set_pose(init_pose)

        # Set robot qpos
        if vis_pregrasp:
            qpos = np.asarray(self._motion_file['robot_qpos'][self._motion_file['pregrasp_step']], dtype=np.float32)
            if len(qpos) != self.robot.dof:
                raise RuntimeError(
                    f"Pregrasp qpos has length {len(qpos)}, but robot {self.robot_name} has dof {self.robot.dof}. "
                    "Retarget the demonstration qpos before visualizing pregrasp."
                )
        else:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = xarm_qpos
            if self.robot_name == "allegro_hand_ur5" and self.robot.dof > 18:
                qpos[18] = 0.5
            elif self.robot_name == "rm75_inspire_right":
                try:
                    qpos = self._rm75_pregrasp_qpos()
                except Exception as exc:
                    print(f"[rm75] pregrasp IK initialization failed: {exc}")

        self.robot.set_qpos(qpos)
        zero_qvel = np.zeros(self.robot.dof, dtype=np.float32)
        if hasattr(self.robot, "set_qvel"):
            self.robot.set_qvel(zero_qvel)
        self.robot.set_drive_target(qpos)
        self.robot.set_drive_velocity_target(zero_qvel)

        # Set object pose
        try:
            object_pose = sapien.Pose(p=self.cur_reference_motion['object_translation'][0, :2].tolist() + [self.init_object_height], q=self.cur_reference_motion['object_orientation'][0].tolist())
            self.manipulated_object.set_pose(object_pose)
            self._zero_manipulated_object_velocity()
            if self.is_vision:
                self.target_object_pos = self.cur_reference_motion['object_translation'][-1]
            else:
                target_object_pose = sapien.Pose(p=self.cur_reference_motion['object_translation'][-1], q=self.cur_reference_motion['object_orientation'][-1])
                self.target_object.set_pose(target_object_pose)
        except:
            rot_matrix = np.array([[np.cos(traj_rot), -np.sin(traj_rot), 0], [np.sin(traj_rot), np.cos(traj_rot), 0], [0, 0, 1]])
            cur_object_quat = Quaternion(matrix=(rot_matrix @ Quaternion(self.init_object_quat).rotation_matrix)).elements
            object_pose = sapien.Pose(p=[self.init_x, self.init_y, self.init_object_height], q=cur_object_quat)
            self.manipulated_object.set_pose(object_pose)
            self._zero_manipulated_object_velocity()
            self.target_object_pos = np.array([self.init_x, self.init_y, 0.2])

        self.base_frame_pos = np.zeros(3)

        try:
            cur_robot_palm_pos = self.palm_link.get_pose().p
            self.start_step = np.argmin(np.linalg.norm(self.cur_reference_motion['robot_pregrasp_jpos'][:, 0, :] - cur_robot_palm_pos, axis=1))
            self.pregrasp_steps = 15
            self.imitate_steps = self.pregrasp_steps + len(self.cur_reference_motion['object_translation']) - 1
            if self.task_name == "pour" or self.task_name == "place":
                constant_steps = 80
            else:
                constant_steps = 60
            rm75_force_steps = int(getattr(self, "rm75_force_imitate_steps", 0))
            rm75_min_steps = int(getattr(self, "rm75_min_imitate_steps", 0))
            if rm75_force_steps > 0:
                constant_steps = rm75_force_steps
            elif rm75_min_steps > 0:
                constant_steps = max(constant_steps, rm75_min_steps)
                
            if self.imitate_steps < constant_steps:
                self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'].tolist()
                self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'].tolist()
                self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'].tolist()
                self.cur_reference_motion['object_translation'] += [self.cur_reference_motion['object_translation'][-1]] * (constant_steps - self.imitate_steps)
                self.cur_reference_motion['object_orientation'] += [self.cur_reference_motion['object_orientation'][-1]] * (constant_steps - self.imitate_steps)
                self.cur_reference_motion['robot_jpos'] += [self.cur_reference_motion['robot_jpos'][-1]] * (constant_steps - self.imitate_steps)
                self.cur_reference_motion['object_translation'] = np.array(self.cur_reference_motion['object_translation'])
                self.cur_reference_motion['object_orientation'] = np.array(self.cur_reference_motion['object_orientation'])
                self.cur_reference_motion['robot_jpos'] = np.array(self.cur_reference_motion['robot_jpos'])
                self.imitate_steps = constant_steps
        except:
            self.imitate_steps = 60

        self.pregrasp_success = False
        self.hand_jpos_err = 0.3
        self.hand_mjpos_err = 0.0
        self.obj_com_err = 0.0
        self.obj_rot_err = 0.0
        self.object_lift = 0.0
        self.robot_object_contact = np.zeros(self.num_contact_groups, dtype=np.float32)
        self.no_contact_steps = 0

        post_reset_settle = getattr(self, "_post_reset_settle", None)
        if callable(post_reset_settle):
            post_reset_settle()

        if self.is_vision and not self.is_demo_rollout:
            test_state = self.get_test_state()
            return self._get_vision_obs_from_test_state(test_state)
        else:
            return self.get_observation()

    def is_done(self):
        if self.is_vision and not self.is_demo_rollout:
            if self.current_step >= self.imitate_steps:
                return True
            else:
                return False
        else:
            if not self.pregrasp_success:
                if self.current_step >= self.pregrasp_steps:
                    return True
                else:
                    self.traj_step += 1
                    return False
            else:
                contact_done, self.no_contact_steps = should_terminate_for_contact_loss(
                    self.robot_name,
                    bool(self.is_contact),
                    self.current_step,
                    self.pregrasp_steps,
                    getattr(self, "no_contact_steps", 0),
                )
                if self.obj_com_err >= 0.15 or contact_done:
                    return True
                else:
                    if self.current_step >= self.imitate_steps:
                        return True
                    else:
                        self.traj_step += 1
                        return False

    def is_success(self):
        if self.norm_traj:
            success_3 = np.linalg.norm(self.manipulated_object.get_pose().p - np.array([self.init_x, self.init_y, 0.2])) < 0.03
            success_10 = self.manipulated_object.get_pose().p[2] - self.init_object_height > 0.05
            self.robot_object_contact[:] = self._contact_groups()
            is_contact = self.robot_object_contact[0] and sum(self.robot_object_contact[1:]) >= 1
        else:
            if self.is_vision:
                success_3 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object_pos) < 0.03
                success_10 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object_pos) < 0.1
            else:
                success_3 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object.get_pose().p) < 0.03
                success_10 = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object.get_pose().p) < 0.1
            self.robot_object_contact[:] = self._contact_groups()
            is_contact = self.robot_object_contact[0] and sum(self.robot_object_contact[1:]) >= 1
        return (is_contact and success_3), (is_contact and success_10)

    @cached_property
    def obs_dim(self):
        if self.is_vision and not self.is_demo_rollout:
            return len(self.get_test_state())
        else:
            return len(self.get_oracle_state())

    @cached_property
    def horizon(self):
        return self.imitate_steps
