from abc import abstractmethod
from typing import Dict, Optional, Callable, List, Union, Tuple

import gym
import numpy as np
import sapien.core as sapien
import transforms3d

from hand_imitation.env.sim_env.base import BaseSimulationEnv
from hand_imitation.env.sim_env.constructor import add_default_scene_light
from hand_imitation.utils.kinematics_helper import PartialKinematicModel
from hand_imitation.utils.common_robot_utils import load_robot, generate_robot_hand_info, ArmRobotInfo
from hand_imitation.env.rl_env.pc_processing import process_pc
from hand_imitation.env.rl_env.reference_utils import (
    expand_rm75_active_hand_action,
    enforce_rm75_coupled_hand_qpos,
    enforce_rm75_mimic_hand_qpos,
    project_rm75_velocity_to_object,
    recover_hand_action,
)
from hand_imitation.utils.random_utils import np_random

VISUAL_OBS_RETURN_TORCH = False
MAX_DEPTH_RANGE = 2.5
gl2sapien = sapien.Pose(q=np.array([0.5, 0.5, -0.5, -0.5]))


def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action


class BaseRLEnv(BaseSimulationEnv, gym.Env):
    def __init__(self, use_gui=False, frame_skip=5, renderer: str = "sapien", **renderer_kwargs):
        # Do not write any meaningful in this __init__ function other than type definition,
        # Since multiple parents are presented for the child RLEnv class
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        # Visual staff for offscreen rendering
        self.camera_infos: Dict[str, Dict] = {}
        self.camera_pose_noise: Dict[
            str, Tuple[Optional[float], sapien.Pose]] = {}  # tuple for noise level and original camera pose
        self.imagination_infos: Dict[str, float] = {}
        self.imagination_data: Dict[str, Dict[str, Tuple[sapien.ActorBase, np.ndarray, int]]] = {}
        self.need_flush_when_change_instance = False
        self.imaginations: Dict[str, np.ndarray] = {}
        self.eval_cam_names = renderer_kwargs['eval_cam_names'] if renderer_kwargs.__contains__("eval_cam_names") else None
        self.use_history_obs = renderer_kwargs['use_history_obs'] if renderer_kwargs.__contains__('use_history_obs') else False
        self.last_obs = None

        # RL related attributes
        self.is_robot_free: Optional[bool] = None
        self.arm_dof: Optional[int] = None
        self.rl_step: Optional[Callable] = None
        self.get_observation: Optional[Callable] = None
        self.robot_collision_links: Optional[List[sapien.Actor]] = None
        self.robot_info: Optional[Union[ArmRobotInfo]] = None
        self.velocity_limit: Optional[np.ndarray] = None
        self.kinematic_model: Optional[PartialKinematicModel] = None

        # Robot cache
        self.control_time_step = None
        self.ee_link_name = None
        self.ee_link: Optional[sapien.Actor] = None
        self.cartesian_error = None

        self.pregrasp_success = False
        self.pregrasp_steps = 10
        self.imitate_steps = 60

        self.hand_jpos_err = 0.3
        self.hand_mjpos_err = 0.0
        self.obj_com_err = 0.0
        self.obj_rot_err = 0.0
        self.object_lift = 0.0
        self.robot_object_contact = np.zeros(1, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, action):
        pass

    def get_info(self):
        info = dict()
        info['pregrasp_success'] = self.pregrasp_success
        info['pregrasp_steps'] = self.pregrasp_steps
        info['imitate_steps'] = self.imitate_steps
        info['hand_jpos_err'] = self.hand_jpos_err
        info['hand_mjpos_err'] = self.hand_mjpos_err
        info['obj_com_err'] = self.obj_com_err
        info['obj_rot_err'] = self.obj_rot_err
        info['obj_lift'] = self.object_lift
        info['contact_count'] = float(np.sum(self.robot_object_contact))
        info['stable_grasp_contact'] = bool(getattr(self, 'stable_grasp_contact', False))
        info['stage'] = self._stage
        info['control_error'] = self.cartesian_error
        if self.is_vision:
            info['obj_tgt_dist'] = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object_pos)
        else:
            info['obj_tgt_dist'] = np.linalg.norm(self.manipulated_object.get_pose().p - self.target_object.get_pose().p)

        return info

    def update_cached_state(self):
        return

    @abstractmethod
    def is_done(self):
        pass

    @property
    @abstractmethod
    def obs_dim(self):
        return 0

    @property
    def action_dim(self):
        return 6 + self.robot_info.hand_dof

    @property
    @abstractmethod
    def horizon(self):
        return 0

    def setup(self, robot_name):
        self.robot_name = robot_name
        # RM75/RH56 collision meshes can self-intersect in SAPIEN before the
        # finger drives move, which locks the middle fingers in place.  Disable
        # robot self-collision for this platform and keep object contacts active.
        disable_self_collision = robot_name == "rm75_inspire_right"
        self.robot = load_robot(self.scene, robot_name, disable_self_collision=disable_self_collision)
        self.robot.set_pose(sapien.Pose(np.array([0, 0, -5])))

        info = generate_robot_hand_info()[robot_name]
        self.arm_dof = info.arm_dof
        hand_dof = info.hand_dof
        velocity_limit = np.array([1] * 3 + [1] * 3 + [np.pi] * hand_dof)
        self.velocity_limit = np.stack([-velocity_limit, velocity_limit], axis=1)
        start_joint_name = self.robot.get_joints()[1].get_name()
        end_joint_name = self.robot.get_active_joints()[self.arm_dof - 1].get_name()
        self.kinematic_model = PartialKinematicModel(self.robot, start_joint_name, end_joint_name)
        self.ee_link_name = self.kinematic_model.end_link_name
        self.ee_link = [link for link in self.robot.get_links() if link.get_name() == self.ee_link_name][0]

        self.robot_info = info
        self.robot_collision_links = [link for link in self.robot.get_links() if len(link.get_collision_shapes()) > 0]
        self.control_time_step = self.scene.get_timestep() * self.frame_skip

        self.rl_step = self.arm_sim_step

        if self.is_vision and not self.is_demo_rollout:
            self.get_observation = self.get_test_state 
        else:
            self.get_observation = self.get_oracle_state 

    def arm_sim_step(self, action: np.ndarray, check=False, denormalized=False):
        if denormalized:
            target_qpos = action
            target_qvel = np.zeros_like(target_qpos)
            self.robot.set_drive_target(target_qpos)
            self.robot.set_drive_velocity_target(target_qvel)
            if check:
                self.robot_sim_obs = [self.robot.get_qpos().tolist()]

            for i in range(self.frame_skip):
                self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
                self.scene.step()
                if check and i < self.frame_skip - 1:
                    self.robot_sim_obs.append(self.robot.get_qpos().tolist())
            self.current_step += 1
            self.cartesian_error = 0.0
        else:
            current_qpos = self.robot.get_qpos()
            ee_link_last_pose = self.ee_link.get_pose()
            action = np.clip(action, -1, 1)
            if self.robot_name == "rm75_inspire_right":
                if bool(getattr(self, "rm75_scripted_action_prior", False)):
                    scripted = np.zeros_like(action, dtype=np.float32)
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    approach_steps = max(int(getattr(self, "rm75_scripted_action_prior_approach_steps", 14)), 1)
                    close_steps = max(int(getattr(self, "rm75_scripted_action_prior_close_steps", 30)), 1)
                    grasp_offset = np.asarray(
                        getattr(self, "rm75_scripted_action_prior_grasp_offset", np.zeros(3, dtype=np.float32)),
                        dtype=np.float32,
                    )
                    if grasp_offset.shape != (3,):
                        grasp_offset = np.zeros(3, dtype=np.float32)
                    offset_pregrasp = bool(
                        getattr(self, "rm75_scripted_action_prior_offset_pregrasp", False)
                    )
                    if int(getattr(self, "current_step", 0)) <= int(getattr(self, "pregrasp_steps", 0)):
                        target_palm = np.asarray(self.cur_reference_motion["robot_pregrasp_jpos"][-1, 0], dtype=np.float32)
                        if offset_pregrasp:
                            target_palm = target_palm + grasp_offset
                        close_alpha = 0.0
                    else:
                        hold_steps = max(int(getattr(self, "rm75_scripted_action_prior_hold_steps", 36)), 0)
                        lift_steps = max(int(getattr(self, "rm75_scripted_action_prior_lift_steps", 140)), 1)
                        grasp_palm = (
                            np.asarray(self.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32)
                            + grasp_offset
                        )
                        follow_reference_approach = bool(
                            getattr(self, "rm75_scripted_action_prior_follow_reference_approach", False)
                        )
                        if follow_reference_approach:
                            try:
                                approach_delta = (
                                    np.asarray(
                                        self.cur_reference_motion["robot_pregrasp_jpos"][-1, 0],
                                        dtype=np.float32,
                                    )
                                    + (grasp_offset if offset_pregrasp else 0.0)
                                    - grasp_palm
                                )
                            except Exception:
                                approach_delta = np.asarray(
                                    getattr(self, "rm75_default_approach_delta", np.zeros(3, dtype=np.float32)),
                                    dtype=np.float32,
                                )
                        else:
                            approach_delta = (
                                np.asarray(
                                    getattr(self, "rm75_default_approach_delta", np.zeros(3, dtype=np.float32)),
                                    dtype=np.float32,
                                )
                            )
                        if approach_delta.shape != (3,):
                            approach_delta = np.zeros(3, dtype=np.float32)
                        approach_alpha = float(np.clip(post_step / approach_steps, 0.0, 1.0))
                        approach_z_power = max(
                            float(getattr(self, "rm75_scripted_action_prior_approach_z_power", 1.0)),
                            1e-6,
                        )
                        approach_alpha_vec = np.full(3, approach_alpha, dtype=np.float32)
                        approach_alpha_vec[2] = float(np.clip(approach_alpha ** approach_z_power, 0.0, 1.0))
                        close_alpha = float(np.clip((post_step - approach_steps) / close_steps, 0.0, 1.0))
                        lift_start = approach_steps + close_steps + hold_steps
                        lift_alpha = float(np.clip((post_step - lift_start) / lift_steps, 0.0, 1.0))
                        target_palm = grasp_palm + (1.0 - approach_alpha_vec) * approach_delta
                        target_palm = target_palm.copy()
                        prior_mode = str(getattr(self, "rm75_scripted_action_prior_mode", "lift"))
                        stable_now = bool(getattr(self, "stable_grasp_contact", False))
                        if stable_now:
                            self.rm75_scripted_action_prior_target_latched = True
                        target_track_allowed = (
                            prior_mode == "target"
                            and post_step >= lift_start
                            and bool(getattr(self, "rm75_scripted_action_prior_target_latched", False))
                        )
                        if bool(getattr(self, "rm75_scripted_action_prior_target_require_stable", False)):
                            target_track_allowed = target_track_allowed and stable_now
                        if target_track_allowed:
                            try:
                                object_pos = np.asarray(self.manipulated_object.get_pose().p, dtype=np.float32)
                                target_object = getattr(self, "target_object", None)
                                if target_object is not None:
                                    target_pos = np.asarray(target_object.get_pose().p, dtype=np.float32)
                                else:
                                    target_pos = np.asarray(
                                        getattr(self, "target_object_pos", getattr(self, "_final_goal", object_pos)),
                                        dtype=np.float32,
                                    )
                                object_error = target_pos - object_pos
                                z_deadband = float(
                                    getattr(self, "rm75_scripted_action_prior_target_z_deadband", 0.002)
                                )
                                if abs(float(object_error[2])) < z_deadband:
                                    object_error[2] = 0.0
                                target_step = object_error * float(
                                    getattr(self, "rm75_scripted_action_prior_target_gain", 0.75)
                                )
                                max_xy = max(
                                    float(getattr(self, "rm75_scripted_action_prior_target_max_xy_step", 0.006)),
                                    0.0,
                                )
                                max_z = max(
                                    float(getattr(self, "rm75_scripted_action_prior_target_max_z_step", 0.006)),
                                    0.0,
                                )
                                target_step[:2] = np.clip(target_step[:2], -max_xy, max_xy)
                                target_step[2] = np.clip(target_step[2], -max_z, max_z)
                                palm_now = np.asarray(
                                    getattr(self, "palm_link", self.ee_link).get_pose().p,
                                    dtype=np.float32,
                                )
                                target_palm = palm_now + target_step
                            except Exception:
                                target_track_allowed = False
                        if not target_track_allowed:
                            if prior_mode == "target":
                                pre_lift_height = min(
                                    max(
                                        float(
                                            getattr(
                                                self,
                                                "rm75_scripted_action_prior_target_pre_lift_height",
                                                0.0,
                                            )
                                        ),
                                        0.0,
                                    ),
                                    0.02,
                                )
                                target_palm[2] += pre_lift_height * lift_alpha
                            else:
                                target_palm[2] += float(
                                    getattr(self, "rm75_scripted_action_prior_lift_height", 0.18)
                                ) * lift_alpha
                    palm_pos = np.asarray(getattr(self, "palm_link", self.ee_link).get_pose().p, dtype=np.float32)
                    pos_gain = max(float(getattr(self, "rm75_scripted_action_prior_pos_gain", 0.04)), 1e-6)
                    scripted[:3] = np.clip((target_palm - palm_pos) / pos_gain, -1.0, 1.0)
                    thumb_delay_steps = max(
                        int(getattr(self, "rm75_scripted_action_prior_thumb_delay_steps", 0)),
                        0,
                    )
                    thumb_close_steps = int(getattr(self, "rm75_scripted_action_prior_thumb_close_steps", 0))
                    if thumb_close_steps <= 0:
                        thumb_close_steps = close_steps
                    thumb_alpha = float(
                        np.clip((post_step - approach_steps - thumb_delay_steps) / max(thumb_close_steps, 1), 0.0, 1.0)
                    )
                    finger_close = float(getattr(self, "rm75_scripted_action_prior_finger_close", 0.75))
                    active = np.asarray(
                        [
                            thumb_alpha * float(getattr(self, "rm75_scripted_action_prior_thumb_yaw", 0.90)),
                            thumb_alpha * float(getattr(self, "rm75_scripted_action_prior_thumb_pitch", 0.85)),
                            close_alpha * finger_close,
                            close_alpha * finger_close,
                            close_alpha * finger_close,
                            close_alpha * float(getattr(self, "rm75_scripted_action_prior_pinky_close", 0.75)),
                        ],
                        dtype=np.float32,
                    )
                    hand_action_dim_for_prior = max(int(action.shape[0]) - 6, 0)
                    if hand_action_dim_for_prior == 12 or (
                        bool(getattr(self, "rm75_native_hand_control", False)) and hand_action_dim_for_prior > 0
                    ):
                        scripted[6:] = expand_rm75_active_hand_action(active, hand_action_dim_for_prior)
                    elif hand_action_dim_for_prior == 6:
                        scripted[6:] = active
                    blend = float(np.clip(getattr(self, "rm75_scripted_action_prior_blend", 1.0), 0.0, 1.0))
                    action = np.clip((1.0 - blend) * action + blend * scripted, -1.0, 1.0)
                arm_scale = float(getattr(self, "rm75_arm_action_scale", 1.0))
                hand_scale = float(getattr(self, "rm75_hand_action_scale", 1.0))
                pregrasp_safe_active = bool(getattr(self, "rm75_pregrasp_safe_action", False)) and (
                    int(getattr(self, "current_step", 0))
                    <= int(getattr(self, "pregrasp_steps", 0))
                    + int(getattr(self, "rm75_pregrasp_safe_margin_steps", 0))
                )
                if pregrasp_safe_active:
                    arm_scale = float(getattr(self, "rm75_pregrasp_arm_action_scale", arm_scale))
                    hand_scale = float(getattr(self, "rm75_pregrasp_hand_action_scale", hand_scale))
                elif int(getattr(self, "rm75_post_pregrasp_arm_ramp_steps", 0)) > 0:
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    ramp_steps = max(int(getattr(self, "rm75_post_pregrasp_arm_ramp_steps", 0)), 1)
                    if 0 <= post_step < ramp_steps:
                        ramp = float(np.clip(post_step / ramp_steps, 0.0, 1.0))
                        start_scale = float(getattr(self, "rm75_post_pregrasp_arm_scale_start", arm_scale))
                        end_scale = float(getattr(self, "rm75_post_pregrasp_arm_scale_end", arm_scale))
                        arm_scale = (1.0 - ramp) * start_scale + ramp * end_scale
                if bool(getattr(self, "rm75_post_pregrasp_hold_until_stable", False)):
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    required_hold = max(
                        int(getattr(self, "rm75_post_pregrasp_required_stable_hold_steps", 2)),
                        1,
                    )
                    stable_hold = int(getattr(self, "rm75_stable_contact_hold_steps", 0))
                    if post_step >= 0 and stable_hold < required_hold:
                        arm_scale = min(
                            arm_scale,
                            float(getattr(self, "rm75_post_pregrasp_hold_arm_scale", 0.02)),
                        )
                project_to_object_active = False
                post_stable_lift_only_active = False
                post_contact_lift_assist_active = False
                post_contact_wrist_bias_active = False
                scripted_lift_prior_active = False
                native_hand_control = bool(getattr(self, "rm75_native_hand_control", False))
                hand_action_dim = max(int(action.shape[0]) - 6, 0)
                hand_bias_weights = np.asarray(
                    getattr(self, "rm75_hand_close_bias_weights", np.ones(6, dtype=np.float32)),
                    dtype=np.float32,
                )
                if hand_bias_weights.shape != (6,):
                    hand_bias_weights = np.ones(6, dtype=np.float32)
                if pregrasp_safe_active:
                    active_hand_bias = (
                        float(getattr(self, "rm75_pregrasp_hand_close_bias", 0.0)) * hand_bias_weights
                    )
                else:
                    active_hand_bias = float(getattr(self, "rm75_hand_close_bias", 0.0)) * hand_bias_weights
                    phase_close_bias = float(getattr(self, "rm75_hand_phase_close_bias", 0.0))
                    if phase_close_bias > 0.0:
                        post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                        phase_start = int(getattr(self, "rm75_hand_phase_close_start_step", 0))
                        phase_ramp = max(int(getattr(self, "rm75_hand_phase_close_ramp_steps", 1)), 1)
                        phase_alpha = float(np.clip((post_step - phase_start) / phase_ramp, 0.0, 1.0))
                        active_hand_bias = np.maximum(
                            active_hand_bias,
                            phase_alpha * phase_close_bias * hand_bias_weights,
                        )
                if (not pregrasp_safe_active) and hasattr(self, "rm75_dynamic_hand_close_bias"):
                    active_hand_bias = active_hand_bias + np.asarray(
                        self.rm75_dynamic_hand_close_bias(),
                        dtype=np.float32,
                    )
                if native_hand_control or hand_action_dim != active_hand_bias.shape[0]:
                    hand_bias = expand_rm75_active_hand_action(active_hand_bias, hand_action_dim)
                else:
                    hand_bias = active_hand_bias
                action = action.copy()
                action[:6] = np.clip(action[:6] * arm_scale, -1.0, 1.0)
                angular_bias = np.asarray(
                    getattr(self, "rm75_wrist_angular_bias", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                )
                if angular_bias.shape == (3,):
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    if (
                        post_step >= int(getattr(self, "rm75_wrist_angular_bias_start_step", 0))
                        and post_step <= int(getattr(self, "rm75_wrist_angular_bias_end_step", 1000000))
                    ):
                        action[3:6] = np.clip(action[3:6] + angular_bias, -1.0, 1.0)
                post_contact_wrist_bias = np.asarray(
                    getattr(self, "rm75_post_contact_wrist_bias", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                )
                if post_contact_wrist_bias.shape == (3,) and np.any(np.abs(post_contact_wrist_bias) > 1e-8):
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    contact_hold = int(getattr(self, "rm75_contact_hold_steps", 0))
                    min_hold = max(int(getattr(self, "rm75_post_contact_wrist_min_hold_steps", 1)), 1)
                    min_non_thumb = max(
                        int(getattr(self, "rm75_post_contact_wrist_min_non_thumb_contacts", 0)),
                        0,
                    )
                    thumb_ok = (
                        not bool(getattr(self, "rm75_post_contact_wrist_requires_thumb", True))
                        or bool(getattr(self, "rm75_thumb_contact", False))
                    )
                    non_thumb_ok = int(getattr(self, "rm75_non_thumb_contact_count", 0)) >= min_non_thumb
                    if post_step >= 0 and contact_hold >= min_hold and thumb_ok and non_thumb_ok:
                        action[3:6] = np.clip(action[3:6] + post_contact_wrist_bias, -1.0, 1.0)
                        post_contact_wrist_bias_active = True
                if (
                    bool(getattr(self, "rm75_scripted_lift_prior", False))
                    and not pregrasp_safe_active
                ):
                    try:
                        post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                        start_step = int(getattr(self, "rm75_scripted_lift_prior_start_step", 0))
                        if post_step >= start_step:
                            reference = np.asarray(self.cur_reference_motion["robot_jpos"], dtype=np.float32)
                            if str(getattr(self, "rm75_scripted_lift_prior_track_mode", "initial")) == "current":
                                ref_idx = min(max(post_step, 0), reference.shape[0] - 1)
                            else:
                                ref_idx = 0
                            grasp_palm = reference[ref_idx, 0].copy()
                            approach_steps = max(
                                int(getattr(self, "rm75_scripted_lift_prior_approach_steps", 0)),
                                0,
                            )
                            follow_reference_approach = bool(
                                getattr(self, "rm75_scripted_lift_prior_follow_reference_approach", False)
                            )
                            if follow_reference_approach:
                                try:
                                    approach_delta = (
                                        np.asarray(
                                            self.cur_reference_motion["robot_pregrasp_jpos"][-1, 0],
                                            dtype=np.float32,
                                        )
                                        - np.asarray(self.cur_reference_motion["robot_jpos"][0, 0], dtype=np.float32)
                                    )
                                except Exception:
                                    approach_delta = np.asarray(
                                        getattr(self, "rm75_default_approach_delta", np.zeros(3, dtype=np.float32)),
                                        dtype=np.float32,
                                    )
                            else:
                                approach_delta = (
                                    np.asarray(
                                        getattr(self, "rm75_default_approach_delta", np.zeros(3, dtype=np.float32)),
                                        dtype=np.float32,
                                    )
                                )
                            if approach_delta.shape != (3,):
                                approach_delta = np.zeros(3, dtype=np.float32)
                            if approach_steps > 0:
                                approach_alpha = float(np.clip(post_step / approach_steps, 0.0, 1.0))
                                target_palm = grasp_palm + (1.0 - approach_alpha) * approach_delta
                            else:
                                target_palm = grasp_palm
                            close_steps = max(int(getattr(self, "rm75_scripted_lift_prior_close_steps", 18)), 1)
                            hold_steps = max(int(getattr(self, "rm75_scripted_lift_prior_hold_steps", 10)), 0)
                            lift_steps = max(int(getattr(self, "rm75_scripted_lift_prior_lift_steps", 45)), 1)
                            lift_start = approach_steps + close_steps + hold_steps
                            lift_alpha = float(np.clip((post_step - lift_start) / lift_steps, 0.0, 1.0))
                            target_palm[2] += float(getattr(self, "rm75_scripted_lift_prior_lift_height", 0.10)) * lift_alpha
                            palm_pos = np.asarray(getattr(self, "palm_link", self.ee_link).get_pose().p, dtype=np.float32)
                            pos_gain = max(float(getattr(self, "rm75_scripted_lift_prior_pos_gain", 0.04)), 1e-6)
                            prior_linear = np.clip((target_palm - palm_pos) / pos_gain, -1.0, 1.0)
                            blend = float(np.clip(getattr(self, "rm75_scripted_lift_prior_blend", 1.0), 0.0, 1.0))
                            action[:3] = np.clip((1.0 - blend) * action[:3] + blend * prior_linear, -1.0, 1.0)
                            if bool(getattr(self, "rm75_scripted_lift_prior_zero_angular", True)):
                                action[3:6] = np.clip((1.0 - blend) * action[3:6], -1.0, 1.0)
                            scripted_lift_prior_active = True
                    except Exception:
                        scripted_lift_prior_active = False
                if (
                    bool(getattr(self, "rm75_post_pregrasp_project_to_object", False))
                    and not pregrasp_safe_active
                ):
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    stable_hold = int(getattr(self, "rm75_stable_contact_hold_steps", 0))
                    required_hold = max(
                        int(getattr(self, "rm75_post_pregrasp_required_stable_hold_steps", 2)),
                        1,
                    )
                    project_until_stable = bool(getattr(self, "rm75_post_pregrasp_project_until_stable", True))
                    should_project = (
                        post_step >= 0
                        and (not project_until_stable or stable_hold < required_hold)
                    )
                    if should_project:
                        try:
                            object_pos = np.asarray(self.manipulated_object.get_pose().p, dtype=np.float32)
                            palm_pos = np.asarray(self.ee_link.get_pose().p, dtype=np.float32)
                            target_velocity = recover_action(action[:6], self.velocity_limit[:6])
                            projected_linear_velocity = project_rm75_velocity_to_object(
                                target_velocity[:3],
                                palm_pos,
                                object_pos,
                                tangent_scale=getattr(self, "rm75_post_pregrasp_project_tangent_scale", 0.18),
                                max_approach_speed=getattr(
                                    self, "rm75_post_pregrasp_project_max_approach_speed", 0.055
                                ),
                                max_retreat_speed=getattr(
                                    self, "rm75_post_pregrasp_project_max_retreat_speed", 0.012
                                ),
                                approach_bias=getattr(
                                    self, "rm75_post_pregrasp_project_approach_bias", 0.0
                                ),
                            )
                            linear_limit = self.velocity_limit[:3]
                            action[:3] = np.clip(
                                (projected_linear_velocity - linear_limit[:, 0])
                                / (linear_limit[:, 1] - linear_limit[:, 0])
                                * 2.0
                                - 1.0,
                                -1.0,
                                1.0,
                            )
                            project_to_object_active = True
                        except Exception:
                            project_to_object_active = False
                if (
                    bool(getattr(self, "rm75_post_stable_lift_only", False))
                    and not pregrasp_safe_active
                ):
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    required_hold = max(
                        int(getattr(self, "rm75_post_pregrasp_required_stable_hold_steps", 2)),
                        1,
                    )
                    stable_hold = int(getattr(self, "rm75_stable_contact_hold_steps", 0))
                    stable_ready = post_step >= 0 and stable_hold >= required_hold
                    latch_steps = max(int(getattr(self, "rm75_post_lift_latch_steps", 0)), 0)
                    latch_left = max(int(getattr(self, "rm75_lift_latch_steps_left", 0)), 0)
                    if stable_ready and latch_steps > 0:
                        latch_left = max(latch_left, latch_steps)
                        self.rm75_lift_latch_steps_left = latch_left
                    latch_active = latch_left > 0
                    if post_step >= 0 and (stable_ready or latch_active):
                        target_velocity = recover_action(action[:6], self.velocity_limit[:6])
                        max_xy = max(float(getattr(self, "rm75_post_stable_max_xy_speed", 0.012)), 0.0)
                        max_down = max(float(getattr(self, "rm75_post_stable_max_down_speed", 0.002)), 0.0)
                        max_up = max(float(getattr(self, "rm75_post_stable_max_up_speed", 0.035)), 0.0)
                        lift_bias = float(getattr(self, "rm75_post_stable_lift_bias", 0.0))
                        max_angular = max(float(getattr(self, "rm75_post_stable_max_angular_speed", 0.20)), 0.0)
                        drift_gain = max(
                            float(getattr(self, "rm75_post_stable_xy_drift_correction_gain", 0.0)),
                            0.0,
                        )
                        drift_max_speed = max(
                            float(getattr(self, "rm75_post_stable_xy_drift_correction_max_speed", 0.0)),
                            0.0,
                        )
                        target_velocity[:2] = np.clip(target_velocity[:2], -max_xy, max_xy)
                        if drift_gain > 0.0 and drift_max_speed > 0.0:
                            drift_vec = np.asarray(
                                getattr(self, "object_xy_drift_vec", np.zeros(2, dtype=np.float32)),
                                dtype=np.float32,
                            )
                            if drift_vec.shape[0] >= 2:
                                correction = np.clip(
                                    -drift_gain * drift_vec[:2],
                                    -drift_max_speed,
                                    drift_max_speed,
                                )
                                target_velocity[:2] = np.clip(
                                    target_velocity[:2] + correction,
                                    -max_xy,
                                    max_xy,
                                )
                        target_velocity[2] = np.clip(target_velocity[2] + lift_bias, -max_down, max_up)
                        target_velocity[3:6] = np.clip(target_velocity[3:6], -max_angular, max_angular)
                        velocity_limit = self.velocity_limit[:6]
                        denom = np.maximum(velocity_limit[:, 1] - velocity_limit[:, 0], 1e-6)
                        action[:6] = np.clip(
                            (target_velocity - velocity_limit[:, 0]) / denom * 2.0 - 1.0,
                            -1.0,
                            1.0,
                        )
                        post_stable_lift_only_active = True
                        if latch_active and not stable_ready:
                            self.rm75_lift_latch_steps_left = max(latch_left - 1, 0)
                if (
                    bool(getattr(self, "rm75_post_contact_lift_assist", False))
                    and not post_stable_lift_only_active
                    and not pregrasp_safe_active
                ):
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    contact_hold = int(getattr(self, "rm75_contact_hold_steps", 0))
                    min_hold = max(int(getattr(self, "rm75_post_contact_lift_min_hold_steps", 1)), 0)
                    min_non_thumb = max(
                        int(getattr(self, "rm75_post_contact_lift_min_non_thumb_contacts", 0)),
                        0,
                    )
                    thumb_ok = (
                        not bool(getattr(self, "rm75_post_contact_lift_requires_thumb", True))
                        or bool(getattr(self, "rm75_thumb_contact", False))
                    )
                    non_thumb_ok = int(getattr(self, "rm75_non_thumb_contact_count", 0)) >= min_non_thumb
                    if post_step >= 0 and contact_hold >= min_hold and thumb_ok and non_thumb_ok:
                        target_velocity = recover_action(action[:6], self.velocity_limit[:6])
                        max_xy = max(float(getattr(self, "rm75_post_contact_max_xy_speed", 0.014)), 0.0)
                        max_down = max(float(getattr(self, "rm75_post_contact_max_down_speed", 0.002)), 0.0)
                        max_up = max(float(getattr(self, "rm75_post_contact_max_up_speed", 0.055)), 0.0)
                        lift_bias = float(getattr(self, "rm75_post_contact_lift_bias", 0.0))
                        max_angular = max(float(getattr(self, "rm75_post_contact_max_angular_speed", 0.16)), 0.0)
                        drift_gain = max(
                            float(getattr(self, "rm75_post_contact_xy_drift_correction_gain", 0.0)),
                            0.0,
                        )
                        drift_max_speed = max(
                            float(getattr(self, "rm75_post_contact_xy_drift_correction_max_speed", 0.0)),
                            0.0,
                        )
                        target_velocity[:2] = np.clip(target_velocity[:2], -max_xy, max_xy)
                        if drift_gain > 0.0 and drift_max_speed > 0.0:
                            drift_vec = np.asarray(
                                getattr(self, "object_xy_drift_vec", np.zeros(2, dtype=np.float32)),
                                dtype=np.float32,
                            )
                            if drift_vec.shape[0] >= 2:
                                correction = np.clip(
                                    -drift_gain * drift_vec[:2],
                                    -drift_max_speed,
                                    drift_max_speed,
                                )
                                target_velocity[:2] = np.clip(
                                    target_velocity[:2] + correction,
                                    -max_xy,
                                    max_xy,
                                )
                        target_velocity[2] = np.clip(target_velocity[2] + lift_bias, -max_down, max_up)
                        target_velocity[3:6] = np.clip(target_velocity[3:6], -max_angular, max_angular)
                        velocity_limit = self.velocity_limit[:6]
                        denom = np.maximum(velocity_limit[:, 1] - velocity_limit[:, 0], 1e-6)
                        action[:6] = np.clip(
                            (target_velocity - velocity_limit[:, 0]) / denom * 2.0 - 1.0,
                            -1.0,
                            1.0,
                        )
                        post_contact_lift_assist_active = True
                if pregrasp_safe_active and bool(getattr(self, "rm75_pregrasp_zero_hand_action", False)):
                    action[6:] = np.clip(hand_bias, -1.0, 1.0)
                else:
                    action[6:] = np.clip(action[6:] * hand_scale + hand_bias, -1.0, 1.0)
                if bool(getattr(self, "rm75_scripted_hand_prior", False)) and not pregrasp_safe_active:
                    post_step = int(getattr(self, "current_step", 0)) - int(getattr(self, "pregrasp_steps", 0))
                    approach_steps = max(int(getattr(self, "rm75_scripted_hand_prior_approach_steps", 14)), 0)
                    close_steps = max(int(getattr(self, "rm75_scripted_hand_prior_close_steps", 30)), 1)
                    close_alpha = float(np.clip((post_step - approach_steps) / close_steps, 0.0, 1.0))
                    active = close_alpha * np.asarray(
                        [
                            float(getattr(self, "rm75_scripted_hand_prior_thumb_yaw", 0.90)),
                            float(getattr(self, "rm75_scripted_hand_prior_thumb_pitch", 0.85)),
                            float(getattr(self, "rm75_scripted_hand_prior_finger_close", 0.75)),
                            float(getattr(self, "rm75_scripted_hand_prior_finger_close", 0.75)),
                            float(getattr(self, "rm75_scripted_hand_prior_finger_close", 0.75)),
                            float(getattr(self, "rm75_scripted_hand_prior_pinky_close", 0.75)),
                        ],
                        dtype=np.float32,
                    )
                    if native_hand_control or hand_action_dim != active.shape[0]:
                        prior_hand = expand_rm75_active_hand_action(active, hand_action_dim)
                    else:
                        prior_hand = active
                    blend = float(np.clip(getattr(self, "rm75_scripted_hand_prior_blend", 1.0), 0.0, 1.0))
                    action[6:] = np.clip((1.0 - blend) * action[6:] + blend * prior_hand, -1.0, 1.0)
                thumb_action_cap = getattr(self, "rm75_thumb_action_cap_until_non_thumb", None)
                if thumb_action_cap is not None:
                    cap = float(np.clip(thumb_action_cap, -1.0, 1.0))
                    release_non_thumb = max(
                        int(getattr(self, "rm75_thumb_action_cap_release_non_thumb_contacts", 2)),
                        0,
                    )
                    release_hold = max(
                        int(getattr(self, "rm75_thumb_action_cap_release_contact_hold_steps", 1)),
                        0,
                    )
                    non_thumb_ready = int(getattr(self, "rm75_non_thumb_contact_count", 0)) >= release_non_thumb
                    hold_ready = int(getattr(self, "rm75_contact_hold_steps", 0)) >= release_hold
                    if not (non_thumb_ready and hold_ready):
                        if hand_action_dim == 12:
                            action[6:8] = np.minimum(action[6:8], cap)
                        elif hand_action_dim == 6:
                            action[6:8] = np.minimum(action[6:8], cap)
                if bool(getattr(self, "rm75_disable_pinky_action", False)) and hand_action_dim > 0:
                    pinky_value = float(np.clip(getattr(self, "rm75_pinky_action_value", 0.0), -1.0, 1.0))
                    if hand_action_dim == 12:
                        action[16:18] = pinky_value
                    elif hand_action_dim == 6:
                        action[11] = pinky_value
                self.rm75_pregrasp_safe_active = pregrasp_safe_active
                self.rm75_last_effective_arm_scale = arm_scale
                self.rm75_last_effective_hand_scale = hand_scale
                self.rm75_last_project_to_object_active = project_to_object_active
                self.rm75_last_post_stable_lift_only_active = post_stable_lift_only_active
                self.rm75_last_post_contact_lift_assist_active = post_contact_lift_assist_active
                self.rm75_last_post_contact_wrist_bias_active = post_contact_wrist_bias_active
                self.rm75_last_scripted_lift_prior_active = scripted_lift_prior_active
            target_root_velocity = recover_action(action[:6], self.velocity_limit[:6])
            palm_jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(current_qpos[:self.arm_dof])
            arm_qvel = compute_inverse_kinematics(target_root_velocity, palm_jacobian)[:self.arm_dof]
            arm_qvel = np.clip(arm_qvel, -np.pi / 1, np.pi / 1)
            arm_qpos = arm_qvel * self.control_time_step + self.robot.get_qpos()[:self.arm_dof]
            hand_qpos = recover_hand_action(action[6:], self.robot.get_qlimits()[self.arm_dof:], self.robot_name)
            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            target_qvel = np.zeros_like(target_qpos)
            target_qvel[:self.arm_dof] = arm_qvel
            self.robot.set_drive_target(target_qpos)
            self.robot.set_drive_velocity_target(target_qvel)
            if check:
                self.robot_sim_obs = [self.robot.get_qpos().tolist()]

            for i in range(self.frame_skip):
                if self.robot_name == "rm75_inspire_right":
                    if bool(getattr(self, "rm75_native_hand_control", False)):
                        if bool(getattr(self, "rm75_enforce_native_mimic_qpos", False)):
                            qpos = enforce_rm75_mimic_hand_qpos(
                                self.robot.get_qpos(),
                                self.arm_dof,
                                self.robot.get_qlimits()[self.arm_dof:],
                            )
                            self.robot.set_qpos(qpos)
                    else:
                        qpos = enforce_rm75_coupled_hand_qpos(
                            self.robot.get_qpos(),
                            self.arm_dof,
                            self.robot.get_qlimits()[self.arm_dof:],
                        )
                        self.robot.set_qpos(qpos)
                self.robot.set_qf(self.robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
                self.scene.step()
                if check and i < self.frame_skip - 1:
                    self.robot_sim_obs.append(self.robot.get_qpos().tolist())
            self.current_step += 1

            ee_link_new_pose = self.ee_link.get_pose()
            relative_pos = ee_link_new_pose.p - ee_link_last_pose.p
            self.cartesian_error = np.linalg.norm(relative_pos - target_root_velocity[:3] * self.control_time_step)

    def reset_internal(self):
        self.current_step = 0
        if self.init_state is not None:
            self.scene.unpack(self.init_state)
        self.reset_env()
        if self.init_state is None:
            self.init_state = self.scene.pack()

    def step(self, action: np.ndarray, check=False, denormalized=False):
        self.rl_step(action, check, denormalized)
        if self.is_vision and not self.is_demo_rollout:
            test_state = self.get_test_state()
            obs = self._get_vision_obs_from_test_state(test_state)
        else:
            obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()
        # Reference: https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
        # Need to consider that is_done and timelimit can happen at the same time
        if self.current_step >= self.horizon:
            info["TimeLimit.truncated"] = not done
            done = True
        return obs, reward, done, info
    
    def setup_visual_obs_config(self, config: Dict[str, Dict]):
        for name, camera_cfg in config.items():
            if name not in self.cameras.keys():
                raise ValueError(f"Camera {name} not created. Existing {len(self.cameras)} cameras: {self.cameras.keys()}")
            self.camera_infos[name] = {}
            banned_modality_set = {"point_cloud", "depth"}
            if len(banned_modality_set.intersection(set(camera_cfg.keys()))) == len(banned_modality_set):
                raise RuntimeError(f"Request both point_cloud and depth for same camera is not allowed. Point cloud contains all information required by the depth.")

            # Add perturb for camera pose
            cam = self.cameras[name]
            if "pose_perturb_level" in camera_cfg:
                cam_pose_perturb = camera_cfg.pop("pose_perturb_level")
            else:
                cam_pose_perturb = None
            self.camera_pose_noise[name] = (cam_pose_perturb, cam.get_pose())

            for modality, cfg in camera_cfg.items():
                if modality == "point_cloud":
                    if "num_points" not in cfg:
                        raise RuntimeError(f"Missing num_points in camera {name} point_cloud config.")

                self.camera_infos[name][modality] = cfg

        modality = []
        for camera_cfg in config.values():
            modality.extend(camera_cfg.keys())
        modality_set = set(modality)

    def get_robot_state(self):
        raise NotImplementedError

    def get_oracle_state(self):
        raise NotImplementedError

    def get_camera_obs(self):
        self.scene.update_render()
        obs_dict = {}
        for name, camera_cfg in self.camera_infos.items():
            cam = self.cameras[name]
            modalities = list(camera_cfg.keys())
            # ic(modalities)
            texture_names = []
            for modality in modalities:
                if modality == "rgb":
                    texture_names.append("Color")
                elif modality == "depth":
                    texture_names.append("Position")
                elif modality == "point_cloud" and camera_cfg["point_cloud"].get("use_seg") is True:
                    texture_names.append("Segmentation")
                    texture_names.append("Position")
                elif modality == "point_cloud":
                    texture_names.append("Position")
                elif modality == "segmentation":
                    texture_names.append("Segmentation")
                else:
                    raise ValueError(f"Visual modality {modality} not supported.")
            await_dl_list = cam.take_picture_and_get_dl_tensors_async(texture_names)  # how is this done?
            dl_list = await_dl_list.wait()

            i = 0  # because pc_seg use 2 dl_list items, we cannot use enumerate here.
            for modality in modalities:
                key_name = f"{name}-{modality}"  # NOTE
                if modality == "point_cloud" and camera_cfg["point_cloud"].get("use_seg") is True:
                    import torch
                    dl_tensor_seg = dl_list[i]
                    output_array_seg = torch.from_dlpack(dl_tensor_seg).cpu().numpy()
                    i += 1
                    dl_tensor_pos = dl_list[i]
                    shape = sapien.dlpack.dl_shape(dl_tensor_pos)
                    output_array_pos = np.zeros(shape, dtype=np.float32)
                    sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dl_tensor_pos, output_array_pos)
                    sapien.dlpack.dl_cuda_sync()

                    obs_pos = np.reshape(output_array_pos[..., :3], (-1, 3))
                    obs_seg = np.reshape(output_array_seg[..., 1:2], (-1, 1))
                    camera_pose = self.get_camera_to_robot_pose(name)
                    kwargs = camera_cfg["point_cloud"].get("process_fn_kwargs", {})

                    obs = process_pc(cloud=obs_pos, camera_pose=camera_pose, num_points=camera_cfg['point_cloud']['num_points'], np_random=self.np_random, grouping_info=None, segmentation=obs_seg, **kwargs)
                    obs_dict[f"{name}-seg_gt"] = obs[:, 3:]  # NOTE: add gt segmentation
                    if obs_dict[f"{name}-seg_gt"].shape != (camera_cfg["point_cloud"]["num_points"], 4):
                        # align the gt segmentation mask
                        obs_dict[f"{name}-seg_gt"] = np.zeros((camera_cfg["point_cloud"]["num_points"], 4))
                    obs = obs[:, :3]
                else:
                    dl_tensor = dl_list[i]
                    shape = sapien.dlpack.dl_shape(dl_tensor)
                    output_array = np.zeros(shape, dtype=np.float32)
                    if modality == "segmentation":
                        import torch
                        output_array = torch.from_dlpack(
                            dl_tensor).cpu().numpy()  # H, W, 4. [..., 0]: mesh-level segmentation; [..., 1]: link-level segmentation
                    elif modality == "rgb":
                        import torch
                        output_array = cam.get_color_rgba()
                    else:
                        output_array = np.zeros(shape, dtype=np.float32)
                        sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dl_tensor, output_array)
                        sapien.dlpack.dl_cuda_sync()
                    if modality == "rgb":
                        obs = output_array[..., :3]
                    elif modality == "depth":
                        obs = -output_array[..., 2:3]
                        obs[obs[..., 0] > MAX_DEPTH_RANGE] = 0  # Set depth out of range to be 0
                    elif modality == "point_cloud":
                        obs = np.reshape(output_array[..., :3], (-1, 3))
                        # ic(obs.shape)
                        camera_pose = self.get_camera_to_robot_pose(name)
                        kwargs = camera_cfg["point_cloud"].get("process_fn_kwargs", {})
                        obs = process_pc(cloud=obs, camera_pose=camera_pose, num_points=camera_cfg['point_cloud']['num_points'], np_random=self.np_random, grouping_info=None, segmentation=None, noise_level=3 if self.pc_noise else 0, **kwargs)
                        if "additional_process_fn" in camera_cfg["point_cloud"]:
                            for fn in camera_cfg["point_cloud"]["additional_process_fn"]:
                                obs = fn(obs, self.np_random)
                        obs_dict[f"{name}-seg_gt"] = np.zeros((camera_cfg["point_cloud"]["num_points"], 4))
                    elif modality == "segmentation":
                        obs = output_array[..., :2].astype(np.uint8)
                    else:
                        raise RuntimeError("What happen? you should not see this error!")
                obs_dict[key_name] = obs
                i += 1

        if len(self.imaginations) > 0:
            obs_dict.update(self.imaginations)

        return obs_dict
    
    def get_camera_to_robot_pose(self, camera_name):
        gl_pose = self.cameras[camera_name].get_pose()
        camera_pose = gl_pose * gl2sapien
        # camera2robot = self.robot.get_pose().inv() * camera_pose
        return camera_pose.to_transformation_matrix()

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    @property
    def observation_space(self):
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        state_space = gym.spaces.Box(low=low, high=high)
        return state_space


def compute_inverse_kinematics(delta_pose_world, palm_jacobian, damping=0.05):
    lmbda = np.eye(6) * (damping ** 2)
    # When you need the pinv for matrix multiplication, always use np.linalg.solve but not np.linalg.pinv
    delta_qpos = palm_jacobian.T @ np.linalg.lstsq(palm_jacobian.dot(palm_jacobian.T) + lmbda, delta_pose_world, rcond=None)[0]

    return delta_qpos
