import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class _Pose:
    def __init__(self, p, q=None):
        self.p = np.asarray(p, dtype=np.float32)
        self.q = np.asarray(q if q is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _Link:
    def __init__(self, p):
        self._pose = _Pose(p)

    def get_pose(self):
        return self._pose


class _Actor:
    def __init__(self, p, q=None):
        self._pose = _Pose(p, q)

    def get_pose(self):
        return self._pose

    def get_velocity(self):
        return np.zeros(3, dtype=np.float32)

    def get_angular_velocity(self):
        return np.zeros(3, dtype=np.float32)


class _Robot:
    def __init__(self, qvel, qpos=None, qlimits=None):
        self._qvel = np.asarray(qvel, dtype=np.float32)
        self._qpos = np.asarray(qpos if qpos is not None else np.zeros_like(self._qvel), dtype=np.float32)
        default_limits = np.stack(
            [np.zeros_like(self._qpos), np.ones_like(self._qpos)],
            axis=1,
        )
        self._qlimits = np.asarray(qlimits if qlimits is not None else default_limits, dtype=np.float32)

    def get_qvel(self):
        return self._qvel

    def get_qpos(self):
        return self._qpos

    def set_qpos(self, qpos):
        self._qpos = np.asarray(qpos, dtype=np.float32)

    def get_qlimits(self):
        return self._qlimits

    def set_drive_target(self, target):
        self.drive_target = np.asarray(target, dtype=np.float32)

    def set_drive_velocity_target(self, target):
        self.drive_velocity_target = np.asarray(target, dtype=np.float32)

    def set_qf(self, qf):
        self.qf = qf

    def compute_passive_force(self, external=False, coriolis_and_centrifugal=False):
        return np.zeros_like(self._qpos)


class _Scene:
    def step(self):
        return None


class _KinematicModel:
    def compute_end_link_spatial_jacobian(self, qpos):
        return np.eye(6, len(qpos), dtype=np.float32)


class RM75RelocateEnvTests(unittest.TestCase):
    def test_constructor_parses_reward_kwargs_before_base_init(self):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv

        calls = []

        def fake_base_init(self, *args, **kwargs):
            calls.append(kwargs)

        original_init = RM75RelocateRLEnv.__mro__[1].__init__
        RM75RelocateRLEnv.__mro__[1].__init__ = fake_base_init
        try:
            env = RM75RelocateRLEnv(
                motion_file="dummy-0",
                task_kwargs={"reward_kwargs": {"pregrasp_success_thresh": 0.09}},
            )
        finally:
            RM75RelocateRLEnv.__mro__[1].__init__ = original_init

        self.assertEqual(env.robot_name, "rm75_inspire_right")
        self.assertAlmostEqual(env.rm75_reward_cfg.pregrasp_success_thresh, 0.09)
        self.assertEqual(calls[0]["robot_name"], "rm75_inspire_right")

    def test_approach_delta_override_can_ignore_trajectory_template(self):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv

        env = RM75RelocateRLEnv.__new__(RM75RelocateRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.rm75_native_hand_control = True
        env.rm75_native_apply_template_approach_pregrasp = True
        env.rm75_default_approach_delta = np.asarray([0.10, -0.20, 0.03], dtype=np.float32)
        env.rm75_override_template_approach_delta = True
        env.cur_reference_motion = {
            "robot_jpos": np.asarray(
                [[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]],
                dtype=np.float32,
            ),
            "robot_pregrasp_jpos": np.zeros((1, 2, 3), dtype=np.float32),
            "rm75_template_approach_delta": np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
        }

        env._apply_rm75_approach_pregrasp_reference()

        expected = env.cur_reference_motion["robot_jpos"][0] + env.rm75_default_approach_delta[None, :]
        np.testing.assert_allclose(env.cur_reference_motion["robot_pregrasp_jpos"][-1], expected)

    def test_approach_delta_uses_trajectory_template_by_default(self):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv

        env = RM75RelocateRLEnv.__new__(RM75RelocateRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.rm75_native_hand_control = True
        env.rm75_native_apply_template_approach_pregrasp = True
        env.rm75_default_approach_delta = np.asarray([0.10, -0.20, 0.03], dtype=np.float32)
        env.rm75_override_template_approach_delta = False
        template_delta = np.asarray([0.01, -0.02, -0.03], dtype=np.float32)
        env.cur_reference_motion = {
            "robot_jpos": np.asarray(
                [[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]],
                dtype=np.float32,
            ),
            "robot_pregrasp_jpos": np.zeros((1, 2, 3), dtype=np.float32),
            "rm75_template_approach_delta": template_delta,
        }

        env._apply_rm75_approach_pregrasp_reference()

        expected = env.cur_reference_motion["robot_jpos"][0] + template_delta[None, :]
        np.testing.assert_allclose(env.cur_reference_motion["robot_pregrasp_jpos"][-1], expected)

    def _make_reward_test_env(self, *, current_step=16):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig

        env = RM75RelocateRLEnv.__new__(RM75RelocateRLEnv)
        env.is_vision = False
        env.is_demo_rollout = False
        env.finger_tip_links = [
            _Link([0.0, 0.0, 0.0]),
            _Link([0.1, 0.0, 0.0]),
            _Link([0.0, 0.1, 0.0]),
            _Link([0.1, 0.1, 0.0]),
            _Link([0.05, 0.05, 0.0]),
        ]
        env.ee_link = _Link([0.0, 0.0, 0.0])
        env.palm_link = _Link([0.0, 0.0, 0.0])
        env.manipulated_object = _Actor([0.0, 0.0, 0.04])
        env.robot = _Robot(np.zeros(19, dtype=np.float32))
        env.arm_dof = 7
        env.current_step = current_step
        env.pregrasp_steps = 15
        env.robot_name = "rm75_inspire_right"
        env.rm75_contact_hold_steps = 0
        env.rm75_stable_contact_hold_steps = 0
        env.rm75_reward_cfg = RM75RewardConfig(
            pregrasp_success_thresh=0.09,
            stable_contact_bonus=2.0,
            required_non_thumb_contacts=2,
            reward_divisor=1.0,
            controller_penalty_scale=0.0,
        )
        env.robot_object_contact = np.zeros(6, dtype=np.float32)
        env._contact_groups = lambda: np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
        env.init_object_height = 0.0
        env.cartesian_error = 0.0
        env.pregrasp_success = False
        env.cur_reference_motion = {
            "object_translation": np.asarray([[0.0, 0.0, 0.04], [0.0, 0.0, 0.04]], dtype=np.float32),
            "object_orientation": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "robot_jpos": np.asarray(
                [
                    [[0.0, 0.0, 0.0]] * 5,
                    [[0.0, 0.0, 0.0]] * 5,
                ],
                dtype=np.float32,
            ),
        }
        return env

    def test_get_reward_uses_rm75_success_threshold_and_contact_bonus(self):
        env = self._make_reward_test_env(current_step=15)

        reward = env.get_reward(np.zeros(13, dtype=np.float32))

        self.assertGreater(reward, 0.0)
        self.assertTrue(env.pregrasp_success)
        self.assertTrue(env.stable_grasp_contact)
        self.assertEqual(env.contact_count, 3.0)

    def test_pregrasp_success_is_only_set_at_gate_step(self):
        env = self._make_reward_test_env(current_step=10)

        env.get_reward(np.zeros(13, dtype=np.float32))

        self.assertFalse(env.pregrasp_success)

    def test_get_reward_uses_hand_palm_link_for_palm_object_distance(self):
        env = self._make_reward_test_env(current_step=16)
        env.ee_link = _Link([10.0, 0.0, 0.0])
        env.palm_link = _Link([0.0, 0.0, 0.04])

        env.get_reward(np.zeros(13, dtype=np.float32))

        self.assertAlmostEqual(env.palm_obj_dist, 0.0)

    def test_bad_push_done_requires_unstable_large_drift_or_tilt(self):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig

        env = RM75RelocateRLEnv.__new__(RM75RelocateRLEnv)
        env.is_vision = False
        env.is_demo_rollout = False
        env.pregrasp_success = True
        env.current_step = 24
        env.pregrasp_steps = 15
        env.robot_name = "rm75_inspire_right"
        env.imitate_steps = 60
        env.traj_step = 0
        env.no_contact_steps = 0
        env.is_contact = True
        env.stable_grasp_contact = False
        env.obj_com_err = 0.03
        env.object_xy_drift = 0.065
        env.object_tilt_err = 0.10
        env.rm75_reward_cfg = RM75RewardConfig(
            bad_push_done=True,
            bad_push_min_step=5,
            bad_push_drift_thresh=0.05,
            bad_push_tilt_thresh=0.35,
            no_contact_grace_steps=100,
        )

        self.assertTrue(env.is_done())

    def test_bad_push_done_does_not_stop_stable_grasp(self):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig

        env = RM75RelocateRLEnv.__new__(RM75RelocateRLEnv)
        env.is_vision = False
        env.is_demo_rollout = False
        env.pregrasp_success = True
        env.current_step = 24
        env.pregrasp_steps = 15
        env.robot_name = "rm75_inspire_right"
        env.imitate_steps = 60
        env.traj_step = 0
        env.no_contact_steps = 0
        env.is_contact = True
        env.stable_grasp_contact = True
        env.obj_com_err = 0.03
        env.object_xy_drift = 0.065
        env.object_tilt_err = 0.50
        env.rm75_reward_cfg = RM75RewardConfig(
            bad_push_done=True,
            bad_push_min_step=5,
            bad_push_drift_thresh=0.05,
            bad_push_tilt_thresh=0.35,
            no_contact_grace_steps=100,
        )

        self.assertFalse(env.is_done())

    def test_reference_close_fraction_follows_robot_qpos_schedule(self):
        from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv

        env = RM75RelocateRLEnv.__new__(RM75RelocateRLEnv)
        env.arm_dof = 7
        env.current_step = 25
        env.pregrasp_steps = 15
        env.cur_reference_motion = {
            "object_translation": np.zeros((20, 3), dtype=np.float32),
            "robot_qpos": np.zeros((20, 13), dtype=np.float32),
        }
        env.cur_reference_motion["robot_qpos"][:, 7:] = np.linspace(
            [0.08, 0.03, 0.03, 0.03, 0.03, 0.03],
            [0.85, 0.45, 1.0, 1.0, 1.0, 1.0],
            20,
            dtype=np.float32,
        )

        close_fraction = env._reference_close_fraction()

        self.assertGreater(close_fraction, 0.45)
        self.assertLess(close_fraction, 0.70)

    def test_hold_until_stable_keeps_arm_scale_after_ramp_window(self):
        from hand_imitation.env.rl_env.base import BaseRLEnv

        env = BaseRLEnv.__new__(BaseRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.arm_dof = 7
        env.robot = _Robot(
            np.zeros(13, dtype=np.float32),
            qpos=np.zeros(13, dtype=np.float32),
            qlimits=np.vstack(
                [
                    np.tile([-1.0, 1.0], (7, 1)),
                    np.tile([0.0, 1.0], (6, 1)),
                ]
            ).astype(np.float32),
        )
        env.ee_link = _Link([0.0, 0.0, 0.0])
        env.kinematic_model = _KinematicModel()
        env.velocity_limit = np.tile([-1.0, 1.0], (6, 1)).astype(np.float32)
        env.control_time_step = 0.1
        env.frame_skip = 1
        env.scene = _Scene()
        env.current_step = 40
        env.pregrasp_steps = 15
        env.rm75_arm_action_scale = 0.5
        env.rm75_hand_action_scale = 1.0
        env.rm75_pregrasp_safe_action = False
        env.rm75_post_pregrasp_arm_ramp_steps = 4
        env.rm75_post_pregrasp_arm_scale_start = 0.01
        env.rm75_post_pregrasp_arm_scale_end = 0.5
        env.rm75_post_pregrasp_hold_until_stable = True
        env.rm75_post_pregrasp_hold_arm_scale = 0.02
        env.rm75_post_pregrasp_required_stable_hold_steps = 3
        env.rm75_post_pregrasp_project_to_object = False
        env.rm75_stable_contact_hold_steps = 0
        env.rm75_hand_close_bias_weights = np.ones(6, dtype=np.float32)
        env.rm75_hand_close_bias = 0.0

        with patch("hand_imitation.env.rl_env.base.enforce_rm75_coupled_hand_qpos", lambda qpos, arm_dof, qlimits: qpos):
            env.arm_sim_step(np.ones(12, dtype=np.float32))

        self.assertAlmostEqual(env.rm75_last_effective_arm_scale, 0.02)

    def test_hold_until_stable_releases_arm_after_required_stable_hold(self):
        from hand_imitation.env.rl_env.base import BaseRLEnv

        env = BaseRLEnv.__new__(BaseRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.arm_dof = 7
        env.robot = _Robot(
            np.zeros(13, dtype=np.float32),
            qpos=np.zeros(13, dtype=np.float32),
            qlimits=np.vstack(
                [
                    np.tile([-1.0, 1.0], (7, 1)),
                    np.tile([0.0, 1.0], (6, 1)),
                ]
            ).astype(np.float32),
        )
        env.ee_link = _Link([0.0, 0.0, 0.0])
        env.kinematic_model = _KinematicModel()
        env.velocity_limit = np.tile([-1.0, 1.0], (6, 1)).astype(np.float32)
        env.control_time_step = 0.1
        env.frame_skip = 1
        env.scene = _Scene()
        env.current_step = 40
        env.pregrasp_steps = 15
        env.rm75_arm_action_scale = 0.5
        env.rm75_hand_action_scale = 1.0
        env.rm75_pregrasp_safe_action = False
        env.rm75_post_pregrasp_arm_ramp_steps = 4
        env.rm75_post_pregrasp_arm_scale_start = 0.01
        env.rm75_post_pregrasp_arm_scale_end = 0.5
        env.rm75_post_pregrasp_hold_until_stable = True
        env.rm75_post_pregrasp_hold_arm_scale = 0.02
        env.rm75_post_pregrasp_required_stable_hold_steps = 3
        env.rm75_post_pregrasp_project_to_object = False
        env.rm75_stable_contact_hold_steps = 3
        env.rm75_hand_close_bias_weights = np.ones(6, dtype=np.float32)
        env.rm75_hand_close_bias = 0.0

        with patch("hand_imitation.env.rl_env.base.enforce_rm75_coupled_hand_qpos", lambda qpos, arm_dof, qlimits: qpos):
            env.arm_sim_step(np.ones(12, dtype=np.float32))

        self.assertAlmostEqual(env.rm75_last_effective_arm_scale, 0.5)

    def test_post_stable_lift_only_clamps_xy_and_keeps_small_up_velocity(self):
        from hand_imitation.env.rl_env.base import BaseRLEnv

        env = BaseRLEnv.__new__(BaseRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.arm_dof = 7
        env.robot = _Robot(
            np.zeros(13, dtype=np.float32),
            qpos=np.zeros(13, dtype=np.float32),
            qlimits=np.vstack(
                [
                    np.tile([-1.0, 1.0], (7, 1)),
                    np.tile([0.0, 1.0], (6, 1)),
                ]
            ).astype(np.float32),
        )
        env.ee_link = _Link([0.0, 0.0, 0.0])
        env.kinematic_model = _KinematicModel()
        env.velocity_limit = np.tile([-1.0, 1.0], (6, 1)).astype(np.float32)
        env.control_time_step = 0.1
        env.frame_skip = 1
        env.scene = _Scene()
        env.current_step = 30
        env.pregrasp_steps = 15
        env.rm75_arm_action_scale = 1.0
        env.rm75_hand_action_scale = 1.0
        env.rm75_pregrasp_safe_action = False
        env.rm75_post_pregrasp_arm_ramp_steps = 0
        env.rm75_post_pregrasp_hold_until_stable = False
        env.rm75_post_pregrasp_project_to_object = False
        env.rm75_post_pregrasp_required_stable_hold_steps = 4
        env.rm75_stable_contact_hold_steps = 4
        env.rm75_post_stable_lift_only = True
        env.rm75_post_stable_max_xy_speed = 0.01
        env.rm75_post_stable_max_down_speed = 0.002
        env.rm75_post_stable_max_up_speed = 0.03
        env.rm75_post_stable_lift_bias = 0.01
        env.rm75_post_stable_max_angular_speed = 0.05
        env.rm75_hand_close_bias_weights = np.ones(6, dtype=np.float32)
        env.rm75_hand_close_bias = 0.0

        action = np.array([1.0, -1.0, 0.5, 1.0, -1.0, 1.0] + [0.0] * 6, dtype=np.float32)
        with patch("hand_imitation.env.rl_env.base.enforce_rm75_coupled_hand_qpos", lambda qpos, arm_dof, qlimits: qpos):
            env.arm_sim_step(action)

        self.assertTrue(env.rm75_last_post_stable_lift_only_active)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[0]), 0.01, places=4)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[1]), -0.01, places=4)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[2]), 0.03, places=3)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[3]), 0.05, places=3)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[4]), -0.05, places=3)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[5]), 0.05, places=3)

    def test_project_to_object_damps_tangent_velocity_before_stable_hold(self):
        from hand_imitation.env.rl_env.base import BaseRLEnv

        env = BaseRLEnv.__new__(BaseRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.arm_dof = 7
        env.robot = _Robot(
            np.zeros(13, dtype=np.float32),
            qpos=np.zeros(13, dtype=np.float32),
            qlimits=np.vstack(
                [
                    np.tile([-1.0, 1.0], (7, 1)),
                    np.tile([0.0, 1.0], (6, 1)),
                ]
            ).astype(np.float32),
        )
        env.ee_link = _Link([0.0, 0.0, 0.0])
        env.manipulated_object = _Actor([1.0, 0.0, 0.0])
        env.kinematic_model = _KinematicModel()
        env.velocity_limit = np.tile([-1.0, 1.0], (6, 1)).astype(np.float32)
        env.control_time_step = 0.1
        env.frame_skip = 1
        env.scene = _Scene()
        env.current_step = 20
        env.pregrasp_steps = 15
        env.rm75_arm_action_scale = 1.0
        env.rm75_hand_action_scale = 1.0
        env.rm75_pregrasp_safe_action = False
        env.rm75_post_pregrasp_arm_ramp_steps = 0
        env.rm75_post_pregrasp_hold_until_stable = False
        env.rm75_post_pregrasp_required_stable_hold_steps = 3
        env.rm75_stable_contact_hold_steps = 0
        env.rm75_post_pregrasp_project_to_object = True
        env.rm75_post_pregrasp_project_until_stable = True
        env.rm75_post_pregrasp_project_tangent_scale = 0.2
        env.rm75_post_pregrasp_project_max_approach_speed = 0.05
        env.rm75_post_pregrasp_project_max_retreat_speed = 0.01
        env.rm75_hand_close_bias_weights = np.ones(6, dtype=np.float32)
        env.rm75_hand_close_bias = 0.0

        with patch("hand_imitation.env.rl_env.base.enforce_rm75_coupled_hand_qpos", lambda qpos, arm_dof, qlimits: qpos):
            env.arm_sim_step(np.array([0.2, 0.5, 0.0, 0.0, 0.0, 0.0] + [0.0] * 6, dtype=np.float32))

        self.assertTrue(env.rm75_last_project_to_object_active)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[0]), 0.05, places=3)
        self.assertAlmostEqual(float(env.robot.drive_velocity_target[1]), 0.10, places=3)

    def test_wrist_angular_bias_is_applied_only_in_post_pregrasp_window(self):
        from hand_imitation.env.rl_env.base import BaseRLEnv

        env = BaseRLEnv.__new__(BaseRLEnv)
        env.robot_name = "rm75_inspire_right"
        env.arm_dof = 7
        env.robot = _Robot(
            np.zeros(13, dtype=np.float32),
            qpos=np.zeros(13, dtype=np.float32),
            qlimits=np.vstack(
                [
                    np.tile([-1.0, 1.0], (7, 1)),
                    np.tile([0.0, 1.0], (6, 1)),
                ]
            ).astype(np.float32),
        )
        env.ee_link = _Link([0.0, 0.0, 0.0])
        env.kinematic_model = _KinematicModel()
        env.velocity_limit = np.tile([-1.0, 1.0], (6, 1)).astype(np.float32)
        env.control_time_step = 0.1
        env.frame_skip = 1
        env.scene = _Scene()
        env.current_step = 20
        env.pregrasp_steps = 15
        env.rm75_arm_action_scale = 1.0
        env.rm75_hand_action_scale = 1.0
        env.rm75_pregrasp_safe_action = False
        env.rm75_post_pregrasp_arm_ramp_steps = 0
        env.rm75_post_pregrasp_hold_until_stable = False
        env.rm75_post_pregrasp_project_to_object = False
        env.rm75_wrist_angular_bias = np.array([0.0, 0.0, -0.35], dtype=np.float32)
        env.rm75_wrist_angular_bias_start_step = 0
        env.rm75_wrist_angular_bias_end_step = 10
        env.rm75_hand_close_bias_weights = np.ones(6, dtype=np.float32)
        env.rm75_hand_close_bias = 0.0

        with patch("hand_imitation.env.rl_env.base.enforce_rm75_coupled_hand_qpos", lambda qpos, arm_dof, qlimits: qpos):
            env.arm_sim_step(np.zeros(12, dtype=np.float32))

        self.assertLess(abs(float(env.robot.drive_velocity_target[5]) + 0.35), 1e-3)

        env.current_step = 40
        with patch("hand_imitation.env.rl_env.base.enforce_rm75_coupled_hand_qpos", lambda qpos, arm_dof, qlimits: qpos):
            env.arm_sim_step(np.zeros(12, dtype=np.float32))

        self.assertAlmostEqual(float(env.robot.drive_velocity_target[5]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
