import unittest

import numpy as np


class RM75ReferenceUtilsTests(unittest.TestCase):
    def test_native_rm75_hand_action_is_centered_on_neutral_zero_qpos(self):
        from hand_imitation.env.rl_env.reference_utils import recover_hand_action

        lower = np.array([0.0, -0.05, 0.0, -0.10], dtype=np.float32)
        upper = np.array([1.0, 1.50, 0.6, 0.20], dtype=np.float32)
        qlimits = np.stack([lower, upper], axis=1)

        neutral = recover_hand_action(np.zeros(4, dtype=np.float32), qlimits, "rm75_inspire_right")
        close = recover_hand_action(np.ones(4, dtype=np.float32), qlimits, "rm75_inspire_right")
        open_more = recover_hand_action(-np.ones(4, dtype=np.float32), qlimits, "rm75_inspire_right")

        np.testing.assert_allclose(neutral, np.zeros(4, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(close, upper, atol=1e-6)
        np.testing.assert_allclose(open_more, lower, atol=1e-6)

    def test_active_hand_action_expands_to_native_twelve_dof_layout(self):
        from hand_imitation.env.rl_env.reference_utils import expand_rm75_active_hand_action

        active = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        expanded = expand_rm75_active_hand_action(active, 12)

        np.testing.assert_allclose(
            expanded,
            np.array([0.1, 0.2, 0.0, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5, 0.0, 0.6, 0.0]),
            atol=1e-6,
        )

    def test_native_rm75_recovered_qpos_obeys_mimic_joints(self):
        from hand_imitation.env.rl_env.reference_utils import recover_hand_action

        qlimits = np.array(
            [
                [0.0, 1.308],
                [0.0, 0.6],
                [0.0, 0.8],
                [0.0, 0.4],
                [0.0, 1.47],
                [-0.04545, 1.56],
                [0.0, 1.47],
                [-0.04545, 1.56],
                [0.0, 1.47],
                [-0.04545, 1.56],
                [0.0, 1.47],
                [-0.04545, 1.56],
            ],
            dtype=np.float32,
        )
        action = np.array([0.2, 0.5, 0.0, 0.0, 0.7, 0.0, 0.6, 0.0, 0.4, 0.0, 0.3, 0.0], dtype=np.float32)

        qpos = recover_hand_action(action, qlimits, "rm75_inspire_right")

        self.assertAlmostEqual(float(qpos[2]), 1.334 * float(qpos[1]), places=5)
        self.assertAlmostEqual(float(qpos[3]), 0.667 * float(qpos[1]), places=5)
        self.assertAlmostEqual(float(qpos[5]), 1.06399 * float(qpos[4]) - 0.04545, places=5)
        self.assertAlmostEqual(float(qpos[7]), 1.06399 * float(qpos[6]) - 0.04545, places=5)
        self.assertAlmostEqual(float(qpos[9]), 1.06399 * float(qpos[8]) - 0.04545, places=5)
        self.assertAlmostEqual(float(qpos[11]), 1.06399 * float(qpos[10]) - 0.04545, places=5)

    def test_native_active_qpos_uses_more_closed_main_finger_joint(self):
        from hand_imitation.env.rl_env.reference_utils import rm75_native_active_qpos_from_full

        hand_qpos = np.zeros(12, dtype=np.float32)
        hand_qpos[0] = 0.2
        hand_qpos[1] = 0.3
        hand_qpos[4] = 0.1
        hand_qpos[5] = 0.7
        hand_qpos[6] = 0.8
        hand_qpos[7] = 0.2
        hand_qpos[8] = 0.0
        hand_qpos[9] = 0.6
        hand_qpos[10] = 0.4
        hand_qpos[11] = 0.9

        active = rm75_native_active_qpos_from_full(hand_qpos)

        np.testing.assert_allclose(
            active,
            np.array([0.2, 0.3, 0.7, 0.8, 0.6, 0.9], dtype=np.float32),
            atol=1e-6,
        )

    def test_project_linear_velocity_keeps_approach_and_damps_tangent(self):
        from hand_imitation.env.rl_env.reference_utils import project_rm75_velocity_to_object

        palm_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        object_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        linear_velocity = np.array([0.08, 0.10, 0.0], dtype=np.float32)

        projected = project_rm75_velocity_to_object(
            linear_velocity,
            palm_pos,
            object_pos,
            tangent_scale=0.2,
            max_approach_speed=0.05,
            max_retreat_speed=0.01,
        )

        self.assertAlmostEqual(float(projected[0]), 0.05, places=6)
        self.assertAlmostEqual(float(projected[1]), 0.02, places=6)
        self.assertAlmostEqual(float(projected[2]), 0.0, places=6)

    def test_project_linear_velocity_limits_retreat(self):
        from hand_imitation.env.rl_env.reference_utils import project_rm75_velocity_to_object

        palm_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        object_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        linear_velocity = np.array([-0.20, 0.0, 0.0], dtype=np.float32)

        projected = project_rm75_velocity_to_object(
            linear_velocity,
            palm_pos,
            object_pos,
            tangent_scale=0.2,
            max_approach_speed=0.05,
            max_retreat_speed=0.01,
        )

        self.assertAlmostEqual(float(projected[0]), -0.01, places=6)

    def test_project_linear_velocity_can_add_small_approach_bias(self):
        from hand_imitation.env.rl_env.reference_utils import project_rm75_velocity_to_object

        palm_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        object_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        linear_velocity = np.zeros(3, dtype=np.float32)

        projected = project_rm75_velocity_to_object(
            linear_velocity,
            palm_pos,
            object_pos,
            tangent_scale=0.2,
            max_approach_speed=0.05,
            max_retreat_speed=0.01,
            approach_bias=0.02,
        )

        self.assertAlmostEqual(float(projected[0]), 0.02, places=6)
        self.assertAlmostEqual(float(projected[1]), 0.0, places=6)

    def test_precision_score_prefers_stable_hold_over_any_contact_hold(self):
        from hand_imitation.env.rl_env.reference_utils import compute_rm75_grasp_scores

        _, weak = compute_rm75_grasp_scores(
            object_lift=0.0,
            contact_count=4.0,
            stable_grasp_contact=False,
            contact_hold_steps=8,
            stable_contact_hold_steps=0,
            thumb_contact=True,
            non_thumb_contact_count=1,
            object_xy_drift=0.03,
            object_speed=0.0,
            object_tilt_err=0.10,
            object_ang_speed=0.0,
        )
        _, stable = compute_rm75_grasp_scores(
            object_lift=0.0,
            contact_count=4.0,
            stable_grasp_contact=True,
            contact_hold_steps=8,
            stable_contact_hold_steps=8,
            thumb_contact=True,
            non_thumb_contact_count=1,
            object_xy_drift=0.03,
            object_speed=0.0,
            object_tilt_err=0.10,
            object_ang_speed=0.0,
        )

        self.assertGreater(stable, weak + 40.0)

    def test_task_score_requires_stable_hold_before_lift_dominates(self):
        from hand_imitation.env.rl_env.reference_utils import compute_rm75_task_score

        stable_no_lift = compute_rm75_task_score(
            obj_com_err=0.0,
            object_lift=0.0,
            stable_grasp_contact=True,
            stable_contact_hold_steps=8,
            thumb_contact=True,
            non_thumb_contact_count=2,
            object_xy_drift=0.0,
            object_speed=0.0,
            object_tilt_err=0.0,
            object_ang_speed=0.0,
        )
        lifted_without_hold = compute_rm75_task_score(
            obj_com_err=0.0,
            object_lift=0.08,
            stable_grasp_contact=False,
            stable_contact_hold_steps=0,
            thumb_contact=True,
            non_thumb_contact_count=2,
            object_xy_drift=0.0,
            object_speed=0.0,
            object_tilt_err=0.0,
            object_ang_speed=0.0,
        )
        stable_lifted = compute_rm75_task_score(
            obj_com_err=0.0,
            object_lift=0.08,
            stable_grasp_contact=True,
            stable_contact_hold_steps=8,
            thumb_contact=True,
            non_thumb_contact_count=2,
            object_xy_drift=0.0,
            object_speed=0.0,
            object_tilt_err=0.0,
            object_ang_speed=0.0,
        )

        self.assertGreater(stable_no_lift, lifted_without_hold)
        self.assertGreater(stable_lifted, stable_no_lift + 60.0)

    def test_task_score_prefers_low_reference_tracking_error_after_lift(self):
        from hand_imitation.env.rl_env.reference_utils import compute_rm75_task_score

        good_tracking = compute_rm75_task_score(
            obj_com_err=0.005,
            object_lift=0.08,
            stable_grasp_contact=True,
            stable_contact_hold_steps=8,
            thumb_contact=True,
            non_thumb_contact_count=2,
            object_xy_drift=0.0,
            object_speed=0.0,
            object_tilt_err=0.0,
            object_ang_speed=0.0,
        )
        poor_tracking = compute_rm75_task_score(
            obj_com_err=0.08,
            object_lift=0.08,
            stable_grasp_contact=True,
            stable_contact_hold_steps=8,
            thumb_contact=True,
            non_thumb_contact_count=2,
            object_xy_drift=0.0,
            object_speed=0.0,
            object_tilt_err=0.0,
            object_ang_speed=0.0,
        )

        self.assertGreater(good_tracking, poor_tracking + 25.0)


if __name__ == "__main__":
    unittest.main()
