import unittest

import numpy as np


class RM75RewardConfigTests(unittest.TestCase):
    def test_default_config_uses_rm75_specific_thresholds(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig

        cfg = RM75RewardConfig()

        self.assertGreater(cfg.pregrasp_success_thresh, 0.05)
        self.assertGreater(cfg.contact_reward_scale, 0.0)
        self.assertGreaterEqual(cfg.required_non_thumb_contacts, 1)

    def test_config_from_kwargs_overrides_known_fields(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig

        cfg = RM75RewardConfig.from_kwargs(
            {
                "pregrasp_success_thresh": 0.075,
                "object_reward_scale": 12.0,
                "unknown_field": 123,
            }
        )

        self.assertAlmostEqual(cfg.pregrasp_success_thresh, 0.075)
        self.assertAlmostEqual(cfg.object_reward_scale, 12.0)
        self.assertFalse(hasattr(cfg, "unknown_field"))


class RM75RewardFunctionTests(unittest.TestCase):
    def test_pregrasp_reward_and_success_use_configured_threshold(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig, compute_rm75_reward

        cfg = RM75RewardConfig(pregrasp_success_thresh=0.08)
        result = compute_rm75_reward(
            cfg,
            current_step=15,
            pregrasp_steps=15,
            hand_jpos_err=0.06,
            hand_mjpos_err=0.0,
            obj_com_err=0.0,
            obj_rot_err=0.0,
            object_lift=0.0,
            contact_groups=np.zeros(6, dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
        )

        self.assertTrue(result.pregrasp_success)
        self.assertGreater(result.reward, 0.0)

    def test_grasp_reward_requires_thumb_and_non_thumb_for_stable_contact(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig, compute_rm75_reward

        cfg = RM75RewardConfig(required_non_thumb_contacts=2, contact_reward_scale=0.5)
        weak = compute_rm75_reward(
            cfg,
            current_step=16,
            pregrasp_steps=15,
            hand_jpos_err=0.0,
            hand_mjpos_err=0.03,
            obj_com_err=0.02,
            obj_rot_err=0.0,
            object_lift=0.03,
            contact_groups=np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
        )
        stable = compute_rm75_reward(
            cfg,
            current_step=16,
            pregrasp_steps=15,
            hand_jpos_err=0.0,
            hand_mjpos_err=0.03,
            obj_com_err=0.02,
            obj_rot_err=0.0,
            object_lift=0.03,
            contact_groups=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
        )

        self.assertFalse(weak.stable_grasp_contact)
        self.assertTrue(stable.stable_grasp_contact)
        self.assertGreater(stable.reward, weak.reward)

    def test_object_reward_is_progressively_gated_by_contact_hold(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig, compute_rm75_reward

        cfg = RM75RewardConfig(
            object_reward_scale=10.0,
            lift_reward_scale=20.0,
            object_reward_contact_hold_steps=4,
            reward_divisor=1.0,
            controller_penalty_scale=0.0,
            action_penalty_scale=0.0,
        )
        kwargs = dict(
            current_step=16,
            pregrasp_steps=15,
            hand_jpos_err=0.0,
            hand_mjpos_err=0.03,
            obj_com_err=0.01,
            obj_rot_err=0.0,
            object_lift=0.06,
            fingertip_obj_dist=0.05,
            object_xy_drift=0.0,
            object_speed=0.0,
            contact_groups=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
        )

        early = compute_rm75_reward(cfg, contact_hold_steps=1, **kwargs)
        held = compute_rm75_reward(cfg, contact_hold_steps=4, **kwargs)

        self.assertGreater(held.reward, early.reward)

    def test_stable_contact_gate_uses_stable_hold_not_any_contact_hold(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig, compute_rm75_reward

        cfg = RM75RewardConfig(
            object_reward_requires_stable_contact=True,
            object_reward_contact_hold_steps=4,
            object_reward_scale=10.0,
            lift_reward_scale=20.0,
            reward_divisor=1.0,
            controller_penalty_scale=0.0,
            action_penalty_scale=0.0,
        )
        kwargs = dict(
            current_step=16,
            pregrasp_steps=15,
            hand_jpos_err=0.0,
            hand_mjpos_err=0.03,
            obj_com_err=0.01,
            obj_rot_err=0.0,
            object_lift=0.06,
            fingertip_obj_dist=0.05,
            object_xy_drift=0.0,
            object_speed=0.0,
            object_tilt_err=0.0,
            object_ang_speed=0.0,
            contact_groups=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
            contact_hold_steps=10,
        )

        unstable_hold = compute_rm75_reward(cfg, stable_contact_hold_steps=0, **kwargs)
        stable_hold = compute_rm75_reward(cfg, stable_contact_hold_steps=4, **kwargs)

        self.assertGreater(stable_hold.reward, unstable_hold.reward)

    def test_tilt_and_angular_speed_reduce_grasp_reward(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig, compute_rm75_reward

        cfg = RM75RewardConfig(
            object_tilt_penalty_scale=10.0,
            object_tilt_free_thresh=0.1,
            object_ang_vel_penalty_scale=0.5,
            reward_divisor=1.0,
            controller_penalty_scale=0.0,
            action_penalty_scale=0.0,
        )
        kwargs = dict(
            current_step=16,
            pregrasp_steps=15,
            hand_jpos_err=0.0,
            hand_mjpos_err=0.03,
            obj_com_err=0.01,
            obj_rot_err=0.0,
            object_lift=0.03,
            fingertip_obj_dist=0.05,
            object_xy_drift=0.0,
            object_speed=0.0,
            contact_groups=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
            contact_hold_steps=4,
        )

        stable = compute_rm75_reward(cfg, object_tilt_err=0.05, object_ang_speed=0.0, **kwargs)
        tipped = compute_rm75_reward(cfg, object_tilt_err=0.5, object_ang_speed=2.0, **kwargs)

        self.assertGreater(stable.reward, tipped.reward)

    def test_reference_close_reward_prefers_matching_reference_schedule(self):
        from hand_imitation.env.rl_env.rm75_reward import RM75RewardConfig, compute_rm75_reward

        cfg = RM75RewardConfig(
            reference_close_reward_scale=4.0,
            reference_close_penalty_scale=3.0,
            reference_close_start_step=0,
            reward_divisor=1.0,
            controller_penalty_scale=0.0,
            action_penalty_scale=0.0,
        )
        kwargs = dict(
            current_step=25,
            pregrasp_steps=15,
            hand_jpos_err=0.0,
            hand_mjpos_err=0.03,
            obj_com_err=0.04,
            obj_rot_err=0.0,
            object_lift=0.0,
            fingertip_obj_dist=0.06,
            palm_obj_dist=0.16,
            min_fingertip_obj_dist=0.05,
            object_xy_drift=0.0,
            object_speed=0.0,
            contact_groups=np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
            cartesian_error=0.0,
            qvel=np.zeros(19, dtype=np.float32),
            reference_close_fraction=0.70,
        )

        under_closed = compute_rm75_reward(cfg, hand_close_fraction=0.25, **kwargs)
        matched = compute_rm75_reward(cfg, hand_close_fraction=0.68, **kwargs)

        self.assertGreater(matched.reward, under_closed.reward)


if __name__ == "__main__":
    unittest.main()
