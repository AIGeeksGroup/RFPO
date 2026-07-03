import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock
from pathlib import Path

import numpy as np

from tools.retarget_rm75_inspire_reference import (
    RM75_DRIVEN12_TOTAL_DOF,
    RM75_TOTAL_DOF,
    build_rm75_opposed_envelope_robot_jpos,
    build_rm75_robot_jpos_from_allegro,
    build_rm75_reachable_robot_jpos,
    build_rm75_seed_qpos,
    build_rm75_vividex_aligned_robot_jpos,
    rm75_pregrasp_hand_qpos,
    retarget_file,
    retarget_motion_dict,
)


class RM75RetargetTests(unittest.TestCase):
    def test_builds_palm_plus_five_fingertips_reference(self):
        allegro = np.zeros((2, 5, 3), dtype=np.float32)
        allegro[:, 0] = [0.1, 0.2, 0.3]
        allegro[:, 1] = [0.0, 0.0, 0.0]
        allegro[:, 2] = [0.0, 0.1, 0.0]
        allegro[:, 3] = [0.0, 0.2, 0.0]
        allegro[:, 4] = [0.0, 0.3, 0.0]

        rm75 = build_rm75_robot_jpos_from_allegro(allegro)

        self.assertEqual(rm75.shape, (2, 6, 3))
        np.testing.assert_allclose(rm75[:, 0], allegro[:, 0])
        np.testing.assert_allclose(rm75[:, 1], allegro[:, 1])
        np.testing.assert_allclose(rm75[:, 2], allegro[:, 2])
        np.testing.assert_allclose(rm75[:, 3], allegro[:, 3])
        np.testing.assert_allclose(rm75[:, 4], allegro[:, 4])
        np.testing.assert_allclose(rm75[:, 5], 0.5 * (allegro[:, 3] + allegro[:, 4]))

    def test_builds_reachable_rm75_reference_from_object_path(self):
        object_translation = np.array(
            [[0.0, 0.0, 0.08], [0.0, 0.0, 0.10], [0.0, 0.0, 0.14]],
            dtype=np.float32,
        )

        rm75 = build_rm75_reachable_robot_jpos(
            object_translation,
            palm_offset=np.array([0.057, -0.089, 0.090], dtype=np.float32),
        )

        self.assertEqual(rm75.shape, (3, 6, 3))
        expected_palm_offset = np.tile(np.array([0.057, -0.089, 0.090], dtype=np.float32), (3, 1))
        np.testing.assert_allclose(rm75[:, 0] - object_translation, expected_palm_offset, atol=1e-6)
        expected_first_tip_offset = np.array([-0.064913, 0.126726, 0.094183], dtype=np.float32)
        np.testing.assert_allclose(rm75[0, 1] - rm75[0, 0], expected_first_tip_offset, atol=1e-6)
        self.assertGreater(rm75[0, 1, 2] - rm75[0, 0, 2], 0.09)
        self.assertGreater(np.mean(rm75[0, 2:, 1] - rm75[0, 0, 1]), 0.20)

    def test_reachable_reference_keeps_pregrasp_open_until_pregrasp_step(self):
        object_translation = np.zeros((5, 3), dtype=np.float32)
        object_translation[:, 2] = 0.08

        rm75 = build_rm75_reachable_robot_jpos(
            object_translation,
            pregrasp_step=2,
            palm_offset=np.array([0.057, -0.089, 0.090], dtype=np.float32),
        )

        np.testing.assert_allclose(rm75[0], rm75[2], atol=1e-6)
        self.assertGreater(np.linalg.norm(rm75[-1, 1:].mean(axis=0) - rm75[2, 1:].mean(axis=0)), 0.02)

    def test_opposed_envelope_places_thumb_opposite_main_fingers(self):
        object_translation = np.zeros((5, 3), dtype=np.float32)
        object_translation[:, 2] = 0.08

        rm75 = build_rm75_opposed_envelope_robot_jpos(object_translation, pregrasp_step=2)

        self.assertEqual(rm75.shape, (5, 6, 3))
        np.testing.assert_allclose(rm75[0], rm75[2], atol=1e-6)
        final_rel = rm75[-1] - object_translation[-1]
        thumb_y = final_rel[1, 1]
        main_y = float(np.mean(final_rel[2:5, 1]))
        self.assertLess(thumb_y, -0.02)
        self.assertGreater(main_y, 0.02)
        self.assertGreater(main_y - thumb_y, 0.05)

    def test_seed_qpos_has_rm75_arm_and_six_hand_dofs(self):
        old_qpos = np.ones((3, 22), dtype=np.float32)
        qpos = build_rm75_seed_qpos(old_qpos)

        self.assertEqual(qpos.shape, (3, RM75_TOTAL_DOF))
        np.testing.assert_allclose(qpos[:, :7], 0.0)
        self.assertTrue(np.all(qpos[:, 7:] >= 0.0))
        self.assertTrue(np.all(qpos[:, 7:] <= 1.56))

    def test_seed_qpos_keeps_pregrasp_hand_until_pregrasp_step(self):
        old_qpos = np.ones((6, 22), dtype=np.float32)
        qpos = build_rm75_seed_qpos(old_qpos, pregrasp_step=2)

        np.testing.assert_allclose(qpos[0, 7:], qpos[2, 7:], atol=1e-6)
        self.assertGreater(np.linalg.norm(qpos[-1, 7:] - qpos[2, 7:]), 0.5)

    def test_seed_qpos_can_keep_pinky_open_for_four_finger_grasp(self):
        old_qpos = np.ones((6, 22), dtype=np.float32)
        closed = np.array([0.85, 0.45, 1.0, 1.0, 1.0, 0.03], dtype=np.float32)

        qpos = build_rm75_seed_qpos(old_qpos, pregrasp_step=2, closed_hand_qpos=closed)

        np.testing.assert_allclose(qpos[-1, 7:12], closed[:5], atol=1e-6)
        self.assertAlmostEqual(float(qpos[-1, 12]), 0.03, places=6)
        self.assertGreater(float(qpos[-1, 11]), 0.9)

    def test_seed_qpos_can_expand_to_driven12_hand(self):
        old_qpos = np.ones((4, 22), dtype=np.float32)
        qpos = build_rm75_seed_qpos(old_qpos, pregrasp_step=1, hand_dof=12)

        self.assertEqual(qpos.shape, (4, RM75_DRIVEN12_TOTAL_DOF))
        np.testing.assert_allclose(qpos[:, :7], 0.0)
        self.assertGreater(qpos[-1, 9], qpos[1, 9])
        self.assertGreater(qpos[-1, 12], qpos[1, 12])

    def test_retarget_motion_records_four_finger_seed_pinky_qpos(self):
        src = {
            "robot_jpos": np.zeros((4, 5, 3), dtype=np.float32),
            "robot_qpos": np.zeros((4, 22), dtype=np.float32),
            "object_translation": np.zeros((4, 3), dtype=np.float32),
            "object_orientation": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (4, 1)),
            "pregrasp_step": np.array(1, dtype=np.int32),
        }

        out = retarget_motion_dict(src, mode="opposed-envelope", hand_dof=12, seed_pinky_qpos=0.03)

        self.assertIn("retarget_seed_pinky_qpos", out)
        self.assertAlmostEqual(float(out["retarget_seed_pinky_qpos"]), 0.03, places=6)
        self.assertGreater(float(out["robot_qpos"][-1, 16]), 0.9)
        self.assertAlmostEqual(float(out["robot_qpos"][-1, 17]), 0.03, places=6)

    def test_seed_qpos_can_expand_to_native_mimic_hand(self):
        old_qpos = np.ones((4, 22), dtype=np.float32)
        qpos = build_rm75_seed_qpos(old_qpos, pregrasp_step=1, hand_dof=12, native_mimic=True)
        hand = qpos[-1, 7:]

        self.assertEqual(qpos.shape, (4, RM75_DRIVEN12_TOTAL_DOF))
        self.assertGreater(hand[4], qpos[1, 11])
        self.assertAlmostEqual(float(hand[5]), 1.06399 * float(hand[4]) - 0.04545, places=5)
        self.assertAlmostEqual(float(hand[7]), 1.06399 * float(hand[6]) - 0.04545, places=5)
        self.assertAlmostEqual(float(hand[9]), 1.06399 * float(hand[8]) - 0.04545, places=5)

    def test_rm75_pregrasp_hand_qpos_is_open_and_robot_specific(self):
        qpos = rm75_pregrasp_hand_qpos()

        self.assertEqual(qpos.shape, (6,))
        np.testing.assert_allclose(qpos, np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32))

    def test_vividex_aligned_reference_preserves_source_palm_path(self):
        source_robot_jpos = np.zeros((4, 5, 3), dtype=np.float32)
        source_robot_jpos[:, 0] = np.array(
            [[0.20, -0.50, 0.01], [0.18, -0.45, 0.04], [0.12, -0.30, 0.08], [0.10, -0.25, 0.12]],
            dtype=np.float32,
        )
        object_translation = np.zeros((4, 3), dtype=np.float32)
        object_translation[:, 2] = 0.08

        rm75 = build_rm75_vividex_aligned_robot_jpos(source_robot_jpos, object_translation, pregrasp_step=1)

        self.assertEqual(rm75.shape, (4, 6, 3))
        np.testing.assert_allclose(rm75[:, 0], source_robot_jpos[:, 0], atol=1e-6)
        self.assertLess(np.linalg.norm(rm75[0, 1:].mean(axis=0) - source_robot_jpos[0, 0]), 0.22)
        self.assertGreater(np.linalg.norm(rm75[-1, 1:].mean(axis=0) - rm75[1, 1:].mean(axis=0)), 0.02)

    def test_retarget_file_writes_robot_specific_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.npz"
            dst = Path(tmp) / "dst.npz"
            np.savez(
                src,
                robot_qpos=np.zeros((4, 22), dtype=np.float32),
                robot_jpos=np.zeros((4, 5, 3), dtype=np.float32),
                object_name=np.array("006_mustard_bottle"),
                object_translation=np.zeros((4, 3), dtype=np.float32),
                object_orientation=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (4, 1)),
                length=np.array(4),
                SIM_SUBSTEPS=np.array(10),
                DATA_SUBSTEPS=np.array(1),
                pregrasp_step=np.array(1),
                init_object_height=np.array(0.08, dtype=np.float32),
                init_object_quat=np.array([1, 0, 0, 0], dtype=np.float32),
            )

            retarget_file(src, dst)
            out = np.load(dst, allow_pickle=True)

            self.assertEqual(out["robot_qpos"].shape, (4, RM75_TOTAL_DOF))
            self.assertEqual(out["robot_jpos"].shape, (4, 6, 3))
            self.assertEqual(str(out["retarget_robot"]), "rm75_inspire_right")
            self.assertEqual(str(out["retarget_mode"]), "reachable")

    def test_retarget_file_can_write_driven12_qpos(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.npz"
            dst = Path(tmp) / "dst.npz"
            np.savez(
                src,
                robot_qpos=np.zeros((4, 22), dtype=np.float32),
                robot_jpos=np.zeros((4, 5, 3), dtype=np.float32),
                object_name=np.array("006_mustard_bottle"),
                object_translation=np.zeros((4, 3), dtype=np.float32),
                object_orientation=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (4, 1)),
                length=np.array(4),
                SIM_SUBSTEPS=np.array(10),
                DATA_SUBSTEPS=np.array(1),
                pregrasp_step=np.array(1),
                init_object_height=np.array(0.08, dtype=np.float32),
                init_object_quat=np.array([1, 0, 0, 0], dtype=np.float32),
            )

            retarget_file(src, dst, hand_dof=12)
            out = np.load(dst, allow_pickle=True)

            self.assertEqual(out["robot_qpos"].shape, (4, RM75_DRIVEN12_TOTAL_DOF))
            self.assertEqual(int(out["retarget_hand_dof"]), 12)

    def test_retarget_motion_dict_vividex_aligned_preserves_source_palm_path(self):
        source_palm = np.array(
            [[0.20, -0.50, 0.01], [0.18, -0.45, 0.04], [0.12, -0.30, 0.08], [0.10, -0.25, 0.12]],
            dtype=np.float32,
        )
        src = {
            "robot_qpos": np.zeros((4, 22), dtype=np.float32),
            "robot_jpos": np.zeros((4, 5, 3), dtype=np.float32),
            "object_translation": np.zeros((4, 3), dtype=np.float32),
            "pregrasp_step": np.array(1),
        }
        src["robot_jpos"][:, 0] = source_palm

        out = retarget_motion_dict(src, mode="vividex-aligned")

        self.assertEqual(out["robot_jpos"].shape, (4, 6, 3))
        self.assertEqual(str(out["retarget_mode"]), "vividex-aligned")
        np.testing.assert_allclose(out["robot_jpos"][:, 0], source_palm, atol=1e-6)

    def test_retarget_motion_dict_opposed_envelope_records_mode(self):
        src = {
            "robot_qpos": np.zeros((4, 22), dtype=np.float32),
            "robot_jpos": np.zeros((4, 5, 3), dtype=np.float32),
            "object_translation": np.zeros((4, 3), dtype=np.float32),
            "pregrasp_step": np.array(1),
        }

        out = retarget_motion_dict(src, mode="opposed-envelope")

        self.assertEqual(out["robot_jpos"].shape, (4, 6, 3))
        self.assertEqual(str(out["retarget_mode"]), "opposed-envelope")
        final_rel = out["robot_jpos"][-1] - src["object_translation"][-1]
        self.assertLess(final_rel[1, 1], -0.02)
        self.assertGreater(float(np.mean(final_rel[2:5, 1])), 0.02)

    def test_vividex_aligned_rejects_object_frame_palm_offset(self):
        src = {
            "robot_qpos": np.zeros((4, 22), dtype=np.float32),
            "robot_jpos": np.zeros((4, 5, 3), dtype=np.float32),
            "object_translation": np.zeros((4, 3), dtype=np.float32),
            "pregrasp_step": np.array(1),
        }

        with self.assertRaises(ValueError):
            retarget_motion_dict(src, palm_offset=np.zeros(3, dtype=np.float32), mode="vividex-aligned")


if __name__ == "__main__":
    unittest.main()

class RM75MotionPathTests(unittest.TestCase):
    def test_robot_specific_motion_path_is_preferred(self):
        from hand_imitation.env.motion_paths import resolve_motion_path

        repo = Path("/tmp/repo")
        path = resolve_motion_path(repo, "demo_seq", True, "rm75_inspire_right")

        self.assertEqual(path, repo / "norm_trajectories" / "rm75_inspire_right" / "demo_seq.npz")

    def test_rm75_motion_path_can_use_reference_subdir_override(self):
        from hand_imitation.env.motion_paths import resolve_motion_path

        repo = Path("/tmp/repo")
        with mock.patch.dict("os.environ", {"VIVIDEX_RM75_TRAJ_SUBDIR": "rm75_inspire_right_vividex_aligned"}):
            path = resolve_motion_path(repo, "demo_seq", True, "rm75_inspire_right")

        self.assertEqual(
            path,
            repo / "norm_trajectories" / "rm75_inspire_right_vividex_aligned" / "demo_seq.npz",
        )

    def test_default_robot_keeps_original_motion_path(self):
        from hand_imitation.env.motion_paths import resolve_motion_path

        repo = Path("/tmp/repo")
        path = resolve_motion_path(repo, "demo_seq", True, "allegro_hand_ur5")

        self.assertEqual(path, repo / "norm_trajectories" / "demo_seq.npz")

class RM75ReferenceHelpersTests(unittest.TestCase):
    def test_hand_qpos_slice_uses_robot_arm_dof(self):
        from hand_imitation.env.rl_env.reference_utils import slice_hand_qpos

        qpos = np.arange(13, dtype=np.float32)
        hand = slice_hand_qpos(qpos, arm_dof=7)

        np.testing.assert_array_equal(hand, np.arange(7, 13, dtype=np.float32))

class RM75MotionLoadingPolicyTests(unittest.TestCase):
    def test_missing_robot_specific_path_raises_by_default(self):
        from hand_imitation.env.motion_paths import resolve_existing_motion_path

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "norm_trajectories").mkdir()
            (repo / "norm_trajectories" / "demo_seq.npz").write_bytes(b"legacy")

            with self.assertRaises(FileNotFoundError):
                resolve_existing_motion_path(repo, "demo_seq", True, "rm75_inspire_right")

    def test_missing_robot_specific_path_can_opt_in_to_legacy_fallback(self):
        from hand_imitation.env.motion_paths import resolve_existing_motion_path

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            legacy = repo / "norm_trajectories" / "demo_seq.npz"
            legacy.parent.mkdir()
            legacy.write_bytes(b"legacy")

            path = resolve_existing_motion_path(
                repo,
                "demo_seq",
                True,
                "rm75_inspire_right",
                allow_legacy_fallback=True,
            )

            self.assertEqual(path, legacy)

class RM75ActionMappingTests(unittest.TestCase):
    def test_rm75_driven12_uses_vividex_aligned_wrist_mount(self):
        urdf = Path(__file__).resolve().parents[1] / "assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"
        root = ET.parse(urdf).getroot()
        joint = root.find(".//joint[@name='rh_base_joint']")
        self.assertIsNotNone(joint)

        origin = joint.find("origin")
        self.assertIsNotNone(origin)
        rpy = np.array([float(x) for x in origin.attrib["rpy"].split()], dtype=np.float32)

        np.testing.assert_allclose(rpy, np.array([-0.522573, 0.820826, 0.548226], dtype=np.float32), atol=1e-5)

    def test_rm75_full_urdf_hand_joint_order_matches_expand_mapping(self):
        urdf = Path(__file__).resolve().parents[1] / "assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"
        root = ET.parse(urdf).getroot()
        active_joints = [
            joint.attrib["name"]
            for joint in root.findall("joint")
            if joint.attrib.get("type") in {"revolute", "prismatic", "continuous"}
        ]

        self.assertEqual(
            active_joints[7:],
            [
                "rh_thumb_proximal_yaw_joint",
                "rh_thumb_proximal_pitch_joint",
                "rh_thumb_intermediate_joint",
                "rh_thumb_distal_joint",
                "rh_index_proximal_joint",
                "rh_index_intermediate_joint",
                "rh_middle_proximal_joint",
                "rh_middle_intermediate_joint",
                "rh_ring_proximal_joint",
                "rh_ring_intermediate_joint",
                "rh_pinky_proximal_joint",
                "rh_pinky_intermediate_joint",
            ],
        )
        self.assertEqual(root.findall(".//mimic"), [])

    def test_rm75_active6_urdf_hand_joint_order_matches_control_interface(self):
        urdf = Path(__file__).resolve().parents[1] / "assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_active6.urdf"
        root = ET.parse(urdf).getroot()
        active_joints = [
            joint.attrib["name"]
            for joint in root.findall("joint")
            if joint.attrib.get("type") in {"revolute", "prismatic", "continuous"}
        ]

        self.assertEqual(
            active_joints[7:],
            [
                "rh_thumb_proximal_yaw_joint",
                "rh_thumb_proximal_pitch_joint",
                "rh_index_proximal_joint",
                "rh_middle_proximal_joint",
                "rh_ring_proximal_joint",
                "rh_pinky_proximal_joint",
            ],
        )

    def test_rm75_pregrasp_target_matches_physical_hand_dof(self):
        from hand_imitation.env.rl_env.reference_utils import rm75_hand_qpos_for_qlimits

        six_dof_qlimits = np.tile(np.array([[0.0, 1.5]], dtype=np.float32), (6, 1))
        twelve_dof_qlimits = np.tile(np.array([[0.0, 1.5]], dtype=np.float32), (12, 1))

        six_dof_target = rm75_hand_qpos_for_qlimits(rm75_pregrasp_hand_qpos(), six_dof_qlimits)
        twelve_dof_target = rm75_hand_qpos_for_qlimits(rm75_pregrasp_hand_qpos(), twelve_dof_qlimits)

        self.assertEqual(six_dof_target.shape, (6,))
        self.assertEqual(twelve_dof_target.shape, (12,))
        np.testing.assert_allclose(six_dof_target, rm75_pregrasp_hand_qpos(), atol=1e-6)

    def test_rm75_zero_hand_action_maps_to_pregrasp_qpos(self):
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
        action = np.zeros(6, dtype=np.float32)

        qpos = recover_hand_action(action, qlimits, robot_name="rm75_inspire_right")

        self.assertEqual(qpos.shape, (12,))
        np.testing.assert_allclose(qpos[[0, 1, 4, 6, 8, 10]], rm75_pregrasp_hand_qpos(), atol=1e-6)

    def test_rm75_closed_hand_action_curls_coupled_intermediate_joints(self):
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

        qpos = recover_hand_action(np.ones(6, dtype=np.float32), qlimits, robot_name="rm75_inspire_right")

        self.assertGreater(qpos[2], 0.7)
        self.assertGreater(qpos[3], 0.35)
        self.assertGreater(qpos[5], 1.4)
        self.assertGreater(qpos[7], 1.4)

    def test_rm75_enforce_coupled_hand_qpos_updates_passive_segments(self):
        from hand_imitation.env.rl_env.reference_utils import enforce_rm75_coupled_hand_qpos

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
        qpos = np.zeros(19, dtype=np.float32)
        qpos[7:] = np.array(
            [1.2, 0.55, 0.1334, 0.0667, 1.34, 0.1673, 1.34, 0.1673, 1.34, 0.1673, 1.34, 0.1673],
            dtype=np.float32,
        )

        coupled = enforce_rm75_coupled_hand_qpos(qpos, arm_dof=7, qlimits=qlimits)

        self.assertGreater(coupled[9], 0.7)
        self.assertGreater(coupled[10], 0.35)
        self.assertGreater(coupled[12], 0.7)
        self.assertGreater(coupled[14], 0.7)

    def test_default_hand_action_mapping_keeps_symmetric_limits(self):
        from hand_imitation.env.rl_env.reference_utils import recover_hand_action

        qlimits = np.array([[0.0, 2.0]], dtype=np.float32)

        np.testing.assert_allclose(recover_hand_action(np.array([0.0]), qlimits, robot_name="allegro_hand_ur5"), [1.0])


class RM75TerminationPolicyTests(unittest.TestCase):
    def test_rm75_has_short_no_contact_grace_after_pregrasp(self):
        from hand_imitation.env.rl_env.reference_utils import should_terminate_for_contact_loss

        done, no_contact_steps = should_terminate_for_contact_loss(
            robot_name="rm75_inspire_right",
            is_contact=False,
            current_step=16,
            pregrasp_steps=15,
            no_contact_steps=0,
        )

        self.assertFalse(done)
        self.assertEqual(no_contact_steps, 1)

    def test_default_robot_still_terminates_immediately_without_contact(self):
        from hand_imitation.env.rl_env.reference_utils import should_terminate_for_contact_loss

        done, no_contact_steps = should_terminate_for_contact_loss(
            robot_name="allegro_hand_ur5",
            is_contact=False,
            current_step=16,
            pregrasp_steps=15,
            no_contact_steps=0,
        )

        self.assertTrue(done)
        self.assertEqual(no_contact_steps, 1)
