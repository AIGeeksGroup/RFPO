#!/usr/bin/env python3
"""Diagnose RM75/RH56 reference alignment in a ViViDex relocate env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--robot-name", default="rm75_inspire_right")
    parser.add_argument("--norm-traj", action="store_true")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--palm-offset", nargs=3, type=float, default=None)
    return parser.parse_args()


def link_positions(links) -> np.ndarray:
    return np.asarray([link.get_pose().p for link in links], dtype=np.float32)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "hand_imitation"))

    from hand_imitation.env.create_rm75_env import create_rm75_env
    from hand_imitation.env.motion_paths import resolve_existing_motion_path
    from hand_imitation.env.rl_env.rm75_relocate_env import RM75RelocateRLEnv
    from tools.retarget_rm75_inspire_reference import retarget_motion_dict

    if args.palm_offset is None:
        env = create_rm75_env(
            args.seq_name,
            use_gui=False,
            is_eval=False,
            is_vision=False,
            norm_traj=args.norm_traj,
            robot_name=args.robot_name,
            task_kwargs={"action": "relocate", "reward_kwargs": {}},
        )
    else:
        src_path = resolve_existing_motion_path(repo_root, args.seq_name, args.norm_traj, args.robot_name)
        src_npz = np.load(src_path, allow_pickle=True)
        src = {key: src_npz[key] for key in src_npz.files}
        motion = retarget_motion_dict(src, palm_offset=np.asarray(args.palm_offset, dtype=np.float32))
        motion["task_name"] = "relocate"
        env = RM75RelocateRLEnv(
            motion_file=motion,
            use_gui=False,
            is_eval=False,
            is_vision=False,
            norm_traj=args.norm_traj,
            robot_name=args.robot_name,
            task_kwargs={"action": "relocate", "reward_kwargs": {}},
            no_rgb=True,
            need_offscreen_render=False,
        )
    env._stage = args.stage
    env.reset()

    palm_pos = np.asarray(env.palm_link.get_pose().p, dtype=np.float32)
    tip_pos = link_positions(env.finger_tip_links)
    object_pos = np.asarray(env.manipulated_object.get_pose().p, dtype=np.float32)
    ref_pre = np.asarray(env.cur_reference_motion["robot_pregrasp_jpos"], dtype=np.float32)
    ref_final = np.asarray(ref_pre[-1], dtype=np.float32)
    ref_palm = ref_final[0]
    ref_tips = env._reference_hand_jpos(ref_final)

    palm_err = float(np.linalg.norm(palm_pos - ref_palm))
    tip_err = np.linalg.norm(tip_pos - ref_tips, axis=1)
    qpos = np.asarray(env.robot.get_qpos(), dtype=np.float32)
    qlimits = np.asarray(env.robot.get_qlimits(), dtype=np.float32)

    print("[diagnose] robot_name:", args.robot_name)
    print("[diagnose] seq_name:", args.seq_name)
    print("[diagnose] stage:", args.stage)
    print("[diagnose] dof:", env.robot.dof, "arm_dof:", env.arm_dof)
    print("[diagnose] object_pos:", np.round(object_pos, 5).tolist())
    print("[diagnose] robot_qpos:", np.round(qpos, 5).tolist())
    print("[diagnose] qlimits_min:", np.round(qlimits[:, 0], 5).tolist())
    print("[diagnose] qlimits_max:", np.round(qlimits[:, 1], 5).tolist())
    print("[diagnose] palm_pos:", np.round(palm_pos, 5).tolist())
    print("[diagnose] ref_palm:", np.round(ref_palm, 5).tolist())
    print("[diagnose] palm_err:", round(palm_err, 6))
    print("[diagnose] tip_err_mean:", round(float(np.mean(tip_err)), 6))
    print("[diagnose] tip_err_each:", np.round(tip_err, 6).tolist())
    print("[diagnose] tip_pos:")
    for i, row in enumerate(tip_pos):
        print(f"  tip_{i}: {np.round(row, 5).tolist()}")
    print("[diagnose] ref_tips:")
    for i, row in enumerate(ref_tips):
        print(f"  ref_{i}: {np.round(row, 5).tolist()}")

    env.close()


if __name__ == "__main__":
    main()
