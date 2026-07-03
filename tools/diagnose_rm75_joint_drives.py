#!/usr/bin/env python3
"""Check whether each RM75/RH56 active joint follows SAPIEN drive targets."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import sapien.core as sapien


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-hand-control",
        action="store_true",
        help="Send the same expanded 12-DoF RM75 hand action used by native training.",
    )
    parser.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Keep RM75 hand self collision enabled instead of matching the RL env default.",
    )
    parser.add_argument(
        "--enforce-mimic",
        action="store_true",
        help="Apply the RM75 URDF mimic formulas to qpos before every simulation step.",
    )
    parser.add_argument("--close", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "hand_imitation"))

    from hand_imitation.utils.common_robot_utils import load_robot
    from hand_imitation.env.rl_env.reference_utils import (
        enforce_rm75_mimic_hand_qpos,
        expand_rm75_active_hand_action,
        recover_hand_action,
    )

    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 240)
    robot = load_robot(
        scene,
        "rm75_inspire_right",
        disable_self_collision=not args.enable_self_collision,
    )
    robot.set_pose(sapien.Pose([0, 0, 0]))

    joints = robot.get_active_joints()
    print("dof", robot.dof)
    print("active_joints", [joint.get_name() for joint in joints])
    print("joint_types", [str(joint.type) for joint in joints])
    print("qlimits", np.round(robot.get_qlimits(), 4).tolist())

    qpos = np.zeros(robot.dof, dtype=np.float32)
    qpos[:7] = 0.0
    if args.native_hand_control:
        zero_action = expand_rm75_active_hand_action(np.zeros(6, dtype=np.float32), robot.dof - 7)
        close_action = expand_rm75_active_hand_action(
            np.ones(6, dtype=np.float32) * float(args.close),
            robot.dof - 7,
        )
    else:
        zero_action = np.zeros(6, dtype=np.float32)
        close_action = np.ones(6, dtype=np.float32) * float(args.close)
    qpos[7:] = recover_hand_action(zero_action, robot.get_qlimits()[7:], "rm75_inspire_right")
    robot.set_qpos(qpos)
    robot.set_drive_target(qpos)

    target = qpos.copy()
    target[7:] = recover_hand_action(close_action, robot.get_qlimits()[7:], "rm75_inspire_right")
    print("mode", "native12" if args.native_hand_control else "logical6")
    print("enforce_mimic", bool(args.enforce_mimic))
    print("target_hand_qpos", np.round(target[7:], 4).tolist())
    robot.set_drive_target(target)
    robot.set_drive_velocity_target(np.zeros(robot.dof, dtype=np.float32))

    checkpoints = {0, 1, 4, 9, 19, 39, max(int(args.steps) - 1, 0)}
    for step in range(max(int(args.steps), 1)):
        if args.enforce_mimic:
            robot.set_qpos(enforce_rm75_mimic_hand_qpos(robot.get_qpos(), 7, robot.get_qlimits()[7:]))
        robot.set_qf(robot.compute_passive_force(external=False, coriolis_and_centrifugal=False))
        scene.step()
        if step in checkpoints:
            print(
                "step",
                step + 1,
                "hand_qpos",
                np.round(robot.get_qpos()[7:], 4).tolist(),
                "error",
                np.round(target[7:] - robot.get_qpos()[7:], 4).tolist(),
            )


if __name__ == "__main__":
    main()
