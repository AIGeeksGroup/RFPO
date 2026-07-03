#!/usr/bin/env python3
"""Preflight checks for RM75/RH56 training runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--robot-name", default="rm75_inspire_right")
    parser.add_argument("--norm-traj", action="store_true")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--profile", choices=("active6", "driven12", "auto"), default="active6")
    parser.add_argument("--max-palm-err", type=float, default=0.050)
    parser.add_argument("--max-tip-err", type=float, default=0.075)
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data:
        return None
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value.tolist()


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    os.environ.setdefault("VIVIDEX_HEADLESS_NO_RENDER", "1")

    urdf_dir = repo_root / "assets/robot/rm75_inspire_right/urdf"
    active6_urdf = (urdf_dir / "rm75_inspire_hand_right_active6.urdf").resolve()
    driven12_urdf = (urdf_dir / "rm75_inspire_hand_right_driven12.urdf").resolve()
    allowed_urdfs_by_profile = {
        "active6": {
            active6_urdf,
            (urdf_dir / "rm75_inspire_hand_right_active6_nocyl.urdf").resolve(),
            (urdf_dir / "rm75_inspire_hand_right_active6_curled.urdf").resolve(),
            (urdf_dir / "rm75_inspire_hand_right_active6_mountfix_nocyl.urdf").resolve(),
        },
        "driven12": {
            driven12_urdf,
            (urdf_dir / "rm75_inspire_hand_right_driven12_nocyl.urdf").resolve(),
        },
    }
    override = os.environ.get("VIVIDEX_RM75_URDF_OVERRIDE")
    override_path = Path(override).expanduser().resolve() if override else None
    profile = args.profile
    if profile == "auto":
        if override_path in allowed_urdfs_by_profile["driven12"]:
            profile = "driven12"
        else:
            profile = "active6"
    expected_robot_dof = 19 if profile == "driven12" else 13
    expected_physical_hand_dof = expected_robot_dof - 7
    allowed_urdfs = allowed_urdfs_by_profile[profile]

    try:
        from hand_imitation.env.create_rm75_env import create_rm75_env
        from hand_imitation.env.motion_paths import resolve_existing_motion_path
    except ModuleNotFoundError as exc:
        if exc.name == "sapien":
            print(
                "Missing SAPIEN in this Python. Run with "
                f"{repo_root / '.venv-vividex/bin/python'} or set VIVIDEX_VENV.",
                file=sys.stderr,
            )
            return 2
        raise

    motion_path = resolve_existing_motion_path(
        repo_root,
        args.seq_name,
        args.norm_traj,
        args.robot_name,
        allow_legacy_fallback=False,
    )
    motion = np.load(motion_path, allow_pickle=True)

    env = None
    try:
        env = create_rm75_env(
            args.seq_name,
            use_gui=False,
            is_eval=False,
            is_vision=False,
            norm_traj=args.norm_traj,
            robot_name=args.robot_name,
            task_kwargs={"action": "relocate", "reward_kwargs": {}},
        )
        env._stage = args.stage
        env.reset()

        palm_pos = np.asarray(env.palm_link.get_pose().p, dtype=np.float32)
        tip_pos = np.asarray([link.get_pose().p for link in env.finger_tip_links], dtype=np.float32)
        ref_pre = np.asarray(env.cur_reference_motion["robot_pregrasp_jpos"], dtype=np.float32)
        ref_final = ref_pre[-1]
        ref_palm = ref_final[0]
        ref_tips = env._reference_hand_jpos(ref_final)

        palm_err = float(np.linalg.norm(palm_pos - ref_palm))
        tip_err_each = np.linalg.norm(tip_pos - ref_tips, axis=1)
        tip_err_mean = float(np.mean(tip_err_each))

        robot_qpos_shape = tuple(motion["robot_qpos"].shape) if "robot_qpos" in motion else None
        robot_jpos_shape = tuple(motion["robot_jpos"].shape) if "robot_jpos" in motion else None
        active_joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        hand_joint_names = active_joint_names[int(env.arm_dof) :]

        checks = {
            f"{profile}_urdf": override_path is None or override_path in allowed_urdfs,
            f"robot_dof_{expected_robot_dof}": int(env.robot.dof) == expected_robot_dof,
            "arm_dof_7": int(env.arm_dof) == 7,
            "control_hand_dof_6": int(env.robot_info.hand_dof) == 6,
            f"physical_hand_dof_{expected_physical_hand_dof}": int(env.robot.dof) - int(env.arm_dof) == expected_physical_hand_dof,
            "action_dim_12": int(env.action_dim) == 12,
            f"trajectory_qpos_{expected_robot_dof}": robot_qpos_shape is not None and robot_qpos_shape[-1] == expected_robot_dof,
            "trajectory_jpos_6x3": robot_jpos_shape is not None and robot_jpos_shape[-2:] == (6, 3),
            "palm_err_ok": palm_err <= float(args.max_palm_err),
            "tip_err_ok": tip_err_mean <= float(args.max_tip_err),
        }

        summary = {
            "seq_name": args.seq_name,
            "motion_path": motion_path,
            "robot_name": args.robot_name,
            "profile": profile,
            "active6_urdf": active6_urdf,
            "driven12_urdf": driven12_urdf,
            "allowed_urdfs": sorted(str(path) for path in allowed_urdfs),
            "urdf_override": override_path,
            "retarget_robot": _npz_scalar(motion, "retarget_robot"),
            "retarget_mode": _npz_scalar(motion, "retarget_mode"),
            "retarget_hand_dof": _npz_scalar(motion, "retarget_hand_dof"),
            "retarget_note": _npz_scalar(motion, "retarget_note"),
            "robot_dof": int(env.robot.dof),
            "arm_dof": int(env.arm_dof),
            "control_hand_dof": int(env.robot_info.hand_dof),
            "physical_hand_dof": int(env.robot.dof) - int(env.arm_dof),
            "action_dim": int(env.action_dim),
            "obs_dim": int(env.obs_dim),
            "trajectory_robot_qpos_shape": robot_qpos_shape,
            "trajectory_robot_jpos_shape": robot_jpos_shape,
            "hand_joint_names": hand_joint_names,
            "palm_err": palm_err,
            "tip_err_mean": tip_err_mean,
            "tip_err_each": tip_err_each,
            "checks": checks,
        }
        print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))

        failed = [name for name, ok in checks.items() if not ok]
        if failed and not args.no_strict:
            print(f"[rm75-check] failed checks: {', '.join(failed)}", file=sys.stderr)
            return 1
        print("[rm75-check] ok")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
