#!/usr/bin/env python3
"""Generate RM75 + Inspire/RH56 reference files from ViViDex Allegro demos.

This is an intentionally conservative reference retargeter.  It fixes the
data-shape mismatch first: RM75/RH56 runs should consume robot-specific qpos
and a palm-plus-five-fingertip reference, instead of silently padding Allegro's
palm-plus-four-fingertip reference inside the reward function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_imitation.env.rl_env.reference_utils import (
    expand_rm75_active_hand_qpos,
    expand_rm75_mimic_active_hand_qpos,
)


RM75_ARM_DOF = 7
INSPIRE_HAND_DOF = 6
INSPIRE_DRIVEN_HAND_DOF = 12
RM75_TOTAL_DOF = RM75_ARM_DOF + INSPIRE_HAND_DOF
RM75_DRIVEN12_TOTAL_DOF = RM75_ARM_DOF + INSPIRE_DRIVEN_HAND_DOF

INSPIRE_HAND_OPEN_QPOS = np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)
INSPIRE_HAND_CLOSED_QPOS = np.array([0.85, 0.45, 1.00, 1.00, 1.00, 1.00], dtype=np.float32)
INSPIRE_HAND_PREGRASP_QPOS = np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)
RM75_REACHABLE_PALM_OFFSET = np.array([0.057, -0.089, 0.070], dtype=np.float32)
RM75_APPROACH_PREGRASP_DELTA = np.array([0.0, -0.05, 0.0], dtype=np.float32)
RM75_VIVIDEX_PALM_OFFSET = np.array([0.0826, -0.0509, 0.0244], dtype=np.float32)

# Inspire/RH56 fingertip templates measured from the actual SAPIEN RM75/RH56
# model around the configured ViViDex-like palm pose.  The first implementation
# used hand-written offsets whose z axis was flipped relative to the loaded
# hand: reference tips sat above the palm while the real fingertips sit below
# it.  That made the pregrasp and hand-mimic rewards chase unreachable targets.
RM75_PREGRASP_TIP_OFFSETS = np.array(
    [
        [-0.064913, 0.126726, 0.094183],
        [-0.017154, 0.214302, 0.015665],
        [-0.015191, 0.216504, -0.006404],
        [-0.012289, 0.211674, -0.029250],
        [-0.007571, 0.200894, -0.050682],
    ],
    dtype=np.float32,
)
RM75_GRASP_TIP_OFFSETS = np.array(
    [
        [-0.047259, 0.156218, 0.002340],
        [-0.046435, 0.189656, 0.007312],
        [-0.047128, 0.189209, -0.014010],
        [-0.042375, 0.185511, -0.035343],
        [-0.057964, 0.127802, -0.047977],
    ],
    dtype=np.float32,
)

# Object-frame envelope template for mustard-bottle style side grasps.  The
# previous RM75 templates placed all fingertips on the same +Y side of the
# bottle after applying the current wrist mount.  The original ViViDex Allegro
# reference instead uses an opposed structure: thumb on one side, main fingers
# on the other.  This template encodes that topology directly in the object
# frame while keeping a palm offset close to the source ViViDex demonstration.
RM75_OPPOSED_PREGRASP_REL_OFFSETS = np.array(
    [
        [0.020, -0.070, 0.075],
        [0.040, 0.070, 0.060],
        [0.036, 0.075, 0.030],
        [0.032, 0.065, 0.000],
        [0.020, 0.045, -0.025],
    ],
    dtype=np.float32,
)
RM75_OPPOSED_GRASP_REL_OFFSETS = np.array(
    [
        [-0.018, -0.034, 0.016],
        [0.012, 0.032, 0.018],
        [0.012, 0.030, -0.006],
        [0.014, 0.022, -0.025],
        [0.004, 0.000, -0.038],
    ],
    dtype=np.float32,
)


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def rm75_pregrasp_hand_qpos() -> np.ndarray:
    return INSPIRE_HAND_PREGRASP_QPOS.copy()


def expand_inspire_active_hand_qpos(
    active_qpos: np.ndarray,
    hand_dof: int = INSPIRE_HAND_DOF,
    native_mimic: bool = False,
) -> np.ndarray:
    active_qpos = np.asarray(active_qpos, dtype=np.float32)
    hand_dof = int(hand_dof)
    if hand_dof == INSPIRE_HAND_DOF:
        return active_qpos.copy()
    if hand_dof == INSPIRE_DRIVEN_HAND_DOF:
        if native_mimic:
            return expand_rm75_mimic_active_hand_qpos(active_qpos)
        return expand_rm75_active_hand_qpos(active_qpos)
    raise ValueError(f"Unsupported RM75/RH56 hand_dof: {hand_dof}")


def build_rm75_robot_jpos_from_allegro(robot_jpos: np.ndarray) -> np.ndarray:
    """Convert Allegro palm+4 fingertips to RM75/RH56 palm+5 fingertips.

    Input order is the ViViDex Allegro convention:
    palm, thumb, index, middle, ring.

    Output order is:
    palm, thumb, index, middle, ring, pinky.

    Without calibrated RH56 retargeting, the pinky target is initialized from
    the middle/ring envelope.  This is better than reward-time implicit padding
    because the generated file is explicit and can be replaced later by a true
    IK-retargeted reference.
    """

    robot_jpos = _as_float32(robot_jpos)
    if robot_jpos.ndim != 3 or robot_jpos.shape[1:] != (5, 3):
        raise ValueError(f"Expected Allegro robot_jpos with shape (T, 5, 3), got {robot_jpos.shape}")

    out = np.empty((robot_jpos.shape[0], 6, 3), dtype=np.float32)
    out[:, :5, :] = robot_jpos
    out[:, 5, :] = 0.5 * (robot_jpos[:, 3, :] + robot_jpos[:, 4, :])
    return out


def build_rm75_vividex_aligned_robot_jpos(
    source_robot_jpos: np.ndarray,
    object_translation: np.ndarray,
    pregrasp_step: int | None = None,
) -> np.ndarray:
    """Build an RM75/RH56 reference aligned to the original ViViDex palm path.

    The source demonstrations contain Allegro palm-plus-four-fingertip targets.
    For RM75/RH56 we preserve the source palm target exactly, then generate five
    Inspire fingertip targets around that moving palm.  This makes the robot
    initialization and pregrasp target use the same hand/object geometry as the
    original ViViDex setup while avoiding unreachable Allegro fingertip targets.
    """

    source_robot_jpos = _as_float32(source_robot_jpos)
    if source_robot_jpos.ndim != 3 or source_robot_jpos.shape[1] < 1 or source_robot_jpos.shape[2] != 3:
        raise ValueError(f"Expected source_robot_jpos with shape (T, J, 3), got {source_robot_jpos.shape}")

    object_translation = _as_float32(object_translation)
    if object_translation.ndim != 2 or object_translation.shape[1] != 3:
        raise ValueError(f"Expected object_translation with shape (T, 3), got {object_translation.shape}")
    if len(source_robot_jpos) != len(object_translation):
        raise ValueError(
            "source_robot_jpos and object_translation must have the same length, "
            f"got {len(source_robot_jpos)} and {len(object_translation)}"
        )

    if len(object_translation) == 1:
        alpha = np.zeros(1, dtype=np.float32)
    else:
        if pregrasp_step is None:
            alpha = np.linspace(0.0, 1.0, len(object_translation), dtype=np.float32)
        else:
            alpha = np.zeros(len(object_translation), dtype=np.float32)
            start = min(max(int(pregrasp_step), 0), len(object_translation) - 1)
            tail_len = len(object_translation) - start
            alpha[start:] = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)

    tip_offsets = (1.0 - alpha[:, None, None]) * RM75_PREGRASP_TIP_OFFSETS + alpha[:, None, None] * RM75_GRASP_TIP_OFFSETS
    out = np.empty((len(object_translation), 6, 3), dtype=np.float32)
    out[:, 0, :] = source_robot_jpos[:, 0, :]
    out[:, 1:, :] = out[:, None, 0, :] + tip_offsets
    return out


def build_rm75_reachable_robot_jpos(
    object_translation: np.ndarray,
    pregrasp_step: int | None = None,
    palm_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Build an RM75/RH56 palm-plus-five-fingertip reference.

    The original ViViDex trajectories store Allegro fingertip positions.  Those
    positions are geometrically incompatible with the RH56 hand and make the
    pregrasp reward chase unreachable targets.  This helper keeps the object
    path from the demonstration but replaces the hand reference with a reachable
    RH56 grasp template in the object frame.
    """

    object_translation = _as_float32(object_translation)
    if object_translation.ndim != 2 or object_translation.shape[1] != 3:
        raise ValueError(f"Expected object_translation with shape (T, 3), got {object_translation.shape}")

    if len(object_translation) == 1:
        alpha = np.zeros(1, dtype=np.float32)
    else:
        if pregrasp_step is None:
            alpha = np.linspace(0.0, 1.0, len(object_translation), dtype=np.float32)
        else:
            alpha = np.zeros(len(object_translation), dtype=np.float32)
            start = min(max(int(pregrasp_step), 0), len(object_translation) - 1)
            tail_len = len(object_translation) - start
            alpha[start:] = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)
    tip_offsets = (1.0 - alpha[:, None, None]) * RM75_PREGRASP_TIP_OFFSETS + alpha[:, None, None] * RM75_GRASP_TIP_OFFSETS

    out = np.empty((len(object_translation), 6, 3), dtype=np.float32)
    if palm_offset is None:
        raise ValueError(
            "palm_offset is required for object-frame template generation. "
            "Use build_rm75_vividex_aligned_robot_jpos to follow the original ViViDex palm path."
        )
    palm_offset = _as_float32(palm_offset)
    if palm_offset.shape != (3,):
        raise ValueError(f"Expected palm_offset with shape (3,), got {palm_offset.shape}")

    out[:, 0, :] = object_translation + palm_offset
    out[:, 1:, :] = out[:, None, 0, :] + tip_offsets
    return out


def build_rm75_opposed_envelope_robot_jpos(
    object_translation: np.ndarray,
    pregrasp_step: int | None = None,
    palm_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Build an object-frame opposed-grasp reference for RM75/RH56.

    Output order is palm, thumb, index, middle, ring, pinky.  Unlike the raw
    RH56 fingertip-offset template, this reference defines the final envelope in
    the object frame so thumb and main fingers are on opposite sides of the
    bottle.  That mirrors the original ViViDex grasp topology and gives the
    staged reward a physically meaningful target.
    """

    object_translation = _as_float32(object_translation)
    if object_translation.ndim != 2 or object_translation.shape[1] != 3:
        raise ValueError(f"Expected object_translation with shape (T, 3), got {object_translation.shape}")

    if len(object_translation) == 1:
        alpha = np.zeros(1, dtype=np.float32)
    else:
        if pregrasp_step is None:
            alpha = np.linspace(0.0, 1.0, len(object_translation), dtype=np.float32)
        else:
            alpha = np.zeros(len(object_translation), dtype=np.float32)
            start = min(max(int(pregrasp_step), 0), len(object_translation) - 1)
            tail_len = len(object_translation) - start
            alpha[start:] = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)

    if palm_offset is None:
        palm_offset = RM75_VIVIDEX_PALM_OFFSET
    palm_offset = _as_float32(palm_offset)
    if palm_offset.shape != (3,):
        raise ValueError(f"Expected palm_offset with shape (3,), got {palm_offset.shape}")

    rel_offsets = (
        (1.0 - alpha[:, None, None]) * RM75_OPPOSED_PREGRASP_REL_OFFSETS
        + alpha[:, None, None] * RM75_OPPOSED_GRASP_REL_OFFSETS
    )
    out = np.empty((len(object_translation), 6, 3), dtype=np.float32)
    out[:, 0, :] = object_translation + palm_offset
    out[:, 1:, :] = object_translation[:, None, :] + rel_offsets
    return out


def build_rm75_seed_qpos(
    robot_qpos: np.ndarray,
    pregrasp_step: int | None = None,
    hand_dof: int = INSPIRE_HAND_DOF,
    native_mimic: bool = False,
    closed_hand_qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Create a stable RM75/RH56 seed qpos sequence.

    The old 22-DoF Allegro qpos is not kinematically compatible with RM75/RH56.
    We therefore keep RM75 at its configured neutral arm pose and provide a
    smooth open-to-closed hand schedule around the original pregrasp phase.
    RL still controls the arm by Cartesian velocity; this qpos is only the demo
    seed used for visualization/pregrasp initialization paths.
    """

    robot_qpos = _as_float32(robot_qpos)
    if robot_qpos.ndim != 2:
        raise ValueError(f"Expected robot_qpos with shape (T, D), got {robot_qpos.shape}")

    hand_dof = int(hand_dof)
    if hand_dof not in (INSPIRE_HAND_DOF, INSPIRE_DRIVEN_HAND_DOF):
        raise ValueError(f"Unsupported RM75/RH56 hand_dof: {hand_dof}")

    out = np.zeros((robot_qpos.shape[0], RM75_ARM_DOF + hand_dof), dtype=np.float32)
    if len(out) == 1:
        alpha = np.ones(1, dtype=np.float32)
    else:
        if pregrasp_step is None:
            alpha = np.linspace(0.0, 1.0, len(out), dtype=np.float32)
        else:
            alpha = np.zeros(len(out), dtype=np.float32)
            start = min(max(int(pregrasp_step), 0), len(out) - 1)
            tail_len = len(out) - start
            alpha[start:] = np.linspace(0.0, 1.0, tail_len, dtype=np.float32)
    closed_hand_qpos = INSPIRE_HAND_CLOSED_QPOS if closed_hand_qpos is None else _as_float32(closed_hand_qpos)
    if closed_hand_qpos.shape != (INSPIRE_HAND_DOF,):
        raise ValueError(f"Expected closed_hand_qpos with shape ({INSPIRE_HAND_DOF},), got {closed_hand_qpos.shape}")
    active_hand_qpos = (1.0 - alpha[:, None]) * INSPIRE_HAND_PREGRASP_QPOS + alpha[:, None] * closed_hand_qpos
    out[:, RM75_ARM_DOF:] = np.asarray(
        [
            expand_inspire_active_hand_qpos(
                qpos,
                hand_dof=hand_dof,
                native_mimic=native_mimic,
            )
            for qpos in active_hand_qpos
        ],
        dtype=np.float32,
    )
    return out


def build_rm75_native_neutral_qpos(robot_qpos: np.ndarray, hand_dof: int = INSPIRE_DRIVEN_HAND_DOF) -> np.ndarray:
    robot_qpos = _as_float32(robot_qpos)
    if robot_qpos.ndim != 2:
        raise ValueError(f"Expected robot_qpos with shape (T, D), got {robot_qpos.shape}")
    return np.zeros((robot_qpos.shape[0], RM75_ARM_DOF + int(hand_dof)), dtype=np.float32)


def _normalize_retarget_mode(mode: str) -> str:
    mode = str(mode).strip().lower().replace("_", "-")
    if mode in {"reachable", "object-frame", "objectframe"}:
        return "reachable"
    if mode in {"vividex-aligned", "vividex", "aligned"}:
        return "vividex-aligned"
    if mode in {"opposed-envelope", "opposed", "envelope", "mustard-envelope"}:
        return "opposed-envelope"
    if mode in {"native-object-only", "native", "object-only"}:
        return "native-object-only"
    raise ValueError(f"Unsupported RM75 retarget mode: {mode}")


def retarget_motion_dict(
    src: dict[str, np.ndarray],
    palm_offset: np.ndarray | None = None,
    mode: str = "reachable",
    hand_dof: int = INSPIRE_HAND_DOF,
    native_mimic: bool = False,
    seed_pinky_qpos: float | None = None,
) -> dict[str, np.ndarray]:
    if "robot_jpos" not in src or "robot_qpos" not in src:
        raise KeyError("Input trajectory must contain robot_jpos and robot_qpos")

    mode = _normalize_retarget_mode(mode)
    out = {key: value for key, value in src.items()}
    pregrasp_step = int(src.get("pregrasp_step", 0))
    if mode == "reachable":
        if palm_offset is None:
            palm_offset = RM75_REACHABLE_PALM_OFFSET
        out["robot_jpos"] = build_rm75_reachable_robot_jpos(
            src["object_translation"],
            pregrasp_step,
            palm_offset=palm_offset,
        )
        retarget_note = "RM75/RH56 object-frame reachable grasp reference with approach-side pregrasp"
    elif mode == "vividex-aligned":
        if palm_offset is not None:
            raise ValueError("--palm-offset is only valid for --mode reachable")
        out["robot_jpos"] = build_rm75_vividex_aligned_robot_jpos(
            src["robot_jpos"],
            src["object_translation"],
            pregrasp_step,
        )
        retarget_note = (
            "RM75/RH56 ViViDex-aligned reference preserving source palm path "
            "with robot-specific fingertip template"
        )
    elif mode == "opposed-envelope":
        out["robot_jpos"] = build_rm75_opposed_envelope_robot_jpos(
            src["object_translation"],
            pregrasp_step,
            palm_offset=palm_offset,
        )
        retarget_note = (
            "RM75/RH56 object-frame opposed-envelope grasp reference with "
            "thumb and main fingers on opposite object sides"
        )
    else:
        if palm_offset is not None:
            raise ValueError("--palm-offset is not valid for --mode native-object-only")
        out["robot_jpos"] = build_rm75_robot_jpos_from_allegro(src["robot_jpos"])
        retarget_note = (
            "RM75/RH56 native object-only reference preserving source palm/tip positions; "
            "hand qpos is neutral and the policy controls all loaded hand joints"
        )
    closed_hand_qpos = None
    if seed_pinky_qpos is not None:
        closed_hand_qpos = INSPIRE_HAND_CLOSED_QPOS.copy()
        closed_hand_qpos[5] = float(seed_pinky_qpos)

    if mode == "native-object-only":
        out["robot_qpos"] = build_rm75_native_neutral_qpos(src["robot_qpos"], hand_dof=hand_dof)
    else:
        out["robot_qpos"] = build_rm75_seed_qpos(
            src["robot_qpos"],
            int(src.get("pregrasp_step", 0)),
            hand_dof=hand_dof,
            native_mimic=native_mimic,
            closed_hand_qpos=closed_hand_qpos,
        )
    out["retarget_robot"] = np.array("rm75_inspire_right")
    out["retarget_mode"] = np.array(mode)
    out["retarget_hand_dof"] = np.array(int(hand_dof), dtype=np.int32)
    if seed_pinky_qpos is not None:
        out["retarget_seed_pinky_qpos"] = np.array(float(seed_pinky_qpos), dtype=np.float32)
    out["rm75_template_approach_delta"] = RM75_APPROACH_PREGRASP_DELTA.copy()
    out["retarget_note"] = np.array(retarget_note)
    return out


def retarget_file(
    src_path: Path | str,
    dst_path: Path | str,
    palm_offset: np.ndarray | None = None,
    mode: str = "reachable",
    hand_dof: int = INSPIRE_HAND_DOF,
    native_mimic: bool = False,
    seed_pinky_qpos: float | None = None,
) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    src_npz = np.load(src_path, allow_pickle=True)
    src = {key: src_npz[key] for key in src_npz.files}
    out = retarget_motion_dict(
        src,
        palm_offset=palm_offset,
        mode=mode,
        hand_dof=hand_dof,
        native_mimic=native_mimic,
        seed_pinky_qpos=seed_pinky_qpos,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_path, **out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Source Allegro trajectory .npz or directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination .npz or directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--palm-offset", nargs=3, type=float, default=None)
    parser.add_argument(
        "--hand-dof",
        type=int,
        choices=(INSPIRE_HAND_DOF, INSPIRE_DRIVEN_HAND_DOF),
        default=INSPIRE_HAND_DOF,
        help="Physical RM75/RH56 hand qpos width to write. Use 12 for driven12/joint12 URDFs.",
    )
    parser.add_argument(
        "--native-mimic-hand-qpos",
        action="store_true",
        help="For 12-DoF native mimic URDFs, write driver joints and derive mimic joints instead of driven12 qpos.",
    )
    parser.add_argument(
        "--seed-pinky-qpos",
        type=float,
        default=None,
        help="Override the final active pinky qpos in retargeted robot_qpos; use 0.03 to keep pinky open.",
    )
    parser.add_argument(
        "--mode",
        choices=("reachable", "object-frame", "vividex-aligned", "opposed-envelope", "native-object-only"),
        default="reachable",
        help=(
            "RM75 reference mode. 'reachable' keeps the current object-frame "
            "RH56 grasp template; 'vividex-aligned' preserves the original "
            "ViViDex palm path and only replaces fingertip targets; "
            "'opposed-envelope' creates an object-frame thumb-vs-fingers "
            "envelope reference; 'native-object-only' avoids hand qpos "
            "templates and lets native RM75 hand actions learn the grasp."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src.is_dir():
        args.dst.mkdir(parents=True, exist_ok=True)
        src_files = sorted(args.src.glob("*.npz"))
        if not src_files:
            raise FileNotFoundError(f"No .npz files found in {args.src}")
        for src_file in src_files:
            dst_file = args.dst / src_file.name
            if dst_file.exists() and not args.overwrite:
                print(f"[skip] {dst_file}")
                continue
            retarget_file(
                src_file,
                dst_file,
                palm_offset=args.palm_offset,
                mode=args.mode,
                hand_dof=args.hand_dof,
                native_mimic=args.native_mimic_hand_qpos,
                seed_pinky_qpos=args.seed_pinky_qpos,
            )
            print(f"[ok] {src_file.name} -> {dst_file}")
    else:
        if args.dst.exists() and not args.overwrite:
            raise FileExistsError(f"{args.dst} exists; pass --overwrite to replace it")
        retarget_file(
            args.src,
            args.dst,
            palm_offset=args.palm_offset,
            mode=args.mode,
            hand_dof=args.hand_dof,
            native_mimic=args.native_mimic_hand_qpos,
            seed_pinky_qpos=args.seed_pinky_qpos,
        )
        print(f"[ok] {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
