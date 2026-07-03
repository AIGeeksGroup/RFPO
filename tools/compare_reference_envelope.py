#!/usr/bin/env python3
"""Compare reference hand envelopes around the object."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "hand_imitation"))

from hand_imitation.utils.ycb_object_utils import YCB_SIZE


def _parse_args() -> argparse.Namespace:
    repo = REPO_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument(
        "--allegro-path",
        type=Path,
        default=repo / "norm_trajectories" / f"{DEFAULT_SEQ}.npz",
    )
    parser.add_argument(
        "--rm75-path",
        type=Path,
        default=repo
        / "norm_trajectories"
        / "rm75_inspire_right_native12_mimic_stage0_refine_truth_best"
        / f"{DEFAULT_SEQ}.npz",
    )
    parser.add_argument("--object-name", default="mustard_bottle")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _angle_deg(vec: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(vec[1]), float(vec[0]))))


def _pairwise_xy_span(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    xy = points[:, :2]
    deltas = xy[:, None, :] - xy[None, :, :]
    return float(np.max(np.linalg.norm(deltas, axis=-1)))


def _opposition_deg(rel_tip_pos: np.ndarray) -> float:
    if rel_tip_pos.shape[0] < 2:
        return 0.0
    thumb = rel_tip_pos[0, :2]
    thumb_norm = float(np.linalg.norm(thumb))
    if thumb_norm < 1e-6:
        return 0.0
    best = 0.0
    for other in rel_tip_pos[1:, :2]:
        other_norm = float(np.linalg.norm(other))
        if other_norm < 1e-6:
            continue
        cosine = float(np.clip(np.dot(thumb, other) / (thumb_norm * other_norm), -1.0, 1.0))
        best = max(best, float(math.degrees(math.acos(cosine))))
    return best


def _ellipse_values(rel_tips: np.ndarray, half_xy: tuple[float, float]) -> np.ndarray:
    denom = np.maximum(np.asarray(half_xy, dtype=np.float32), 1e-6)
    return (rel_tips[:, 0] / denom[0]) ** 2 + (rel_tips[:, 1] / denom[1]) ** 2


def _summarize_frame(
    points: np.ndarray,
    object_pos: np.ndarray,
    half_height: float,
    half_xy: tuple[float, float],
) -> dict:
    palm = points[0]
    tips = points[1:]
    rel_palm = palm - object_pos
    rel_tips = tips - object_pos[None, :]
    tip_xy = np.linalg.norm(rel_tips[:, :2], axis=1)
    tip_z = rel_tips[:, 2]
    ellipse = _ellipse_values(rel_tips, half_xy)
    return {
        "palm_rel_xyz": rel_palm.tolist(),
        "palm_xy_radius": float(np.linalg.norm(rel_palm[:2])),
        "palm_angle_deg": _angle_deg(rel_palm[:2]),
        "tip_rel_xyz": rel_tips.tolist(),
        "tip_xy_radius_min_mean_max": [
            float(np.min(tip_xy)),
            float(np.mean(tip_xy)),
            float(np.max(tip_xy)),
        ],
        "tip_z_min_mean_max": [
            float(np.min(tip_z)),
            float(np.mean(tip_z)),
            float(np.max(tip_z)),
        ],
        "tip_z_over_object_half_height_min_mean_max": [
            float(np.min(tip_z) / half_height),
            float(np.mean(tip_z) / half_height),
            float(np.max(tip_z) / half_height),
        ],
        "tip_angles_deg": [_angle_deg(v[:2]) for v in rel_tips],
        "tip_xy_ellipse_value_min_mean_max": [
            float(np.min(ellipse)),
            float(np.mean(ellipse)),
            float(np.max(ellipse)),
        ],
        "tip_xy_inside_object_ellipse_count": int(np.sum(ellipse < 1.0)),
        "tip_pairwise_xy_span": _pairwise_xy_span(tips),
        "thumb_to_finger_opposition_max_deg": _opposition_deg(rel_tips),
    }


def _range_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    return [float(np.min(values)), float(np.mean(values)), float(np.max(values))]


def _summarize_window(
    robot_jpos: np.ndarray,
    object_translation: np.ndarray,
    half_height: float,
    half_xy: tuple[float, float],
) -> dict:
    rel_tips = robot_jpos[:, 1:, :] - object_translation[:, None, :]
    tip_xy = np.linalg.norm(rel_tips[:, :, :2], axis=-1)
    tip_z = rel_tips[:, :, 2]
    ellipse = _ellipse_values(rel_tips.reshape(-1, 3), half_xy)
    spans = np.asarray([_pairwise_xy_span(frame[1:]) for frame in robot_jpos], dtype=np.float32)
    opposition = np.asarray(
        [_opposition_deg(frame[1:] - obj[None, :]) for frame, obj in zip(robot_jpos, object_translation)],
        dtype=np.float32,
    )
    return {
        "tip_xy_radius_min_mean_max": _range_stats(tip_xy),
        "tip_z_min_mean_max": _range_stats(tip_z),
        "tip_z_over_half_height_min_mean_max": [
            float(np.min(tip_z) / half_height),
            float(np.mean(tip_z) / half_height),
            float(np.max(tip_z) / half_height),
        ],
        "tip_xy_ellipse_value_min_mean_max": _range_stats(ellipse),
        "tip_xy_inside_object_ellipse_fraction": float(np.mean(ellipse < 1.0)),
        "pairwise_xy_span_min_mean_max": _range_stats(spans),
        "opposition_deg_min_mean_max": _range_stats(opposition),
    }


def _summarize_traj(
    path: Path,
    point_names: list[str],
    half_height: float,
    half_xy: tuple[float, float],
) -> dict:
    data = np.load(path, allow_pickle=True)
    robot_jpos = np.asarray(data["robot_jpos"], dtype=np.float32)
    object_translation = np.asarray(data["object_translation"], dtype=np.float32)
    if robot_jpos.shape[1] != len(point_names):
        raise ValueError(f"{path} has {robot_jpos.shape[1]} points, expected {len(point_names)}")
    pregrasp_step = int(data["pregrasp_step"]) if "pregrasp_step" in data.files else 0
    post_start = min(max(pregrasp_step, 0), len(robot_jpos) - 1)
    frame_indices = sorted(
        set(
            [
                0,
                min(5, len(robot_jpos) - 1),
                post_start,
                min(post_start + 5, len(robot_jpos) - 1),
                len(robot_jpos) // 2,
                len(robot_jpos) - 1,
            ]
        )
    )
    per_frame = {
        str(index): _summarize_frame(robot_jpos[index], object_translation[index], half_height, half_xy)
        for index in frame_indices
    }
    return {
        "path": str(path),
        "pregrasp_step": pregrasp_step,
        "point_names": point_names,
        "frames": per_frame,
        "whole_traj": _summarize_window(robot_jpos, object_translation, half_height, half_xy),
        "post_pregrasp_traj": _summarize_window(
            robot_jpos[post_start:],
            object_translation[post_start:],
            half_height,
            half_xy,
        ),
    }


def main() -> None:
    args = _parse_args()
    object_size = tuple(float(v) * float(args.object_scale) for v in YCB_SIZE[args.object_name])
    half_height = object_size[2] / 2.0
    half_xy = (object_size[0] / 2.0, object_size[1] / 2.0)
    summary = {
        "object": {
            "name": args.object_name,
            "scale": float(args.object_scale),
            "size_xyz": object_size,
            "half_height": half_height,
        },
        "allegro_hand_ur5": _summarize_traj(
            args.allegro_path,
            ["palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip"],
            half_height,
            half_xy,
        ),
        "rm75_inspire_right": _summarize_traj(
            args.rm75_path,
            ["palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
            half_height,
            half_xy,
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
