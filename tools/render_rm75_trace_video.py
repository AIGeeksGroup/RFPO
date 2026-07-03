#!/usr/bin/env python3
"""Render an RM75/RH56 smoke trace as a lightweight 3D MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def set_equal_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points[:, i])) for i in range(3))
    radius = max(radius, 0.18) * 0.6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def main() -> None:
    args = parse_args()
    data = np.load(args.trace)
    palm = data["palm_pos"]
    tips = data["tip_pos"]
    obj = data["object_pos"]
    target = data["target_pos"]
    lift = data["obj_lift"]
    contact = data["contact"]
    steps = data["step"]

    all_points = np.concatenate([palm[:, None], tips, obj[:, None], target[:, None]], axis=1).reshape(-1, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(out, fps=args.fps)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i in range(len(steps)):
        fig = plt.figure(figsize=(8, 6), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        set_equal_axes(ax, all_points)
        ax.view_init(elev=24, azim=-55)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(
            f"RM75/RH56 scripted smoke | step {int(steps[i])} | "
            f"lift {float(lift[i]):.3f} m | contacts {int(contact[i].sum())}"
        )

        ax.scatter(obj[i, 0], obj[i, 1], obj[i, 2], s=180, c="#111827", marker="o", label="object")
        ax.scatter(target[i, 0], target[i, 1], target[i, 2], s=120, c="#10b981", marker="*", label="target")
        ax.scatter(palm[i, 0], palm[i, 1], palm[i, 2], s=100, c="#0f172a", marker="s", label="palm")
        for finger_idx in range(tips.shape[1]):
            tip = tips[i, finger_idx]
            ax.plot([palm[i, 0], tip[0]], [palm[i, 1], tip[1]], [palm[i, 2], tip[2]], c=colors[finger_idx], lw=2)
            ax.scatter(tip[0], tip[1], tip[2], s=70, c=colors[finger_idx], marker="^")
        ax.plot(obj[: i + 1, 0], obj[: i + 1, 1], obj[: i + 1, 2], c="#111827", lw=1.5, alpha=0.7)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        plt.close(fig)
    writer.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
