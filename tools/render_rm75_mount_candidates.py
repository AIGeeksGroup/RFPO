#!/usr/bin/env python3
"""Render RM75/RH56 wrist-mount candidates in the real SAPIEN scene."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_candidate_urdf(src: Path, dst: Path, rpy: tuple[float, float, float]) -> None:
    text = src.read_text()
    replacement = f'<origin xyz="0 0 0" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}" />'
    text = re.sub(
        r'<origin xyz="0 0 0" rpy="[^"]+" />\s*</joint>\s*<link name="rh_hand_base_link">',
        replacement + "\n  </joint>\n  <link name=\"rh_hand_base_link\">",
        text,
        count=1,
    )
    dst.write_text(text)


def _render_one(urdf: Path, out_png: Path, close_val: float = 0.55, steps: int = 18) -> None:
    os.environ["VIVIDEX_RM75_URDF_OVERRIDE"] = str(urdf)
    repo = _repo_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "hand_imitation"))

    from hand_imitation.env.create_env import create_env
    from hand_imitation.env.gym_wrapper import GymWrapper

    env = create_env(
        DEFAULT_SEQ,
        use_gui=False,
        is_eval=True,
        is_vision=False,
        norm_traj=True,
        robot_name="rm75_inspire_right",
        task_kwargs={"action": "relocate", "reward_kwargs": {}},
    )
    env._stage = 2
    env.reset()
    wrapper = GymWrapper(env)
    frame = wrapper.render(mode="rgb_array")
    try:
        for _ in range(steps):
            action = np.zeros(env.action_dim, dtype=np.float32)
            if env.current_step <= env.pregrasp_steps:
                target_palm = env.cur_reference_motion["robot_pregrasp_jpos"][-1, 0]
                hand_action = 0.0
            else:
                ref_idx = min(env.current_step - env.pregrasp_steps, len(env.cur_reference_motion["robot_jpos"]) - 1)
                target_palm = env.cur_reference_motion["robot_jpos"][ref_idx, 0]
                hand_action = close_val
            palm_error = np.asarray(target_palm - env.palm_link.get_pose().p, dtype=np.float32)
            action[:3] = np.clip(palm_error / 0.04, -1.0, 1.0)
            action[3:6] = 0.0
            action[6:] = hand_action
            _, _, done, _ = env.step(action)
            frame = wrapper.render(mode="rgb_array")
            if done:
                break
    finally:
        env.close()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, frame)


def _make_montage(image_paths: list[Path], out_png: Path, cols: int = 3) -> None:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in image_paths]
    w, h = images[0].size
    label_h = 28
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (path, img) in enumerate(zip(image_paths, images)):
        x = (i % cols) * w
        y = (i // cols) * (h + label_h)
        canvas.paste(img, (x, y + label_h))
        draw.text((x + 8, y + 7), path.stem, fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main() -> None:
    repo = _repo_root()
    urdf_dir = repo / "assets/robot/rm75_inspire_right/urdf"
    src = urdf_dir / "rm75_inspire_hand_right_driven12.urdf"
    # Keep candidates beside the source URDF so all "../meshes/..." relative
    # paths resolve exactly as they do for the production robot file.
    cand_dir = urdf_dir
    cand_dir.mkdir(exist_ok=True)
    out_dir = Path("/home/why/桌面/rm75_pose_candidates_render_fixed")
    out_dir.mkdir(parents=True, exist_ok=True)

    pi = np.pi
    candidates = {
        "pitch_m90_yaw_m90": (0.0, -pi / 2, -pi / 2),
        "pitch_m90_yaw_m45": (0.0, -pi / 2, -pi / 4),
        "pitch_m90_yaw_0": (0.0, -pi / 2, 0.0),
        "pitch_m90_yaw_p45": (0.0, -pi / 2, pi / 4),
        "pitch_m90_yaw_p90": (0.0, -pi / 2, pi / 2),
        "roll_m45_pitch_m90": (-pi / 4, -pi / 2, 0.0),
        "roll_p45_pitch_m90": (pi / 4, -pi / 2, 0.0),
        "roll_m30_pitch_m90_yaw_p30": (-pi / 6, -pi / 2, pi / 6),
        "roll_p30_pitch_m90_yaw_m30": (pi / 6, -pi / 2, -pi / 6),
    }

    rendered: list[Path] = []
    for name, rpy in candidates.items():
        cand = cand_dir / f"candidate_mount_{name}.urdf"
        _write_candidate_urdf(src, cand, rpy)
        out_png = out_dir / f"{name}.png"
        print(f"[render] {name}: rpy={rpy}")
        _render_one(cand, out_png)
        rendered.append(out_png)
    _make_montage(rendered, out_dir / "montage.png")
    print(f"saved {out_dir / 'montage.png'}")


if __name__ == "__main__":
    main()
