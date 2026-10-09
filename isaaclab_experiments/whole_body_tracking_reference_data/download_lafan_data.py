# Adapted from whole_body_tracking (MIT License):
#   https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py

"""Download LAFAN1 CSV data from HuggingFace and convert to NPZ via IsaacSim FK.

Requires IsaacSim (must run on a GPU machine after setup_env.sh).

Usage:
    python whole_body_tracking_reference_data/download_lafan_data.py --headless

This downloads 7 G1 motion CSVs, runs each through the simulator to compute
body poses via forward kinematics, and saves the results as .npz files
alongside this script in whole_body_tracking_reference_data/.
"""

import argparse
import os
import tempfile
import urllib.request

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Download LAFAN1 CSV data and convert to NPZ."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_BASE_URL = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1"

MOTIONS = [
    "walk1_subject1",
    "run1_subject2",
    "dance1_subject1",
    "dance1_subject2",
    "fight1_subject2",
    "jumps1_subject1",
    "fallAndGetUp1_subject1",
]

INPUT_FPS = 30
OUTPUT_FPS = 50

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------
# Motion loading & interpolation (adapted from csv_to_npz.py)
# ---------------------------------------------------------------------------


class MotionLoader:
    def __init__(self, csv_path: str, device: torch.device):
        self.device = device
        self.input_dt = 1.0 / INPUT_FPS
        self.output_dt = 1.0 / OUTPUT_FPS
        self.current_idx = 0
        self._load_csv(csv_path)
        self._interpolate()
        self._compute_velocities()

    def _load_csv(self, csv_path: str):
        motion = (
            torch.from_numpy(np.loadtxt(csv_path, delimiter=","))
            .float()
            .to(self.device)
        )
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        # CSV stores xyzw, convert to wxyz
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate(self):
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(
            index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device)
        )
        blend = phase * (self.input_frames - 1) - index_0

        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    @staticmethod
    def _lerp(a, b, t):
        return a * (1 - t) + b * t

    @staticmethod
    def _slerp(a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        q_prev = self.motion_base_rots[:-2]
        q_next = self.motion_base_rots[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


NUM_ENVS = 1024


def convert_one(
    motion: MotionLoader,
    sim: SimulationContext,
    scene: InteractiveScene,
    joint_indexes: list[int],
) -> dict:
    """Run one motion through the sim and collect body data.

    Uses NUM_ENVS parallel environments to batch frames — each env processes
    one frame per render call, so a 500-frame motion needs just 1 render.
    """
    robot = scene["robot"]
    num_envs = robot.data.default_root_state.shape[0]
    results = {k: [] for k in (
        "joint_pos", "joint_vel",
        "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    )}

    for chunk_start in range(0, motion.output_frames, num_envs):
        chunk_end = min(chunk_start + num_envs, motion.output_frames)
        chunk_size = chunk_end - chunk_start

        # Set root states: each env gets a different frame
        root_states = robot.data.default_root_state.clone()
        root_states[:chunk_size, :3] = motion.motion_base_poss[chunk_start:chunk_end]
        root_states[:chunk_size, :2] += scene.env_origins[:chunk_size, :2]
        root_states[:chunk_size, 3:7] = motion.motion_base_rots[chunk_start:chunk_end]
        root_states[:chunk_size, 7:10] = motion.motion_base_lin_vels[chunk_start:chunk_end]
        root_states[:chunk_size, 10:] = motion.motion_base_ang_vels[chunk_start:chunk_end]
        robot.write_root_state_to_sim(root_states)

        # Set joint states
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:chunk_size, joint_indexes] = motion.motion_dof_poss[chunk_start:chunk_end]
        joint_vel[:chunk_size, joint_indexes] = motion.motion_dof_vels[chunk_start:chunk_end]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Single render + update for the whole chunk
        sim.render()
        scene.update(sim.get_physics_dt())

        # Subtract env origins from body positions (each env has a different
        # spatial offset; we want positions relative to the global origin).
        body_pos = robot.data.body_pos_w[:chunk_size].clone()
        body_pos[:, :, :3] -= scene.env_origins[:chunk_size].unsqueeze(1)

        # Collect results for the active envs only
        results["joint_pos"].append(robot.data.joint_pos[:chunk_size].cpu().numpy())
        results["joint_vel"].append(robot.data.joint_vel[:chunk_size].cpu().numpy())
        results["body_pos_w"].append(body_pos.cpu().numpy())
        results["body_quat_w"].append(robot.data.body_quat_w[:chunk_size].cpu().numpy())
        results["body_lin_vel_w"].append(robot.data.body_lin_vel_w[:chunk_size].cpu().numpy())
        results["body_ang_vel_w"].append(robot.data.body_ang_vel_w[:chunk_size].cpu().numpy())

    log = {"fps": [OUTPUT_FPS]}
    for k in results:
        log[k] = np.concatenate(results[k], axis=0)

    return log


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Check which files already exist
    to_download = []
    for name in MOTIONS:
        npz_path = os.path.join(out_dir, f"{name}.npz")
        if os.path.exists(npz_path):
            print(f"[SKIP] {name}.npz already exists")
        else:
            to_download.append(name)

    if not to_download:
        print("[INFO] All motion files already exist, nothing to do.")
        # simulation_app.close()  # Hangs when run from setup_env.sh
        return

    # Download CSVs
    csv_dir = tempfile.mkdtemp(prefix="lafan_csv_")
    for name in to_download:
        url = f"{HF_BASE_URL}/{name}.csv"
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        print(f"[DOWNLOAD] {url}")
        urllib.request.urlretrieve(url, csv_path)
        print(f"  -> {csv_path}")

    # Set up simulator
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / OUTPUT_FPS
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=NUM_ENVS, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

    # Convert each motion
    for name in to_download:
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        npz_path = os.path.join(out_dir, f"{name}.npz")

        print(f"[CONVERT] {name} ...")
        motion = MotionLoader(csv_path, device=sim.device)
        log = convert_one(motion, sim, scene, joint_indexes)
        np.savez(npz_path, **log)
        print(f"  -> {npz_path} ({motion.output_frames} frames)")

    # Clean up CSVs
    import shutil

    shutil.rmtree(csv_dir, ignore_errors=True)

    print(f"[DONE] Converted {len(to_download)} motion files.")


if __name__ == "__main__":
    main()
    # simulation_app.close()  # Hangs when run from setup_env.sh
