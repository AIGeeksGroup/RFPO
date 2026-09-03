# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 training variants for sim2real ablations.

Both sets use the same sim2real domain randomization and the real-Go2 Sport
BalanceStand pose as ``init_state.joint_pos`` / default (captured 2026-08-29).

Set A (DR): 48-D policy obs (still includes GT lin vel).
Set B (NoLinVel): 45-D actor (no GT lin vel); critic keeps privileged lin vel.
Set C (EstLinVel): 48-D actor with kinematic lin vel; critic keeps GT lin vel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.types import ArticulationActions

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from isaaclab_fpo.go2_leg_odom_obs import estimated_base_lin_vel

GO2_DR_TASK = "Isaac-Velocity-Flat-Unitree-Go2-DR-v0"
GO2_DR_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-DR-Play-v0"
GO2_NOLINVEL_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-v0"
GO2_NOLINVEL_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Play-v0"
GO2_ESTLINVEL_TASK = "Isaac-Velocity-Flat-Unitree-Go2-EstLinVel-v0"
GO2_ESTLINVEL_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-EstLinVel-Play-v0"
GO2_DR_SQUAT_TASK = "Isaac-Velocity-Flat-Unitree-Go2-DR-Squat-v0"
GO2_DR_SQUAT_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-DR-Squat-Play-v0"
GO2_NOLINVEL_SQUAT_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Squat-v0"
GO2_NOLINVEL_SQUAT_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Squat-Play-v0"
GO2_NOLINVEL_OFFICIAL_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-v0"
GO2_NOLINVEL_OFFICIAL_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-Play-v0"
GO2_NOLINVEL_OFFICIAL_PDGAP_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PdGap-v0"
GO2_NOLINVEL_OFFICIAL_PDGAP_PLAY_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PdGap-Play-v0"
GO2_NOLINVEL_OFFICIAL_PLANTDR_TASK = "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-v0"
GO2_NOLINVEL_OFFICIAL_PLANTDR_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV5_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv5-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV5_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv5-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV6_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv6-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV6_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv6-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_SOFTSTART_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-SoftStart-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_SOFTSTART_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-SoftStart-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Curr-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Curr-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_SLOW_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-CurrSlow-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_SLOW_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-CurrSlow-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE2_HOLD_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Stage2Hold-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE2_HOLD_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Stage2Hold-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE1P5_HOLD_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Stage1p5Hold-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE1P5_HOLD_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-Stage1p5Hold-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_ANTICROUCH_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-AntiCrouch-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV7_ANTICROUCH_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv7-AntiCrouch-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-RealCal-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-RealCal-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_HOLD_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-RealCal-Hold-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_HOLD_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-RealCal-Hold-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_CMD03_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-RealCal-Cmd03-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_CMD03_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDR-RealCal-Cmd03-Play-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV6_CMD03_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv6-Cmd03-v0"
)
GO2_NOLINVEL_OFFICIAL_PLANTDRV6_CMD03_PLAY_TASK = (
    "Isaac-Velocity-Flat-Unitree-Go2-NoLinVel-Official-PlantDRv6-Cmd03-Play-v0"
)

class DelayedJointPositionAction(JointPositionAction):
    """Hold q_des for 0–N control steps before sending it to the PD.

    ``last_action`` still reads the latest policy output (deploy-aligned).
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._min_delay = int(cfg.delay_min_steps)
        self._max_delay = int(cfg.delay_max_steps)
        if self._min_delay < 0 or self._max_delay < self._min_delay:
            raise ValueError(
                f"Invalid action delay range [{self._min_delay}, {self._max_delay}]"
            )
        buf = self._max_delay + 1
        self._hist = torch.zeros(self.num_envs, buf, self.action_dim, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ptr = 0
        self.reset(slice(None))

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self._max_delay <= 0:
            return
        self._ptr = (self._ptr + 1) % self._hist.shape[1]
        self._hist[:, self._ptr] = self._processed_actions

    def apply_actions(self):
        if self._max_delay <= 0:
            super().apply_actions()
            return
        idx = (self._ptr - self._delay) % self._hist.shape[1]
        delayed = self._hist[torch.arange(self.num_envs, device=self.device), idx]
        self._asset.set_joint_position_target(delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        default_q = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._hist[env_ids] = default_q[env_ids].unsqueeze(1)
        if self._max_delay <= 0:
            return
        if isinstance(env_ids, slice):
            self._delay.random_(self._min_delay, self._max_delay + 1)
        else:
            n = env_ids.numel() if torch.is_tensor(env_ids) else len(env_ids)
            self._delay[env_ids] = torch.randint(
                self._min_delay,
                self._max_delay + 1,
                (n,),
                device=self.device,
                dtype=torch.long,
            )


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = DelayedJointPositionAction
    delay_min_steps: int = 1
    delay_max_steps: int = 2


class PlantGapJointPositionAction(JointPositionAction):
    """Deploy-style plant: slew |Δq_des|, first-order lag, then optional delay to PD.

    ``action_manager.action`` / last_action still see the latest policy output.
    Only the position target sent to the actuator is lagged.

    When ``activity_tau_gain > 0``, the lag time-constant grows with multi-joint
    commanded step size (mean |Δq|), approximating soft-actuator coupling loss.

    ``delay_capacity_steps`` sizes the ring buffer so curriculum can raise
    ``delay_max_steps`` without reallocating (RealCal transport is 4–6 ticks;
    0.15–0.20s xcorr is delay+lag combined).
    """

    cfg: "PlantGapJointPositionActionCfg"

    def __init__(self, cfg: "PlantGapJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._min_delay = int(cfg.delay_min_steps)
        self._max_delay = int(cfg.delay_max_steps)
        if self._min_delay < 0 or self._max_delay < self._min_delay:
            raise ValueError(f"Invalid action delay range [{self._min_delay}, {self._max_delay}]")
        self._max_step = float(cfg.max_step_rad)
        self._tau_s = float(cfg.tau_s)
        self._activity_tau_gain = float(getattr(cfg, "activity_tau_gain", 0.0))
        self._tau_s_max = float(getattr(cfg, "tau_s_max", 0.12))
        # Preallocate for curriculum; capacity >= current max delay.
        cap = int(getattr(cfg, "delay_capacity_steps", self._max_delay))
        self._delay_capacity = max(cap, self._max_delay, 0)
        buf = self._delay_capacity + 1
        self._hist = torch.zeros(self.num_envs, buf, self.action_dim, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._q_pub = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._q_cmd = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._activity = torch.zeros(self.num_envs, device=self.device)
        self._ptr = 0
        self.reset(slice(None))

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        q_des = self._processed_actions
        if self._max_step > 0.0:
            dq = torch.clamp(q_des - self._q_pub, -self._max_step, self._max_step)
            self._activity = dq.abs().mean(dim=-1)
            self._q_pub = self._q_pub + dq
        else:
            self._activity = (q_des - self._q_pub).abs().mean(dim=-1)
            self._q_pub = q_des
        # Always fill ring if capacity > 0 so delay can be raised mid-training.
        if self._delay_capacity <= 0:
            return
        self._ptr = (self._ptr + 1) % self._hist.shape[1]
        self._hist[:, self._ptr] = self._q_pub

    def apply_actions(self):
        if self._max_delay > 0:
            d = torch.clamp(self._delay, min=0, max=self._delay_capacity)
            idx = (self._ptr - d) % self._hist.shape[1]
            target = self._hist[torch.arange(self.num_envs, device=self.device), idx]
        else:
            target = self._q_pub
        tau = self._tau_s
        if self._activity_tau_gain > 0.0 or tau > 1e-6:
            dt = float(self._env.step_dt)
            if self._activity_tau_gain > 0.0:
                tau_eff = tau + self._activity_tau_gain * self._activity
                tau_eff = torch.clamp(tau_eff, min=1e-4, max=self._tau_s_max)
                alpha = 1.0 - torch.exp(-dt / tau_eff)
                self._q_cmd = self._q_cmd + alpha.unsqueeze(-1) * (target - self._q_cmd)
            else:
                alpha = 1.0 - math.exp(-dt / tau)
                self._q_cmd = self._q_cmd + alpha * (target - self._q_cmd)
        else:
            self._q_cmd = target
        self._asset.set_joint_position_target(self._q_cmd, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        default_q = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._hist[env_ids] = default_q[env_ids].unsqueeze(1)
        self._q_pub[env_ids] = default_q[env_ids]
        self._q_cmd[env_ids] = default_q[env_ids]
        self._activity[env_ids] = 0.0
        if self._max_delay <= 0:
            self._delay[env_ids] = 0
            return
        if isinstance(env_ids, slice):
            self._delay.random_(self._min_delay, self._max_delay + 1)
        else:
            n = env_ids.numel() if torch.is_tensor(env_ids) else len(env_ids)
            self._delay[env_ids] = torch.randint(
                self._min_delay,
                self._max_delay + 1,
                (n,),
                device=self.device,
                dtype=torch.long,
            )


@configclass
class PlantGapJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = PlantGapJointPositionAction
    delay_min_steps: int = 1
    delay_max_steps: int = 3
    # Ring-buffer length floor; set >= final curriculum delay (realcal uses 10).
    delay_capacity_steps: int = 0
    # Per control step |Δq_des| cap (rad). Real LowCmd bandwidth ≪ 0.1 rad/tick.
    max_step_rad: float = 0.08
    # First-order lag time-constant on the PD target (seconds). ~40 ms default.
    tau_s: float = 0.04
    # Extra lag (s) per rad of mean |Δq| across joints. 0 = disabled (v4).
    activity_tau_gain: float = 0.0
    # Cap on effective tau when activity_tau_gain > 0.
    tau_s_max: float = 0.12


class LaggedJointPosRel(ManagerTermBase):
    """Policy joint_pos_rel from delayed / noisy measurements (not perfect execution).

    Critic should keep clean ``mdp.joint_pos_rel``. last_action stays the fresh policy out.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset_name = "robot"
        self._max_delay = int(cfg.params.get("max_delay", 2))
        n = env.scene[self._asset_name].num_joints
        buf = self._max_delay + 1
        self._hist = torch.zeros(self.num_envs, buf, n, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ptr = 0
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        asset = self._env.scene[self._asset_name]
        q_rel = asset.data.joint_pos - asset.data.default_joint_pos
        self._hist[env_ids] = q_rel[env_ids].unsqueeze(1)
        if self._max_delay <= 0:
            return
        if isinstance(env_ids, slice):
            self._delay.random_(0, self._max_delay + 1)
        else:
            n = env_ids.numel() if torch.is_tensor(env_ids) else len(env_ids)
            self._delay[env_ids] = torch.randint(
                0, self._max_delay + 1, (n,), device=self.device, dtype=torch.long
            )

    def __call__(self, env, max_delay: int = 2, noise: float = 0.02):
        asset = env.scene[self._asset_name]
        q_rel = asset.data.joint_pos - asset.data.default_joint_pos
        if self._max_delay > 0:
            self._ptr = (self._ptr + 1) % self._hist.shape[1]
            self._hist[:, self._ptr] = q_rel
            idx = (self._ptr - self._delay) % self._hist.shape[1]
            out = self._hist[torch.arange(self.num_envs, device=self.device), idx].clone()
        else:
            out = q_rel.clone()
        if noise > 0:
            out = out + (torch.rand_like(out) * 2.0 - 1.0) * noise
        return out


class Go2HVActuator(IdealPDActuator):
    """Unitree Go2HV torque-speed envelope (T–N) plus viscous/static friction.

    Same-direction peak Y1=20.2 Nm, reverse Y2=23.4 Nm, knee X1=13.5 rad/s,
    no-load X2=30 rad/s. Matches unitree_rl_lab ``UnitreeActuatorCfg_Go2HV``.
    """

    cfg: "Go2HVActuatorCfg"

    def __init__(self, cfg: "Go2HVActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)
        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel)
            + self._friction_dynamic * joint_vel
        )
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class Go2HVActuatorCfg(IdealPDActuatorCfg):
    class_type: type = Go2HVActuator
    X1: float = 13.5
    X2: float = 30.0
    Y1: float = 20.2
    Y2: float = 23.4
    Fs: float = 0.0
    Fd: float = 0.0
    Va: float = 0.01


# Real Go2 Sport BalanceStand (启用模型), Isaac BFS order. Captured 2026-08-29.
# Becomes Articulation default → joint_pos_rel and action offset.
GO2_SPORT_STAND_ARM_JOINT_POS = {
    "FL_hip_joint": 0.0239,
    "FR_hip_joint": -0.0228,
    "RL_hip_joint": 0.1144,
    "RR_hip_joint": -0.0043,
    "FL_thigh_joint": 0.7076,
    "FR_thigh_joint": 0.6980,
    "RL_thigh_joint": 0.6721,
    "RR_thigh_joint": 0.7220,
    "FL_calf_joint": -1.4971,
    "FR_calf_joint": -1.4455,
    "RL_calf_joint": -1.4297,
    "RR_calf_joint": -1.4435,
}


def apply_sport_stand_arm_init(cfg: UnitreeGo2FlatEnvCfg) -> None:
    robot = cfg.scene.robot
    cfg.scene.robot = robot.replace(
        init_state=robot.init_state.replace(joint_pos=dict(GO2_SPORT_STAND_ARM_JOINT_POS))
    )


def apply_sim2real_training_overrides(cfg: UnitreeGo2FlatEnvCfg) -> None:
    apply_sport_stand_arm_init(cfg)
    apply_sim2real_domain_rand(cfg)


def base_height_l2(env, target_height: float = 0.32, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """L2 penalty on base height vs a standing target (applied with negative weight)."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def rear_feet_long_contact(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_contact_time: float = 0.30,
):
    """Excess rear-foot stance time while commanded to walk (use with negative weight)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(contact_time - max_contact_time, min=0.0)
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return torch.sum(excess, dim=1) * moving.float()


def _standing_mask(env, command_name: str, cmd_threshold: float = 0.1) -> torch.Tensor:
    """True only for standing commands, not slow walks.

    Official curriculum starts at ±0.1; using |cmd|<0.1 treated almost every
    env as standing and blocked gait. Prefer the command term's standing flag.
    """
    term = env.command_manager.get_term(command_name)
    standing = getattr(term, "is_standing_env", None)
    if standing is not None:
        return standing
    cmd = env.command_manager.get_command(command_name)
    return torch.norm(cmd[:, :2], dim=1) < cmd_threshold


def stand_still_action_l2(env, command_name: str = "base_velocity", cmd_threshold: float = 0.1):
    """Penalize large actions while commanded to stand (cmd≈0)."""
    mask = _standing_mask(env, command_name, cmd_threshold)
    return torch.sum(torch.square(env.action_manager.action), dim=1) * mask.float()


def stand_still_action_excess(
    env,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
    action_limit: float = 0.3,
):
    """Extra penalty when any joint exceeds |a|=0.3 while standing."""
    mask = _standing_mask(env, command_name, cmd_threshold)
    excess = torch.clamp(torch.abs(env.action_manager.action) - action_limit, min=0.0)
    return torch.sum(excess, dim=1) * mask.float()


def stand_still_joint_l2(
    env,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalize leaving the sport_stand default while commanded to stand."""
    asset = env.scene[asset_cfg.name]
    q_err = asset.data.joint_pos - asset.data.default_joint_pos
    mask = _standing_mask(env, command_name, cmd_threshold)
    return torch.sum(torch.square(q_err), dim=1) * mask.float()


def _gait_init_window(env, command_name: str, window_s: float) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    mode = getattr(term, "init_mode", None)
    age = getattr(term, "walk_age", None)
    if mode is None or age is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (mode > 0) & (age > 0) & (age < window_s)


def fail_to_walk(
    env,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.12,
    grace_s: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalize commanded walking that stays in the standing attractor."""
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = torch.norm(cmd[:, :2], dim=1)
    vel_xy = torch.norm(env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], dim=1)
    walk = cmd_xy > cmd_threshold
    term = env.command_manager.get_term(command_name)
    mode = getattr(term, "init_mode", None)
    age = getattr(term, "walk_age", None)
    if mode is not None and age is not None:
        in_init = mode > 0
        walk = walk & ((~in_init) | (age > grace_s))
    ratio = vel_xy / cmd_xy.clamp(min=1.0e-3)
    stuck = (1.0 - ratio.clamp(max=1.0)).square()
    return stuck * walk.float()


def gait_init_action_sat(
    env,
    command_name: str = "base_velocity",
    limit: float = 0.85,
    window_s: float = 0.6,
):
    """First-step clip: don't spend the initiation window at |a|≈1.2."""
    win = _gait_init_window(env, command_name, window_s)
    excess = torch.clamp(torch.abs(env.action_manager.action) - limit, min=0.0)
    return torch.sum(excess, dim=1) * win.float()


def gait_init_lr_asym(
    env,
    command_name: str = "base_velocity",
    window_s: float = 0.6,
):
    """Penalize RR vs RL first-kick (Isaac BFS calves 10/11)."""
    win = _gait_init_window(env, command_name, window_s)
    a = env.action_manager.action
    rear = torch.abs(a[:, 10] - a[:, 11])
    front = torch.abs(a[:, 8] - a[:, 9])
    hips = torch.abs(a[:, 2] - a[:, 3])
    return (rear + 0.5 * front + 0.5 * hips) * win.float()


def gait_init_roll_rate(
    env,
    command_name: str = "base_velocity",
    window_s: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalize first-second roll rate that matches the real-robot fall."""
    win = _gait_init_window(env, command_name, window_s)
    wx = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 0]
    return torch.square(wx) * win.float()


class RecoverLastAction(ManagerTermBase):
    """Policy last_action with delay / drop / noise. Critic stays on clean last_action.

    Deploy still feeds the latest network output; this is train-time mismatch DR.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        nact = env.action_manager.total_action_dim
        self._max_delay = int(cfg.params.get("max_delay", 2))
        buf = self._max_delay + 1
        self._hist = torch.zeros(self.num_envs, buf, nact, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ptr = 0
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._hist[env_ids] = 0.0
        if self._max_delay <= 0:
            return
        if isinstance(env_ids, slice):
            self._delay.random_(0, self._max_delay + 1)
        else:
            n = env_ids.numel() if torch.is_tensor(env_ids) else len(env_ids)
            self._delay[env_ids] = torch.randint(
                0, self._max_delay + 1, (n,), device=self.device, dtype=torch.long
            )

    def __call__(self, env, drop_prob: float = 0.06, noise: float = 0.30, max_delay: int = 2):
        raw = env.action_manager.action
        if self._max_delay > 0:
            self._ptr = (self._ptr + 1) % self._hist.shape[1]
            self._hist[:, self._ptr] = raw
            idx = (self._ptr - self._delay) % self._hist.shape[1]
            out = self._hist[torch.arange(self.num_envs, device=self.device), idx].clone()
        else:
            out = raw.clone()
        if noise > 0:
            out = out + (torch.rand_like(out) * 2.0 - 1.0) * noise
        if drop_prob > 0:
            drop = torch.rand(self.num_envs, 1, device=self.device) < drop_prob
            out = torch.where(drop, torch.zeros_like(out), out)
        return out


class GaitInitVelocityCommand(UniformVelocityCommand):
    """Uniform walk commands plus stand-hold then discrete/ramp vx.

    Stops the policy from living only on an already-walking distribution.
    ``init_mode``: 0 uniform, 1 jump after hold, 2 ramp after hold.
    """

    cfg: "GaitInitVelocityCommandCfg"

    def __init__(self, cfg: "GaitInitVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.init_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hold_left = torch.zeros(self.num_envs, device=self.device)
        self.walk_age = torch.zeros(self.num_envs, device=self.device)
        self.target_vx = torch.zeros(self.num_envs, device=self.device)
        self.ramp_s = torch.ones(self.num_envs, device=self.device)
        self._speeds = torch.tensor(list(cfg.discrete_vx), device=self.device, dtype=torch.float32)

    def schedule_gait_init(self, env_ids: torch.Tensor, vx: float = 0.3, hold_s: float = 0.8):
        if env_ids is None or env_ids.numel() == 0:
            return
        self.init_mode[env_ids] = 1
        self.target_vx[env_ids] = float(vx)
        self.hold_left[env_ids] = float(hold_s)
        self.walk_age[env_ids] = 0.0
        # Hold is cmd=0 before a walk, not a true stand. Do not mark standing
        # or stand_still_* will punish the walking policy during the wait.
        self.is_standing_env[env_ids] = False
        self.vel_command_b[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        self.init_mode[ids] = 0
        self.hold_left[ids] = 0.0
        self.walk_age[ids] = 0.0
        prob = float(getattr(self.cfg, "gait_init_prob", 0.0))
        if prob <= 0.0:
            return
        pick = torch.rand(ids.numel(), device=self.device) < prob
        if not pick.any():
            return
        gids = ids[pick]
        n = gids.numel()
        sp_idx = torch.randint(0, self._speeds.numel(), (n,), device=self.device)
        self.target_vx[gids] = self._speeds[sp_idx]
        self.hold_left[gids] = torch.empty(n, device=self.device).uniform_(*self.cfg.hold_time_s)
        ramp = torch.rand(n, device=self.device) < float(self.cfg.ramp_prob)
        self.init_mode[gids] = torch.where(
            ramp,
            torch.full((n,), 2, device=self.device, dtype=torch.long),
            torch.ones(n, device=self.device, dtype=torch.long),
        )
        self.ramp_s[gids] = torch.empty(n, device=self.device).uniform_(*self.cfg.ramp_time_s)
        self.is_standing_env[gids] = False
        self.is_heading_env[gids] = False
        self.vel_command_b[gids] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        delay = float(getattr(self.cfg, "heading_delay_s", 0.8))
        was_holding = self.hold_left > 0
        self.hold_left.sub_(dt).clamp_(min=0.0)
        active = (self.init_mode > 0) & (self.hold_left <= 0)
        self.walk_age = torch.where(active, self.walk_age + dt, torch.zeros_like(self.walk_age))
        just_released = was_holding & active
        if just_released.any() and self.cfg.heading_command:
            noise = (torch.rand(self.num_envs, device=self.device) * 2.0 - 1.0) * 0.8
            self.heading_target[just_released] = self.robot.data.heading_w[just_released] + noise[just_released]
        super()._update_command()
        still_hold = self.hold_left > 0
        self.vel_command_b[still_hold] = 0.0
        # Keep hold off the standing mask so stand_still_* only hits true stands.
        self.is_standing_env[still_hold] = False
        jump = (self.init_mode == 1) & (self.hold_left <= 0)
        if jump.any():
            self.vel_command_b[jump, 0] = self.target_vx[jump]
            self.vel_command_b[jump, 1] = 0.0
            self.is_standing_env[jump] = False
        ramp = (self.init_mode == 2) & (self.hold_left <= 0)
        if ramp.any():
            frac = (self.walk_age[ramp] / self.ramp_s[ramp].clamp(min=1.0e-4)).clamp(max=1.0)
            self.vel_command_b[ramp, 0] = self.target_vx[ramp] * frac
            self.vel_command_b[ramp, 1] = 0.0
            self.is_standing_env[ramp] = False
        # Straight first steps; heading error comes in after the gait exists.
        early = active & (self.walk_age <= delay)
        self.is_heading_env[early] = False
        self.vel_command_b[early, 2] = 0.0
        late = active & (self.walk_age > delay)
        self.is_heading_env[late] = True


@configclass
class GaitInitVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GaitInitVelocityCommand
    gait_init_prob: float = 0.45
    discrete_vx: tuple = (0.15, 0.20, 0.25, 0.30)
    hold_time_s: tuple[float, float] = (1.0, 2.4)
    ramp_prob: float = 0.40
    ramp_time_s: tuple[float, float] = (0.4, 1.0)
    heading_delay_s: float = 0.8


def snap_reset_to_sport_stand(
    env,
    env_ids: torch.Tensor,
    stand_prob: float = 0.40,
    roll_rad: float = 0.08,
    yaw_rate: float = 0.45,
):
    """A fraction of resets land on sport_stand with small roll / yaw-rate noise."""
    if env_ids is None or env_ids.numel() == 0:
        return
    asset = env.scene["robot"]
    stand = torch.rand(env_ids.numel(), device=asset.device) < stand_prob
    ids = env_ids[stand]
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = asset.device
    q = asset.data.default_joint_pos[ids]
    dq = torch.zeros_like(q)
    asset.write_joint_state_to_sim(q, dq, env_ids=ids)
    root_pose = asset.data.root_state_w[ids, :7].clone()
    roll = (torch.rand(n, device=device) * 2.0 - 1.0) * roll_rad
    zeros = torch.zeros(n, device=device)
    root_pose[:, 3:7] = math_utils.quat_mul(
        root_pose[:, 3:7], math_utils.quat_from_euler_xyz(roll, zeros, zeros)
    )
    root_vel = torch.zeros(n, 6, device=device)
    root_vel[:, 3] = (torch.rand(n, device=device) * 2.0 - 1.0) * 0.25
    root_vel[:, 5] = (torch.rand(n, device=device) * 2.0 - 1.0) * yaw_rate
    asset.write_root_pose_to_sim(root_pose, env_ids=ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=ids)
    env.action_manager.action[ids] = 0.0
    env.action_manager.prev_action[ids] = 0.0


def perturb_gait_handoff(
    env,
    env_ids: torch.Tensor,
    last_action_scale: float = 0.30,
    joint_pos_rad: float = 0.18,
    stand_snap_prob: float = 0.20,
):
    """Mid-episode handoff: noisy last_action, offset q, sometimes snap to stand."""
    if env_ids is None or env_ids.numel() == 0:
        return
    asset = env.scene["robot"]
    n = env_ids.numel()
    device = asset.device
    act = env.action_manager.action
    act[env_ids] = act[env_ids] + (torch.rand(n, act.shape[1], device=device) * 2.0 - 1.0) * last_action_scale
    q = asset.data.joint_pos[env_ids].clone()
    q = q + (torch.rand_like(q) * 2.0 - 1.0) * joint_pos_rad
    snap = torch.rand(n, device=device) < stand_snap_prob
    if snap.any():
        q[snap] = asset.data.default_joint_pos[env_ids][snap]
        act[env_ids[snap]] = 0.0
        cmd = env.command_manager.get_term("base_velocity")
        if hasattr(cmd, "schedule_gait_init"):
            cmd.schedule_gait_init(env_ids[snap], vx=0.3, hold_s=0.6)
    asset.write_joint_state_to_sim(q, asset.data.joint_vel[env_ids], env_ids=env_ids)


def apply_nolinvel_stance_rewards(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Penalize a crouched shuffle: low base, planted rear feet, thigh/calf scrape."""
    if getattr(cfg.rewards, "feet_air_time", None) is not None:
        cfg.rewards.feet_air_time.weight = 0.4
    cfg.rewards.base_height_l2 = RewTerm(
        func=base_height_l2,
        weight=-12.0,
        params={"target_height": 0.32},
    )
    cfg.rewards.rear_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    cfg.rewards.rear_feet_long_contact = RewTerm(
        func=rear_feet_long_contact,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
            "command_name": "base_velocity",
            "max_contact_time": 0.30,
        },
    )
    cfg.rewards.rear_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["RL_foot", "RR_foot"]),
        },
    )
    cfg.rewards.rear_leg_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["RL_thigh", "RR_thigh", "RL_calf", "RR_calf"],
            ),
            "threshold": 1.0,
        },
    )
    cfg.rewards.stand_still_action_l2 = RewTerm(
        func=stand_still_action_l2,
        weight=-1.5,
        params={"command_name": "base_velocity", "cmd_threshold": 0.1},
    )
    cfg.rewards.stand_still_action_excess = RewTerm(
        func=stand_still_action_excess,
        weight=-8.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.1, "action_limit": 0.3},
    )
    cfg.rewards.stand_still_joint_l2 = RewTerm(
        func=stand_still_joint_l2,
        weight=-1.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.1},
    )


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize |qvel| * |applied torque| (Unitree official energy term)."""
    asset = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def air_time_variance_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize leftover variance of last air / contact times across feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def lin_vel_cmd_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    limit_x: tuple[float, float] = (-1.0, 1.0),
    limit_y: tuple[float, float] = (-0.4, 0.4),
) -> torch.Tensor:
    """Expand vx/vy command ranges when tracking reward is high enough."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta,
                limit_x[0],
                limit_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta,
                limit_y[0],
                limit_y[1],
            ).tolist()
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def apply_plant_dr_v4(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Deploy-aligned plant DR on Official squat — plant path only.

    Matches deploy: action_scale=0.25, max_step_rad=0.20, kp∈[15,35], delay 0–2.
    Does NOT touch obs, rewards, or cmd/action_scale curriculum (unlike PdGap v3).
    """
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=0,
        delay_max_steps=2,
        max_step_rad=0.20,
        tau_s=0.03,
    )
    kp_scale = (0.6, 1.4)  # base kp=25 → [15, 35]
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = kp_scale
        cfg.events.actuator_gains.params["damping_distribution_params"] = kp_scale
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": kp_scale,
                "damping_distribution_params": kp_scale,
                "operation": "scale",
                "distribution": "uniform",
            },
        )


def apply_plant_dr_v5(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Soft-actuator plant DR: low kp bias + multi-joint activity lag.

    Aligns IdealPD toward real soft tracking without tightening T–N limits.
    Obs / rewards / cmd curriculum stay Official.
    """
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=0,
        delay_max_steps=2,
        max_step_rad=0.20,
        tau_s=0.03,
        activity_tau_gain=0.25,  # mean|Δq|~0.1 → +25 ms lag
        tau_s_max=0.12,
    )
    # base kp=25 → [7.5, 25]; never harder than Official deploy
    kp_scale = (0.3, 1.0)
    gain_params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "stiffness_distribution_params": kp_scale,
        "damping_distribution_params": kp_scale,
        "operation": "scale",
        "distribution": "uniform",
    }
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.mode = "reset"
        cfg.events.actuator_gains.params.update(gain_params)
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params=gain_params,
        )


def apply_plant_dr_v6(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """v4 delay/slew + v5 soft kp/activity lag; drop long_contact conflict.

    Soft plant slows rear-leg lift → rear_feet_long_contact (−2) explodes and
    collapses gait (v5 FT fingerprint). Disable that term only; keep other rewards.
    """
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=0,
        delay_max_steps=2,  # v4 delay
        max_step_rad=0.20,  # deploy-aligned slew
        tau_s=0.03,
        activity_tau_gain=0.20,  # slightly milder than v5's 0.25
        tau_s_max=0.10,
    )
    kp_scale = (0.3, 1.0)  # [7.5, 25]
    gain_params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "stiffness_distribution_params": kp_scale,
        "damping_distribution_params": kp_scale,
        "operation": "scale",
        "distribution": "uniform",
    }
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.mode = "reset"
        cfg.events.actuator_gains.params.update(gain_params)
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params=gain_params,
        )
    if hasattr(cfg.rewards, "rear_feet_long_contact"):
        cfg.rewards.rear_feet_long_contact = None


def apply_plant_dr_v7(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Mild soft-plant on top of v4: light activity lag + mild low-kp bias.

    Intended for short FT from a healthy v4 ckpt. Softer than v4, much milder
    than v6 (keep rear_feet_long_contact; kp floor ~12.5 not 7.5).
    """
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=0,
        delay_max_steps=2,
        max_step_rad=0.20,
        tau_s=0.03,
        activity_tau_gain=0.10,  # half of v6
        tau_s_max=0.07,
    )
    # base kp=25 → [12.5, 26.25]; slight soft bias, still near deploy
    kp_scale = (0.5, 1.05)
    gain_params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "stiffness_distribution_params": kp_scale,
        "damping_distribution_params": kp_scale,
        "operation": "scale",
        "distribution": "uniform",
    }
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.mode = "reset"
        cfg.events.actuator_gains.params.update(gain_params)
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params=gain_params,
        )
    # Keep long_contact; only soften weight so mild lag does not dominate.
    if getattr(cfg.rewards, "rear_feet_long_contact", None) is not None:
        cfg.rewards.rear_feet_long_contact.weight = -1.0


def apply_plant_dr_v7_softstart(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """v7 mild soft-plant + gait-init soft-start (hold → jump/ramp).

    Targets the deploy fingerprint: tip / collapse right after leaving squat.
    Reuses ``apply_nolinvel_tracking_sim2real``; bumps gait-init rate and keeps
    discrete vx around the real stick ≈0.3 band.
    """
    apply_plant_dr_v7(cfg)
    apply_nolinvel_tracking_sim2real(cfg)
    # More stand→walk practice than the generic nolinvel FT pass.
    if hasattr(cfg.commands.base_velocity, "gait_init_prob"):
        cfg.commands.base_velocity.gait_init_prob = 0.40
        cfg.commands.base_velocity.discrete_vx = (0.15, 0.20, 0.25, 0.30, 0.35)
        cfg.commands.base_velocity.ramp_prob = 0.50
        cfg.commands.base_velocity.hold_time_s = (0.8, 2.0)
        cfg.commands.base_velocity.ramp_time_s = (0.5, 1.2)


# Real plant card (standing squat load, 2026-08-31). Source of truth for RealCal.
# kp/kd deploy=25/0.5; effort_limit 23.5 NOT binding → do not DR torque walls.
# actuator_delay_s 0.15–0.20 is TOTAL group delay from sine xcorr (dead-time + lag),
# not extra transport on top of τ. start-of-motion dead-time ~0.015s.
# bandwidth_fl_thigh ~1Hz; |H|≈0.78 @1Hz → τ≈0.13s; phase_lag_1hz~70°.
# Split: transport 4–6 ticks (0.08–0.12s) + FO lag 0.13s → phase ~68–82° (card 70°),
# 1Hz group delay ~0.16–0.20s (card 0.15–0.20s). Do NOT also put 8–10 tick delay.
# ess_single~0.04; inactive_hold~0.25; coupling_R_all_thigh~2.4; coupling_R_all12~5.5.
_REAL_PLANT_DT_S = 0.02
_REAL_PLANT_DELAY_STEPS = (4, 6)  # transport only; 0.08–0.12s @ 50 Hz
# |H|≈0.78 @1Hz → τ≈0.13s. Activity coupling adds lag, not another long delay.
_REAL_PLANT_TAU_S = 0.13
_REAL_PLANT_TAU_S_MAX = 0.22
_REAL_PLANT_ACTIVITY_TAU_GAIN = 0.50  # mean|Δq|~0.1 → +50ms; clamps at tau_s_max
_REAL_PLANT_KP_SCALE = (0.50, 1.00)  # base 25 → [12.5, 25]

# Curriculum → RealCal end state (delay+lag share the xcorr budget; no double count).
_PLANT_HARDNESS_STAGES: tuple[dict, ...] = (
    # 0: near IdealPD
    {
        "kp_scale": (0.95, 1.05),
        "activity_tau_gain": 0.0,
        "tau_s": 0.0,
        "tau_s_max": 0.05,
        "max_step_rad": 0.30,
        "delay_min_steps": 0,
        "delay_max_steps": 0,
    },
    # 1: start-of-motion delay + light lag
    {
        "kp_scale": (0.70, 1.10),
        "activity_tau_gain": 0.20,
        "tau_s": 0.04,
        "tau_s_max": 0.12,
        "max_step_rad": 0.25,
        "delay_min_steps": 0,
        "delay_max_steps": 2,
    },
    # 2: mid transport + mid lag
    {
        "kp_scale": (0.55, 1.05),
        "activity_tau_gain": 0.35,
        "tau_s": 0.08,
        "tau_s_max": 0.18,
        "max_step_rad": 0.20,
        "delay_min_steps": 2,
        "delay_max_steps": 4,
    },
    # 3: RealCal — |H| and 1Hz phase matched; xcorr delay is delay+lag combined
    {
        "kp_scale": _REAL_PLANT_KP_SCALE,
        "activity_tau_gain": _REAL_PLANT_ACTIVITY_TAU_GAIN,
        "tau_s": _REAL_PLANT_TAU_S,
        "tau_s_max": _REAL_PLANT_TAU_S_MAX,
        "max_step_rad": 0.18,
        "delay_min_steps": _REAL_PLANT_DELAY_STEPS[0],
        "delay_max_steps": _REAL_PLANT_DELAY_STEPS[1],
    },
)

# Between stage 1 (delay 0–2, τ=0.04) and stage 2 (delay 2–4, τ=0.08).
_PLANT_MID15_STAGE: dict = {
    "kp_scale": (0.62, 1.08),
    "activity_tau_gain": 0.28,
    "tau_s": 0.06,
    "tau_s_max": 0.15,
    "max_step_rad": 0.22,
    "delay_min_steps": 1,
    "delay_max_steps": 3,
}

# Scratch curriculum: delay present from iter 0 (no IdealPD honeymoon), 7 rungs to RealCal.
_PLANT_ANTICROUCH_STAGES: tuple[dict, ...] = (
    {
        "kp_scale": (0.80, 1.10),
        "activity_tau_gain": 0.10,
        "tau_s": 0.02,
        "tau_s_max": 0.08,
        "max_step_rad": 0.28,
        "delay_min_steps": 0,
        "delay_max_steps": 3,
    },
    {
        "kp_scale": (0.72, 1.08),
        "activity_tau_gain": 0.18,
        "tau_s": 0.04,
        "tau_s_max": 0.12,
        "max_step_rad": 0.26,
        "delay_min_steps": 1,
        "delay_max_steps": 3,
    },
    {
        "kp_scale": (0.68, 1.06),
        "activity_tau_gain": 0.24,
        "tau_s": 0.06,
        "tau_s_max": 0.14,
        "max_step_rad": 0.24,
        "delay_min_steps": 1,
        "delay_max_steps": 4,
    },
    {
        "kp_scale": (0.62, 1.05),
        "activity_tau_gain": 0.30,
        "tau_s": 0.08,
        "tau_s_max": 0.16,
        "max_step_rad": 0.22,
        "delay_min_steps": 2,
        "delay_max_steps": 4,
    },
    {
        "kp_scale": (0.58, 1.04),
        "activity_tau_gain": 0.38,
        "tau_s": 0.10,
        "tau_s_max": 0.18,
        "max_step_rad": 0.20,
        "delay_min_steps": 2,
        "delay_max_steps": 5,
    },
    {
        "kp_scale": (0.54, 1.02),
        "activity_tau_gain": 0.44,
        "tau_s": 0.12,
        "tau_s_max": 0.20,
        "max_step_rad": 0.19,
        "delay_min_steps": 3,
        "delay_max_steps": 6,
    },
    {
        "kp_scale": _REAL_PLANT_KP_SCALE,
        "activity_tau_gain": _REAL_PLANT_ACTIVITY_TAU_GAIN,
        "tau_s": _REAL_PLANT_TAU_S,
        "tau_s_max": _REAL_PLANT_TAU_S_MAX,
        "max_step_rad": 0.18,
        "delay_min_steps": _REAL_PLANT_DELAY_STEPS[0],
        "delay_max_steps": _REAL_PLANT_DELAY_STEPS[1],
    },
)


def _plant_stage_table(stages_name: str) -> tuple[dict, ...]:
    if stages_name == "anticouch":
        return _PLANT_ANTICROUCH_STAGES
    return _PLANT_HARDNESS_STAGES


def _apply_plant_hardness_stage(env, stage: dict) -> None:
    """Push plant action + actuator-gain DR to a curriculum stage."""
    term = env.action_manager.get_term("joint_pos")
    kp_scale = tuple(stage["kp_scale"])
    act_gain = float(stage["activity_tau_gain"])
    tau_s = float(stage["tau_s"])
    tau_max = float(stage["tau_s_max"])
    max_step = float(stage["max_step_rad"])
    dmin = int(stage.get("delay_min_steps", 0))
    dmax = int(stage.get("delay_max_steps", 0))
    if hasattr(term, "_activity_tau_gain"):
        term._activity_tau_gain = act_gain
        term._tau_s = tau_s
        term._tau_s_max = tau_max
        term._max_step = max_step
        term._min_delay = dmin
        term._max_delay = dmax
    if hasattr(term, "cfg"):
        term.cfg.activity_tau_gain = act_gain
        term.cfg.tau_s = tau_s
        term.cfg.tau_s_max = tau_max
        term.cfg.max_step_rad = max_step
        term.cfg.delay_min_steps = dmin
        term.cfg.delay_max_steps = dmax
    gains = getattr(env.cfg.events, "actuator_gains", None)
    if gains is not None and getattr(gains, "params", None) is not None:
        gains.params["stiffness_distribution_params"] = kp_scale
        gains.params["damping_distribution_params"] = kp_scale


def _set_cfg_plant_stage(cfg: UnitreeGo2FlatEnvCfg, stage: dict) -> None:
    """Write a hardness stage into the env cfg (before the env is constructed)."""
    act = cfg.actions.joint_pos
    act.delay_min_steps = int(stage["delay_min_steps"])
    act.delay_max_steps = int(stage["delay_max_steps"])
    act.max_step_rad = float(stage["max_step_rad"])
    act.tau_s = float(stage["tau_s"])
    act.activity_tau_gain = float(stage["activity_tau_gain"])
    act.tau_s_max = float(stage["tau_s_max"])
    kp_scale = tuple(stage["kp_scale"])
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = kp_scale
        cfg.events.actuator_gains.params["damping_distribution_params"] = kp_scale


def plant_hardness_levels(
    env,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    threshold_frac: float = 0.85,
    min_boundaries_at_level: int = 12,
    start_stage: int = 0,
    stages_name: str = "v7",
) -> torch.Tensor:
    """Curriculum: gradually harden plant DR (delay + soft multi-joint) when tracking OK.

    Do not key off ``common_step_counter % episode_length``. Resume restores that
    counter to ``iter * steps * num_envs``, so the remainder never hits 0 at
    reset and hardness stays stuck at stage 0.
    """
    stages = _plant_stage_table(stages_name)
    if not hasattr(env, "_plant_hardness_stage"):
        start_i = max(0, min(int(start_stage), len(stages) - 1))
        env._plant_hardness_stage = start_i
        env._plant_hardness_boundaries = 0
        env._plant_hardness_origin = int(env.common_step_counter)
        env._plant_hardness_milestone = -1
        _apply_plant_hardness_stage(env, stages[start_i])
    elapsed = int(env.common_step_counter) - int(env._plant_hardness_origin)
    ep_len = max(int(env.max_episode_length), 1)
    milestone = elapsed // ep_len
    if milestone == int(env._plant_hardness_milestone):
        return torch.tensor(float(env._plant_hardness_stage), device=env.device)
    env._plant_hardness_milestone = milestone
    env._plant_hardness_boundaries += 1
    stage_i = int(env._plant_hardness_stage)
    if stage_i >= len(stages) - 1:
        return torch.tensor(float(stage_i), device=env.device)
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )
    if (
        env._plant_hardness_boundaries >= int(min_boundaries_at_level)
        and reward > reward_term.weight * float(threshold_frac)
    ):
        stage_i += 1
        env._plant_hardness_stage = stage_i
        env._plant_hardness_boundaries = 0
        _apply_plant_hardness_stage(env, stages[stage_i])
    return torch.tensor(float(stage_i), device=env.device)


def apply_plant_dr_v7_curr(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Scratch plant curriculum ending at RealCal (2026-08-31 card).

    Starts near IdealPD; hardens toward delay 4–6 + τ=0.13 (matched 1Hz card).
    No torque-limit DR.
    """
    stage0 = _PLANT_HARDNESS_STAGES[0]
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=int(stage0["delay_min_steps"]),
        delay_max_steps=int(stage0["delay_max_steps"]),
        delay_capacity_steps=_REAL_PLANT_DELAY_STEPS[1],
        max_step_rad=float(stage0["max_step_rad"]),
        tau_s=float(stage0["tau_s"]),
        activity_tau_gain=float(stage0["activity_tau_gain"]),
        tau_s_max=float(stage0["tau_s_max"]),
    )
    kp_scale = tuple(stage0["kp_scale"])
    gain_params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "stiffness_distribution_params": kp_scale,
        "damping_distribution_params": kp_scale,
        "operation": "scale",
        "distribution": "uniform",
    }
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.mode = "reset"
        cfg.events.actuator_gains.params.update(gain_params)
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params=gain_params,
        )
    if getattr(cfg.rewards, "rear_feet_long_contact", None) is not None:
        cfg.rewards.rear_feet_long_contact.weight = -1.0
    cfg.curriculum.plant_hardness_levels = CurrTerm(
        func=plant_hardness_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "threshold_frac": 0.85,
            "min_boundaries_at_level": 12,
        },
    )


def apply_plant_dr_v7_curr_anticouch(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Scratch: delay from iter 0, 7 rungs to RealCal, expensive crouch.

    Does not start on IdealPD. Height penalty is ~7× the Official default so
    sitting-to-survive under delay is not cheaper than tracking.
    """
    apply_plant_dr_v7_curr(cfg)
    stage0 = _PLANT_ANTICROUCH_STAGES[0]
    _set_cfg_plant_stage(cfg, stage0)
    cfg.curriculum.plant_hardness_levels.params["stages_name"] = "anticouch"
    cfg.curriculum.plant_hardness_levels.params["min_boundaries_at_level"] = 16
    cfg.curriculum.plant_hardness_levels.params["threshold_frac"] = 0.85
    if getattr(cfg.rewards, "base_height_l2", None) is not None:
        cfg.rewards.base_height_l2.weight = -80.0
    if getattr(cfg.rewards, "rear_feet_long_contact", None) is not None:
        cfg.rewards.rear_feet_long_contact.weight = -2.0
    cfg.rewards.fail_to_walk = RewTerm(
        func=fail_to_walk,
        weight=-0.5,
        params={"command_name": "base_velocity", "cmd_threshold": 0.12, "grace_s": 0.45},
    )


def apply_plant_dr_v7_curr_slow(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """FT from a finished Official walker: start plant stage 1, climb to RealCal.

    Cmd frozen at vx±1 / vy±0.4. min_boundaries=6 so stages move every ~250 iters
    instead of idling on IdealPD.
    """
    apply_plant_dr_v7_curr(cfg)
    apply_cmd_full_freeze(cfg)
    start = 1
    cfg.curriculum.plant_hardness_levels.params["min_boundaries_at_level"] = 6
    cfg.curriculum.plant_hardness_levels.params["threshold_frac"] = 0.85
    cfg.curriculum.plant_hardness_levels.params["start_stage"] = start
    _set_cfg_plant_stage(cfg, _PLANT_HARDNESS_STAGES[start])


def apply_plant_dr_stage2_hold(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Pin plant at curriculum stage 2 (delay 2–4 + τ=0.08). No hardness climb."""
    apply_plant_dr_v7_curr(cfg)
    apply_cmd_full_freeze(cfg)
    cfg.curriculum.plant_hardness_levels = None
    _set_cfg_plant_stage(cfg, _PLANT_HARDNESS_STAGES[2])


def apply_plant_dr_stage1p5_hold(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Pin plant at delay 1–3 + τ=0.06. One rung above stage 1. No hardness climb."""
    apply_plant_dr_v7_curr(cfg)
    apply_cmd_full_freeze(cfg)
    cfg.curriculum.plant_hardness_levels = None
    _set_cfg_plant_stage(cfg, _PLANT_MID15_STAGE)


def apply_plant_dr_realcal(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Fixed plant matching 2026-08-31 real card (for short FT onto Curr/v4).

    delay 4–6 ticks + τ≈0.13s share the 0.15–0.20s xcorr budget (not stacked
    8–10 tick delay). Activity coupling + kp×[0.50,1.0]. No effort_limit DR.
    """
    end = _PLANT_HARDNESS_STAGES[-1]
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=int(end["delay_min_steps"]),
        delay_max_steps=int(end["delay_max_steps"]),
        delay_capacity_steps=_REAL_PLANT_DELAY_STEPS[1],
        max_step_rad=float(end["max_step_rad"]),
        tau_s=float(end["tau_s"]),
        activity_tau_gain=float(end["activity_tau_gain"]),
        tau_s_max=float(end["tau_s_max"]),
    )
    kp_scale = tuple(end["kp_scale"])
    gain_params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "stiffness_distribution_params": kp_scale,
        "damping_distribution_params": kp_scale,
        "operation": "scale",
        "distribution": "uniform",
    }
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.mode = "reset"
        cfg.events.actuator_gains.params.update(gain_params)
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params=gain_params,
        )
    if getattr(cfg.rewards, "rear_feet_long_contact", None) is not None:
        cfg.rewards.rear_feet_long_contact.weight = -1.0


def apply_cmd03_focus(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Freeze velocity curriculum and concentrate commands around vx≈0.3."""
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
    if hasattr(cfg.commands.base_velocity.ranges, "ang_vel_z"):
        cfg.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
    cfg.curriculum.lin_vel_cmd_levels = None
    # Fewer pure-standing episodes so most rollouts practice ~0.3 walk.
    if hasattr(cfg.commands.base_velocity, "rel_standing_envs"):
        cfg.commands.base_velocity.rel_standing_envs = 0.05


def apply_cmd_full_freeze(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Pin vx/vy at the finished curriculum so resume does not reset to ±0.1."""
    cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
    cfg.curriculum.lin_vel_cmd_levels = None


def strip_plant_dr_for_play(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Play/eval: clean PD target path (no slew/delay/lag)."""
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = JointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.25,
        use_default_offset=getattr(old, "use_default_offset", True),
    )


def apply_pd_tracking_gap(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Plant-gap FT on Official (v3): lag plant, milder penalties, slow curriculum.

    v2 late-collapse fingerprint was rear_feet_long_contact exploding under lag;
    that term is disabled here. action_rate stays near Official; scale/cmd climb slowly.
    """
    old = cfg.actions.joint_pos
    cfg.actions.joint_pos = PlantGapJointPositionActionCfg(
        asset_name=old.asset_name,
        joint_names=old.joint_names,
        scale=0.15,  # curriculum bumps toward 0.25
        use_default_offset=getattr(old, "use_default_offset", True),
        delay_min_steps=1,
        delay_max_steps=3,
        max_step_rad=0.08,
        tau_s=0.04,
    )
    if getattr(cfg.events, "actuator_gains", None) is not None:
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = (0.35, 1.2)
        cfg.events.actuator_gains.params["damping_distribution_params"] = (0.5, 1.2)
    else:
        cfg.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.35, 1.2),
                "damping_distribution_params": (0.5, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    # Keep last_action = latest policy output (deploy-aligned). Corrupt joint_pos instead.
    if getattr(cfg.observations, "policy", None) is not None:
        if hasattr(cfg.observations.policy, "actions"):
            cfg.observations.policy.actions = ObsTerm(func=mdp.last_action)
        if hasattr(cfg.observations.policy, "joint_pos"):
            cfg.observations.policy.joint_pos = ObsTerm(
                func=LaggedJointPosRel,
                params={"max_delay": 2, "noise": 0.02},
            )

    # Structural conflict with plant lag (v2 late collapse). Official keeps it; PdGap drops it.
    if hasattr(cfg.rewards, "rear_feet_long_contact"):
        cfg.rewards.rear_feet_long_contact = None

    if getattr(cfg.rewards, "action_rate_l2", None) is not None:
        cfg.rewards.action_rate_l2.weight = -0.015

    # Cap cmd curriculum during plant-gap FT (v2 rushed to ±0.4).
    cfg.curriculum.lin_vel_cmd_levels = CurrTerm(
        func=lin_vel_cmd_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "limit_x": (-0.3, 0.3),
            "limit_y": (-0.2, 0.2),
        },
    )
    # Higher track bar + dwell so scale does not jump 0.15→0.25 in minutes.
    cfg.curriculum.action_scale_levels = CurrTerm(
        func=action_scale_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "levels": (0.15, 0.20, 0.25),
            "threshold_frac": 0.90,
            "min_boundaries_at_level": 8,
        },
    )


def action_scale_levels(
    env,
    env_ids,
    reward_term_name: str = "track_lin_vel_xy_exp",
    levels: tuple[float, ...] = (0.15, 0.20, 0.25),
    threshold_frac: float = 0.90,
    min_boundaries_at_level: int = 8,
):
    """Bump joint_pos.scale up the ladder when tracking reward is healthy."""
    term = env.action_manager.get_term("joint_pos")
    cur = float(term._scale) if not torch.is_tensor(term._scale) else float(term._scale.mean().item())
    # Only evaluate once per episode boundary-ish (same cadence as cmd curriculum).
    if env.common_step_counter % env.max_episode_length != 0:
        return torch.tensor(cur, device=env.device)
    if not hasattr(env, "_action_scale_level_boundaries"):
        env._action_scale_level_boundaries = 0
        env._action_scale_level_value = cur
    if abs(float(env._action_scale_level_value) - cur) > 1e-6:
        env._action_scale_level_value = cur
        env._action_scale_level_boundaries = 0
    env._action_scale_level_boundaries += 1
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if (
        env._action_scale_level_boundaries >= int(min_boundaries_at_level)
        and reward > reward_term.weight * threshold_frac
    ):
        nxt = cur
        for lv in levels:
            if cur + 1e-6 < float(lv):
                nxt = float(lv)
                break
        if abs(nxt - cur) > 1e-6:
            if torch.is_tensor(term._scale):
                term._scale[:] = nxt
            else:
                term._scale = nxt
            if hasattr(term, "cfg"):
                term.cfg.scale = nxt
            env._action_scale_level_value = nxt
            env._action_scale_level_boundaries = 0
            cur = nxt
    return torch.tensor(cur, device=env.device)


def apply_unitree_official_tricks(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Go2HV T–N motor, velocity curriculum, energy + symmetric air-time.

    Does not change sport_stand or obs scales. Heading stays as-is.
    """
    robot = cfg.scene.robot
    cfg.scene.robot = robot.replace(
        actuators={
            "GO2HV": Go2HVActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=25.0,
                damping=0.5,
                friction=0.01,
                effort_limit=23.4,
                velocity_limit=30.0,
            )
        }
    )
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
    cfg.curriculum.lin_vel_cmd_levels = CurrTerm(
        func=lin_vel_cmd_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "limit_x": (-1.0, 1.0),
            "limit_y": (-0.4, 0.4),
        },
    )
    cfg.rewards.energy = RewTerm(func=energy, weight=-2.0e-5)
    cfg.rewards.air_time_variance = RewTerm(
        func=air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )


def apply_nolinvel_tracking_sim2real(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Light gait-init FT on top of a walking sport_stand policy.

    Hold before jump/ramp is cmd=0 but NOT marked standing, so stand_still_*
    only hits true rel_standing envs. No action delay / last_action DR / clip
    change in this pass — those belong in a later sim2real stage.
    """
    # Keep 2499's action-rate scale; -0.10 was grinding the gait away.
    cfg.rewards.action_rate_l2.weight = -0.01
    if getattr(cfg.rewards, "flat_orientation_l2", None) is not None:
        cfg.rewards.flat_orientation_l2.weight = -2.5
    if getattr(cfg.rewards, "stand_still_action_excess", None) is not None:
        cfg.rewards.stand_still_action_excess.weight = -1.0
    if getattr(cfg.rewards, "stand_still_action_l2", None) is not None:
        cfg.rewards.stand_still_action_l2.weight = -0.5
    cfg.rewards.fail_to_walk = RewTerm(
        func=fail_to_walk,
        weight=-0.5,
        params={"command_name": "base_velocity", "cmd_threshold": 0.12, "grace_s": 0.45},
    )
    cfg.rewards.gait_init_action_sat = RewTerm(
        func=gait_init_action_sat,
        weight=-0.8,
        params={"command_name": "base_velocity", "limit": 0.85, "window_s": 0.6},
    )
    cfg.rewards.gait_init_lr_asym = RewTerm(
        func=gait_init_lr_asym,
        weight=-0.3,
        params={"command_name": "base_velocity", "window_s": 0.6},
    )
    cfg.rewards.gait_init_roll_rate = RewTerm(
        func=gait_init_roll_rate,
        weight=-0.2,
        params={"command_name": "base_velocity", "window_s": 1.0},
    )

    old = cfg.commands.base_velocity
    cfg.commands.base_velocity = GaitInitVelocityCommandCfg(
        asset_name=old.asset_name,
        resampling_time_range=(4.0, 10.0),
        rel_standing_envs=0.08,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=old.heading_control_stiffness,
        debug_vis=old.debug_vis,
        ranges=old.ranges,
        gait_init_prob=0.25,
        discrete_vx=(0.15, 0.20, 0.25, 0.30),
        hold_time_s=(1.0, 2.4),
        ramp_prob=0.40,
        ramp_time_s=(0.4, 1.0),
        heading_delay_s=0.8,
    )

    if getattr(cfg.events, "push_robot", None) is not None:
        cfg.events.push_robot.interval_range_s = (5.0, 9.0)
        cfg.events.push_robot.params["velocity_range"] = {
            "x": (-0.6, 0.6),
            "y": (-0.6, 0.6),
            "yaw": (-0.6, 0.6),
        }
    cfg.events.snap_sport_stand = EventTerm(
        func=snap_reset_to_sport_stand,
        mode="reset",
        params={"stand_prob": 0.20, "roll_rad": 0.06, "yaw_rate": 0.35},
    )
    cfg.events.gait_handoff = EventTerm(
        func=perturb_gait_handoff,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "last_action_scale": 0.20,
            "joint_pos_rad": 0.12,
            "stand_snap_prob": 0.05,
        },
    )


def apply_play_viz_overrides(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Turn off remaining DR so Play robots stand at the same height."""
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    if hasattr(cfg.events, "actuator_gains"):
        cfg.events.actuator_gains = None
    if hasattr(cfg.events, "joint_armature"):
        cfg.events.joint_armature = None
    if hasattr(cfg.events, "snap_sport_stand"):
        cfg.events.snap_sport_stand = None
    if hasattr(cfg.events, "gait_handoff"):
        cfg.events.gait_handoff = None
    if hasattr(cfg.commands.base_velocity, "gait_init_prob"):
        cfg.commands.base_velocity.gait_init_prob = 0.0
        cfg.commands.base_velocity.rel_standing_envs = 0.0
    if hasattr(cfg.actions.joint_pos, "delay_max_steps"):
        cfg.actions.joint_pos.delay_min_steps = 0
        cfg.actions.joint_pos.delay_max_steps = 0
    if hasattr(cfg.actions.joint_pos, "max_step_rad"):
        cfg.actions.joint_pos.max_step_rad = 0.0
    if hasattr(cfg.actions.joint_pos, "tau_s"):
        cfg.actions.joint_pos.tau_s = 0.0
    if hasattr(cfg.actions.joint_pos, "scale"):
        cfg.actions.joint_pos.scale = 0.25
    if hasattr(cfg.observations, "policy") and hasattr(cfg.observations.policy, "actions"):
        cfg.observations.policy.actions = ObsTerm(func=mdp.last_action)
    if hasattr(cfg.observations, "policy") and hasattr(cfg.observations.policy, "joint_pos"):
        cfg.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
    if hasattr(cfg.curriculum, "action_scale_levels"):
        cfg.curriculum.action_scale_levels = None
    if hasattr(cfg.curriculum, "plant_hardness_levels"):
        cfg.curriculum.plant_hardness_levels = None
    if getattr(cfg.events, "physics_material", None) is not None:
        cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    if getattr(cfg.events, "reset_robot_joints", None) is not None:
        cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if getattr(cfg.events, "reset_base", None) is not None:
        cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }


def apply_sim2real_domain_rand(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Widen physics / reset / push randomization toward typical sim2real.

    Original Go2 flat (after robot-specific overrides) is almost locked:
    friction (0.8, 0.8)/(0.6, 0.6), no push, reset joints scale (1, 1),
    reset root velocity 0, mass only base −1..+3 kg.
    """
    cfg.events.physics_material.params["static_friction_range"] = (0.3, 1.6)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.2)
    cfg.events.physics_material.params["restitution_range"] = (0.0, 0.4)
    cfg.events.physics_material.params["num_buckets"] = 64
    cfg.events.physics_material.params["make_consistent"] = True

    cfg.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)

    cfg.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-0.6, 0.6)
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.4, 0.4),
        "pitch": (-0.4, 0.4),
        "yaw": (-0.4, 0.4),
    }

    cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
    )
    cfg.events.actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    policy = cfg.observations.policy
    lin = getattr(policy, "base_lin_vel", None)
    if lin is not None:
        lin.noise = Unoise(n_min=-0.2, n_max=0.2)
    policy.base_ang_vel.noise = Unoise(n_min=-0.3, n_max=0.3)
    policy.projected_gravity.noise = Unoise(n_min=-0.08, n_max=0.08)
    policy.joint_vel.noise = Unoise(n_min=-2.0, n_max=2.0)


@configclass
class UnitreeGo2FlatDREnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat Go2 with sim2real-style domain randomization. Policy obs still 48-D."""

    def __post_init__(self):
        super().__post_init__()
        apply_sim2real_training_overrides(self)


@configclass
class UnitreeGo2FlatDREnvCfg_PLAY(UnitreeGo2FlatDREnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)


@configclass
class Go2NoLinVelObservationsCfg:
    """Actor: proprio + command (no GT lin vel). Critic: same plus privileged lin vel."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2EstLinVelObservationsCfg:
    """Actor: 48-D with kinematic lin vel. Critic: 48-D with privileged GT lin vel."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=estimated_base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2FlatEstLinVelEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat Go2: 48-D actor with kinematic lin vel, 48-D privileged critic (GT)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = Go2EstLinVelObservationsCfg()
        apply_sim2real_training_overrides(self)
        # Estimator already is the deploy observation; do not add extra GT-style lin-vel noise.
        self.observations.policy.base_lin_vel.noise = None


@configclass
class UnitreeGo2FlatEstLinVelEnvCfg_PLAY(UnitreeGo2FlatEstLinVelEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False


@configclass
class UnitreeGo2FlatNoLinVelEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat Go2: 45-D actor (no lin vel), 48-D privileged critic."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = Go2NoLinVelObservationsCfg()
        apply_sim2real_training_overrides(self)
        apply_nolinvel_stance_rewards(self)
        apply_nolinvel_tracking_sim2real(self)


@configclass
class UnitreeGo2FlatNoLinVelEnvCfg_PLAY(UnitreeGo2FlatNoLinVelEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False


@configclass
class UnitreeGo2FlatNoLinVelOfficialEnvCfg(UnitreeGo2FlatEnvCfg):
    """45-D Isaac squat default plus Unitree T–N / velocity curriculum / energy.

    Does not use sport_stand. Standing penalties use is_standing_env so the
    ±0.1 curriculum is not treated as stand-still.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations = Go2NoLinVelObservationsCfg()
        apply_sim2real_domain_rand(self)
        apply_nolinvel_stance_rewards(self)
        apply_unitree_official_tricks(self)
        if getattr(self.rewards, "stand_still_action_excess", None) is not None:
            self.rewards.stand_still_action_excess.weight = -2.0
        if getattr(self.rewards, "stand_still_action_l2", None) is not None:
            self.rewards.stand_still_action_l2.weight = -1.0


@configclass
class UnitreeGo2FlatNoLinVelOfficialEnvCfg_PLAY(UnitreeGo2FlatNoLinVelOfficialEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None


@configclass
class UnitreeGo2FlatNoLinVelOfficialPdGapEnvCfg(UnitreeGo2FlatNoLinVelOfficialEnvCfg):
    """Official squat + explicit PD tracking-gap DR for real tip fingerprint."""

    def __post_init__(self):
        super().__post_init__()
        apply_pd_tracking_gap(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPdGapEnvCfg_PLAY(UnitreeGo2FlatNoLinVelOfficialPdGapEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDREnvCfg(UnitreeGo2FlatNoLinVelOfficialEnvCfg):
    """Official squat + minimal deploy-aligned plant DR (v4)."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v4(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDREnvCfg_PLAY(UnitreeGo2FlatNoLinVelOfficialPlantDREnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv5EnvCfg(UnitreeGo2FlatNoLinVelOfficialEnvCfg):
    """Official squat + soft-actuator plant DR (v5): low kp + activity lag."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v5(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv5EnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv5EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv6EnvCfg(UnitreeGo2FlatNoLinVelOfficialEnvCfg):
    """Official squat + soft plant DR v6 (v4 delay + v5 soft kp, no long_contact)."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v6(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv6EnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv6EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7EnvCfg(UnitreeGo2FlatNoLinVelOfficialEnvCfg):
    """Official squat + mild soft plant on v4 (for FT from healthy v4 ckpt)."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v7(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7EnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7SoftStartEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """v7 mild soft-plant + soft-start gait-init (short FT from healthy v4)."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v7_softstart(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7SoftStartEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7SoftStartEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """Official squat + plant hardness curriculum (scratch, deploy kp=25)."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v7_curr(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7AntiCrouchEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """Scratch on delayed plant from iter 0 + anti-crouch rewards."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v7_curr_anticouch(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7AntiCrouchEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7AntiCrouchEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrSlowEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """Official walker FT: frozen full cmd + slower plant hardness climb."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_v7_curr_slow(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrSlowEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrSlowEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        apply_cmd_full_freeze(self)
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage2HoldEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """Pinned stage-2 plant + frozen full cmd. Adapt before RealCal."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_stage2_hold(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage2HoldEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage2HoldEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        apply_cmd_full_freeze(self)
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage1p5HoldEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """Pinned delay 1–3 / τ=0.06 plant + frozen full cmd."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_stage1p5_hold(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage1p5HoldEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage1p5HoldEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        apply_cmd_full_freeze(self)
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialEnvCfg
):
    """Official squat + fixed RealCal plant (2026-08-31 delay/soft card)."""

    def __post_init__(self):
        super().__post_init__()
        apply_plant_dr_realcal(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalHoldEnvCfg(
    UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg
):
    """RealCal plant + frozen full cmd. Safe continue of a finished Curr ckpt."""

    def __post_init__(self):
        super().__post_init__()
        apply_cmd_full_freeze(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalHoldEnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalHoldEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        apply_cmd_full_freeze(self)
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalCmd03EnvCfg(
    UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg
):
    """RealCal plant + frozen vx≈0.3 band for deploy-speed specialty FT."""

    def __post_init__(self):
        super().__post_init__()
        apply_cmd03_focus(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalCmd03EnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalCmd03EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv6Cmd03EnvCfg(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv6EnvCfg
):
    """v6 plant + frozen cmd band around vx≈0.3 for short specialty FT."""

    def __post_init__(self):
        super().__post_init__()
        apply_cmd03_focus(self)


@configclass
class UnitreeGo2FlatNoLinVelOfficialPlantDRv6Cmd03EnvCfg_PLAY(
    UnitreeGo2FlatNoLinVelOfficialPlantDRv6Cmd03EnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False
        # Play: allow pin via --vx; keep clean PD path.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.curriculum.lin_vel_cmd_levels = None
        strip_plant_dr_for_play(self)


def apply_sim2real_squat_training_overrides(cfg: UnitreeGo2FlatEnvCfg) -> None:
    """Same DR as the Sport sets, but keep Isaac's original squat default pose."""
    apply_sim2real_domain_rand(cfg)


@configclass
class UnitreeGo2FlatDRSquatEnvCfg(UnitreeGo2FlatEnvCfg):
    """DR + 48-D actor, Isaac squat default (hip ±0.1, thigh 0.8/1.0, calf -1.5)."""

    def __post_init__(self):
        super().__post_init__()
        apply_sim2real_squat_training_overrides(self)


@configclass
class UnitreeGo2FlatDRSquatEnvCfg_PLAY(UnitreeGo2FlatDRSquatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)


@configclass
class UnitreeGo2FlatNoLinVelSquatEnvCfg(UnitreeGo2FlatEnvCfg):
    """45-D actor / 48-D critic, DR, Isaac squat default."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = Go2NoLinVelObservationsCfg()
        apply_sim2real_squat_training_overrides(self)
        apply_nolinvel_stance_rewards(self)


@configclass
class UnitreeGo2FlatNoLinVelSquatEnvCfg_PLAY(UnitreeGo2FlatNoLinVelSquatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        apply_play_viz_overrides(self)
        self.observations.critic.enable_corruption = False


def register_go2_extra_envs() -> None:
    """Register gym ids. Safe to call more than once."""
    specs = (
        (
            GO2_DR_TASK,
            f"{__name__}:UnitreeGo2FlatDREnvCfg",
        ),
        (
            GO2_DR_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatDREnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelEnvCfg",
        ),
        (
            GO2_NOLINVEL_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelEnvCfg_PLAY",
        ),
        (
            GO2_ESTLINVEL_TASK,
            f"{__name__}:UnitreeGo2FlatEstLinVelEnvCfg",
        ),
        (
            GO2_ESTLINVEL_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatEstLinVelEnvCfg_PLAY",
        ),
        (
            GO2_DR_SQUAT_TASK,
            f"{__name__}:UnitreeGo2FlatDRSquatEnvCfg",
        ),
        (
            GO2_DR_SQUAT_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatDRSquatEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_SQUAT_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelSquatEnvCfg",
        ),
        (
            GO2_NOLINVEL_SQUAT_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelSquatEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PDGAP_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPdGapEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PDGAP_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPdGapEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDREnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDREnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV5_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv5EnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV5_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv5EnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV6_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv6EnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV6_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv6EnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7EnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7EnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_SOFTSTART_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7SoftStartEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_SOFTSTART_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7SoftStartEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_ANTICROUCH_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7AntiCrouchEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_ANTICROUCH_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7AntiCrouchEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_SLOW_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrSlowEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_CURR_SLOW_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7CurrSlowEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE2_HOLD_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage2HoldEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE2_HOLD_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage2HoldEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE1P5_HOLD_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage1p5HoldEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV7_STAGE1P5_HOLD_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv7Stage1p5HoldEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_HOLD_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalHoldEnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_HOLD_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalHoldEnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_CMD03_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalCmd03EnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDR_REALCAL_CMD03_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRRealCalCmd03EnvCfg_PLAY",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV6_CMD03_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv6Cmd03EnvCfg",
        ),
        (
            GO2_NOLINVEL_OFFICIAL_PLANTDRV6_CMD03_PLAY_TASK,
            f"{__name__}:UnitreeGo2FlatNoLinVelOfficialPlantDRv6Cmd03EnvCfg_PLAY",
        ),
    )
    for env_id, entry in specs:
        if env_id in gym.envs.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": entry},
        )


register_go2_extra_envs()
