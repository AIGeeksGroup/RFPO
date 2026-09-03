"""Batched port of go2_deploy.policy.leg_odom.LegOdom.update.

Deploy runs this on every LowState (~500 Hz):
  v_leg = -J dq - ω × r_foot   (stance average)
  a_lin = accel_imu + g * projected_gravity
  v += dt * a_lin
  v += (K_COMP * dt) * (v_leg - v)
  clip ±2.5

Simulation hooks the same update onto every physics substep, then Euler-substeps
at 2 ms so the discrete filter matches the robot.
"""

from __future__ import annotations

import torch

_LEGS = ("FL", "FR", "RL", "RR")
_HIP_XYZ = torch.tensor(
    [
        [0.1934, 0.0465, 0.0],
        [0.1934, -0.0465, 0.0],
        [-0.1934, 0.0465, 0.0],
        [-0.1934, -0.0465, 0.0],
    ],
    dtype=torch.float32,
)
_THIGH_XYZ = torch.tensor(
    [
        [0.0, 0.0955, 0.0],
        [0.0, -0.0955, 0.0],
        [0.0, 0.0955, 0.0],
        [0.0, -0.0955, 0.0],
    ],
    dtype=torch.float32,
)
_CALF_XYZ = torch.tensor([0.0, 0.0, -0.213], dtype=torch.float32)
_FOOT_XYZ = torch.tensor([0.0, 0.0, -0.213], dtype=torch.float32)
_G = 9.81
_FOOT_FORCE_THRESH = 20.0
_K_COMP = 18.0
_V_CLAMP = 2.5
_JAC_EPS = 1e-5
_DEPLOY_DT = 0.002
_GRAVITY_BIAS_W = torch.tensor([0.0, 0.0, _G], dtype=torch.float32)


def _rx(q: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(q), torch.sin(q)
    z, o = torch.zeros_like(q), torch.ones_like(q)
    row0 = torch.stack([o, z, z], dim=-1)
    row1 = torch.stack([z, c, -s], dim=-1)
    row2 = torch.stack([z, s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _ry(q: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(q), torch.sin(q)
    z, o = torch.zeros_like(q), torch.ones_like(q)
    row0 = torch.stack([c, z, s], dim=-1)
    row1 = torch.stack([z, o, z], dim=-1)
    row2 = torch.stack([-s, z, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _rot_apply(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", r, v)


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors ``v`` into the frame of ``q`` (wxyz)."""
    w = q[:, :1]
    xyz = q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def _expand_vec(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return vec.to(dtype=like.dtype, device=like.device).view(1, 1, 3).expand_as(like)


def feet_pos_body(q_hip: torch.Tensor, q_thigh: torch.Tensor, q_calf: torch.Tensor) -> torch.Tensor:
    hip_xyz = _HIP_XYZ.to(device=q_hip.device, dtype=q_hip.dtype).unsqueeze(0).expand(q_hip.shape[0], -1, -1)
    thigh_xyz = _THIGH_XYZ.to(device=q_hip.device, dtype=q_hip.dtype).unsqueeze(0).expand_as(hip_xyz)
    dummy = hip_xyz
    calf_xyz = _expand_vec(_CALF_XYZ, dummy)
    foot_xyz = _expand_vec(_FOOT_XYZ, dummy)
    r_hip = _rx(q_hip)
    r_th = _ry(q_thigh)
    r_ca = _ry(q_calf)
    p_in_thigh = calf_xyz + _rot_apply(r_ca, foot_xyz)
    p_in_hip = thigh_xyz + _rot_apply(r_th, p_in_thigh)
    return hip_xyz + _rot_apply(r_hip, p_in_hip)


def _leg_jacobian(q_hip: torch.Tensor, q_thigh: torch.Tensor, q_calf: torch.Tensor) -> torch.Tensor:
    """One-sided FD, eps=1e-5, float64 — same as go2_deploy.policy.leg_odom._leg_jacobian."""
    dtype = q_hip.dtype
    q_hip = q_hip.double()
    q_thigh = q_thigh.double()
    q_calf = q_calf.double()
    p0 = feet_pos_body(q_hip, q_thigh, q_calf)
    cols = []
    for which in range(3):
        d_h = _JAC_EPS if which == 0 else 0.0
        d_t = _JAC_EPS if which == 1 else 0.0
        d_c = _JAC_EPS if which == 2 else 0.0
        cols.append((feet_pos_body(q_hip + d_h, q_thigh + d_t, q_calf + d_c) - p0) / _JAC_EPS)
    return torch.stack(cols, dim=-1).to(dtype=dtype)


def _stance_mask(foot_z: torch.Tensor, foot_force: torch.Tensor | None) -> torch.Tensor:
    n = foot_z.shape[0]
    if foot_force is not None:
        usable = foot_force.abs().amax(dim=1) >= _FOOT_FORCE_THRESH
        force_mask = foot_force >= _FOOT_FORCE_THRESH
        n_stance = force_mask.sum(dim=1)
        need_top2 = usable & (n_stance < 1)
        if bool(need_top2.any()):
            top2 = torch.topk(foot_force, k=2, dim=1).indices
            force_mask = force_mask.clone()
            rows = torch.nonzero(need_top2, as_tuple=False).squeeze(-1)
            force_mask[rows, top2[rows, 0]] = True
            force_mask[rows, top2[rows, 1]] = True
        if bool(usable.any()):
            lowest = torch.zeros(n, 4, dtype=torch.bool, device=foot_z.device)
            idx = torch.topk(-foot_z, k=2, dim=1).indices
            lowest.scatter_(1, idx, True)
            return torch.where(usable.unsqueeze(-1), force_mask, lowest)
    lowest = torch.zeros(n, 4, dtype=torch.bool, device=foot_z.device)
    idx = torch.topk(-foot_z, k=2, dim=1).indices
    lowest.scatter_(1, idx, True)
    return lowest


def stance_lin_vel_body(
    q_hip: torch.Tensor,
    q_thigh: torch.Tensor,
    q_calf: torch.Tensor,
    dq_hip: torch.Tensor,
    dq_thigh: torch.Tensor,
    dq_calf: torch.Tensor,
    omega_body: torch.Tensor,
    foot_force: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = feet_pos_body(q_hip, q_thigh, q_calf)
    jac = _leg_jacobian(q_hip, q_thigh, q_calf)
    dq_leg = torch.stack([dq_hip, dq_thigh, dq_calf], dim=-1)
    v_rel = torch.einsum("...ij,...j->...i", jac, dq_leg)
    v_foot = -v_rel - torch.cross(omega_body[:, None, :].expand_as(p), p, dim=-1)
    stance = _stance_mask(p[:, :, 2], foot_force)
    weight = stance.to(dtype=v_foot.dtype)
    denom = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    v = (v_foot * weight.unsqueeze(-1)).sum(dim=1) / denom
    return v, stance


def _name_index(names: list[str], wanted: str) -> int:
    if wanted in names:
        return names.index(wanted)
    matches = [i for i, n in enumerate(names) if n.endswith(wanted) or n.split("/")[-1] == wanted]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve '{wanted}' in {names}")
    return matches[0]


def _cache_ids(env, asset, contact_sensor):
    cached = getattr(env, "_go2_leg_odom_cache", None)
    if cached is not None:
        return cached
    device = asset.data.joint_pos.device
    jnames = list(asset.joint_names)
    bnames = list(contact_sensor.body_names)
    cache = {
        "hip": torch.tensor([_name_index(jnames, f"{leg}_hip_joint") for leg in _LEGS], device=device),
        "thigh": torch.tensor([_name_index(jnames, f"{leg}_thigh_joint") for leg in _LEGS], device=device),
        "calf": torch.tensor([_name_index(jnames, f"{leg}_calf_joint") for leg in _LEGS], device=device),
        "foot": torch.tensor([_name_index(bnames, f"{leg}_foot") for leg in _LEGS], device=device),
    }
    env._go2_leg_odom_cache = cache
    return cache


def _ensure_state(env, n: int, device, dtype):
    if not hasattr(env, "_leg_odom_v") or env._leg_odom_v.shape[0] != n:
        env._leg_odom_v = torch.zeros(n, 3, device=device, dtype=dtype)
        env._leg_odom_prev_vw = torch.zeros(n, 3, device=device, dtype=dtype)
        env._leg_odom_prev_vw_valid = torch.zeros(n, dtype=torch.bool, device=device)


def _imu_accel_body(env, asset, dt: float) -> torch.Tensor:
    """Unitree-style specific force in body frame (includes gravity)."""
    vel_w = asset.data.root_lin_vel_w
    quat = asset.data.root_quat_w
    n = vel_w.shape[0]
    _ensure_state(env, n, vel_w.device, vel_w.dtype)
    dt_safe = max(dt, 1e-4)
    bias = _GRAVITY_BIAS_W.to(device=vel_w.device, dtype=vel_w.dtype)
    a_coord = torch.where(
        env._leg_odom_prev_vw_valid.unsqueeze(-1),
        (vel_w - env._leg_odom_prev_vw) / dt_safe,
        torch.zeros_like(vel_w),
    )
    env._leg_odom_prev_vw = vel_w.clone()
    env._leg_odom_prev_vw_valid[:] = True
    return _quat_rotate_inverse(quat, a_coord + bias)


def _leg_odom_tick(v: torch.Tensor, v_leg: torch.Tensor, a_lin: torch.Tensor, use_comp: torch.Tensor, dt: float):
    """One LegOdom.update integration step."""
    dt = float(max(1e-4, min(0.05, dt)))
    v1 = v + dt * a_lin
    v_comp = v1 + (_K_COMP * dt) * (v_leg - v1)
    alpha = min(1.0, 8.0 * dt)
    v_ema = (1.0 - alpha) * v + alpha * v_leg
    out = torch.where(use_comp.unsqueeze(-1), v_comp, v_ema)
    return torch.clamp(out, -_V_CLAMP, _V_CLAMP)


def update_leg_odom(env, dt: float) -> torch.Tensor:
    """Run one (or more 2 ms) LegOdom updates from current sim sensors."""
    asset = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    ids = _cache_ids(env, asset, contact_sensor)
    q = asset.data.joint_pos
    dq = asset.data.joint_vel
    n = q.shape[0]
    _ensure_state(env, n, q.device, q.dtype)

    reset = torch.zeros(n, dtype=torch.bool, device=q.device)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf == 0
    if bool(reset.any()):
        env._leg_odom_v[reset] = 0.0
        env._leg_odom_prev_vw_valid[reset] = False

    q_hip, q_thigh, q_calf = q[:, ids["hip"]], q[:, ids["thigh"]], q[:, ids["calf"]]
    dq_hip, dq_thigh, dq_calf = dq[:, ids["hip"]], dq[:, ids["thigh"]], dq[:, ids["calf"]]
    forces = contact_sensor.data.net_forces_w
    if forces.ndim == 4:
        forces = forces[:, -1]
    foot_force = forces[:, ids["foot"], 2].abs()
    v_leg, stance = stance_lin_vel_body(
        q_hip, q_thigh, q_calf, dq_hip, dq_thigh, dq_calf, asset.data.root_ang_vel_b, foot_force
    )
    accel_body = _imu_accel_body(env, asset, dt)
    g_b = asset.data.projected_gravity_b
    a_lin = accel_body + _G * g_b
    use_comp = stance.sum(dim=1) >= 1

    v = env._leg_odom_v.to(dtype=v_leg.dtype)
    n_sub = max(1, int(round(float(dt) / _DEPLOY_DT)))
    sub_dt = float(dt) / n_sub
    for _ in range(n_sub):
        v = _leg_odom_tick(v, v_leg, a_lin, use_comp, sub_dt)
    env._leg_odom_v = v
    return v


def install_leg_odom_hook(env) -> None:
    """Update estimator after every physics ``scene.update`` (same role as LowState callback)."""
    if getattr(env, "_leg_odom_hooked", False):
        return
    orig = env.scene.update

    def _update(dt: float, *args, **kwargs):
        orig(dt, *args, **kwargs)
        update_leg_odom(env, float(dt))

    env.scene.update = _update
    env._leg_odom_hooked = True


def estimated_base_lin_vel(env, asset_cfg=None) -> torch.Tensor:
    """Policy obs: latest LegOdom velocity (48-D slot 0:3)."""
    install_leg_odom_hook(env)
    n = env.num_envs
    if not hasattr(env, "_leg_odom_v") or env._leg_odom_v.shape[0] != n:
        dt = float(getattr(env, "physics_dt", _DEPLOY_DT))
        return update_leg_odom(env, dt)
    return env._leg_odom_v
