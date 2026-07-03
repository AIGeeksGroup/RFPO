import numpy as np


RM75_PREGRASP_HAND_QPOS = np.array([0.08, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)
RM75_CLOSED_HAND_QPOS = np.array([0.85, 0.45, 1.00, 1.00, 1.00, 1.00], dtype=np.float32)
RM75_ACTIVE_HAND_QPOS_INDICES = np.array([0, 1, 5, 7, 9, 10], dtype=np.int64)
RM75_MIMIC_DRIVER_HAND_QPOS_INDICES = np.array([0, 1, 4, 6, 8, 10], dtype=np.int64)
RM75_NO_CONTACT_GRACE_STEPS = 5


def _enforce_rm75_mimic_hand_only_qpos(hand_qpos: np.ndarray, qlimits: np.ndarray) -> np.ndarray:
    hand_qpos = np.asarray(hand_qpos, dtype=np.float32).copy()
    if hand_qpos.shape[0] != 12:
        return hand_qpos
    qlimits = np.asarray(qlimits, dtype=np.float32)
    hand_qpos[2] = 1.334 * hand_qpos[1]
    hand_qpos[3] = 0.667 * hand_qpos[1]
    hand_qpos[5] = 1.06399 * hand_qpos[4] - 0.04545
    hand_qpos[7] = 1.06399 * hand_qpos[6] - 0.04545
    hand_qpos[9] = 1.06399 * hand_qpos[8] - 0.04545
    hand_qpos[11] = 1.06399 * hand_qpos[10] - 0.04545
    return np.clip(hand_qpos, qlimits[:, 0], qlimits[:, 1])


def slice_hand_qpos(qpos: np.ndarray, arm_dof: int) -> np.ndarray:
    return np.asarray(qpos)[arm_dof:]


def recover_hand_action(action: np.ndarray, qlimits: np.ndarray, robot_name: str) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    qlimits = np.asarray(qlimits, dtype=np.float32)
    if robot_name == "rm75_inspire_right":
        if action.shape[0] == qlimits.shape[0] and qlimits.shape[0] != RM75_PREGRASP_HAND_QPOS.shape[0]:
            neutral = neutral_hand_qpos_for_qlimits(qlimits)
            lower_delta = neutral - qlimits[:, 0]
            upper_delta = qlimits[:, 1] - neutral
            qpos = np.where(action < 0.0, neutral + action * lower_delta, neutral + action * upper_delta)
            qpos = np.clip(qpos, qlimits[:, 0], qlimits[:, 1])
            if qlimits.shape[0] == 12:
                qpos = _enforce_rm75_mimic_hand_only_qpos(qpos, qlimits)
            return qpos
        if action.shape[0] != RM75_PREGRASP_HAND_QPOS.shape[0]:
            raise ValueError(
                f"RM75/RH56 hand action must have {RM75_PREGRASP_HAND_QPOS.shape[0]} DoF, "
                f"got {action.shape[0]}"
            )
        active_qlimits = qlimits[RM75_ACTIVE_HAND_QPOS_INDICES] if qlimits.shape[0] == 12 else qlimits
        center = RM75_PREGRASP_HAND_QPOS
        lower_delta = center - active_qlimits[:, 0]
        upper_delta = active_qlimits[:, 1] - center
        active_qpos = np.where(action < 0.0, center + action * lower_delta, center + action * upper_delta)
        if qlimits.shape[0] == 6:
            return active_qpos
        if qlimits.shape[0] != 12:
            raise ValueError(f"RM75/RH56 hand qlimits must have 6 or 12 rows, got {qlimits.shape[0]}")
        return expand_rm75_active_hand_qpos(active_qpos, qlimits)
    return (action + 1.0) / 2.0 * (qlimits[:, 1] - qlimits[:, 0]) + qlimits[:, 0]


def neutral_hand_qpos_for_qlimits(qlimits: np.ndarray) -> np.ndarray:
    qlimits = np.asarray(qlimits, dtype=np.float32)
    return np.clip(np.zeros(qlimits.shape[0], dtype=np.float32), qlimits[:, 0], qlimits[:, 1])


def expand_rm75_active_hand_action(active_action: np.ndarray, hand_action_dim: int) -> np.ndarray:
    active_action = np.asarray(active_action, dtype=np.float32)
    if active_action.shape != (6,):
        raise ValueError(f"Expected RM75/RH56 active hand action with shape (6,), got {active_action.shape}")
    hand_action_dim = int(hand_action_dim)
    if hand_action_dim == 6:
        return active_action.copy()
    if hand_action_dim != 12:
        raise ValueError(f"RM75/RH56 native hand action dim must be 6 or 12, got {hand_action_dim}")
    return np.array(
        [
            active_action[0],
            active_action[1],
            0.0,
            0.0,
            active_action[2],
            0.0,
            active_action[3],
            0.0,
            active_action[4],
            0.0,
            active_action[5],
            0.0,
        ],
        dtype=np.float32,
    )


def expand_rm75_active_hand_qpos(active_qpos: np.ndarray, qlimits: np.ndarray | None = None) -> np.ndarray:
    active_qpos = np.asarray(active_qpos, dtype=np.float32)
    if active_qpos.shape != (6,):
        raise ValueError(f"Expected RM75/RH56 active hand qpos with shape (6,), got {active_qpos.shape}")

    full = np.zeros(12, dtype=np.float32)
    full[0] = active_qpos[0]
    full[1] = active_qpos[1]
    full[2] = 1.334 * active_qpos[1]
    full[3] = 0.667 * active_qpos[1]
    # SAPIEN does not drive the index/middle/ring proximal joints from this
    # URDF reliably.  Use the responsive intermediate joints for those fingers,
    # while pinky remains controlled at the proximal joint.
    full[4] = 0.03
    full[5] = active_qpos[2]
    full[6] = 0.03
    full[7] = active_qpos[3]
    full[8] = 0.03
    full[9] = active_qpos[4]
    full[10] = active_qpos[5]
    full[11] = 1.06399 * active_qpos[5] - 0.04545
    if qlimits is not None:
        qlimits = np.asarray(qlimits, dtype=np.float32)
        full = np.clip(full, qlimits[:, 0], qlimits[:, 1])
    return full


def expand_rm75_mimic_active_hand_qpos(active_qpos: np.ndarray, qlimits: np.ndarray | None = None) -> np.ndarray:
    active_qpos = np.asarray(active_qpos, dtype=np.float32)
    if active_qpos.shape != (6,):
        raise ValueError(f"Expected RM75/RH56 active hand qpos with shape (6,), got {active_qpos.shape}")

    full = np.zeros(12, dtype=np.float32)
    full[RM75_MIMIC_DRIVER_HAND_QPOS_INDICES] = active_qpos
    if qlimits is None:
        default_limits = np.array(
            [
                [0.0, 1.308],
                [0.0, 0.6],
                [0.0, 0.8],
                [0.0, 0.4],
                [0.0, 1.47],
                [-0.04545, 1.56],
                [0.0, 1.47],
                [-0.04545, 1.56],
                [0.0, 1.47],
                [-0.04545, 1.56],
                [0.0, 1.47],
                [-0.04545, 1.56],
            ],
            dtype=np.float32,
        )
        qlimits = default_limits
    return _enforce_rm75_mimic_hand_only_qpos(full, qlimits)


def rm75_hand_qpos_for_qlimits(active_qpos: np.ndarray, qlimits: np.ndarray) -> np.ndarray:
    qlimits = np.asarray(qlimits, dtype=np.float32)
    if qlimits.shape[0] == 6:
        active_qpos = np.asarray(active_qpos, dtype=np.float32)
        if active_qpos.shape != (6,):
            raise ValueError(f"Expected RM75/RH56 active hand qpos with shape (6,), got {active_qpos.shape}")
        return np.clip(active_qpos, qlimits[:, 0], qlimits[:, 1])
    if qlimits.shape[0] == 12:
        return expand_rm75_active_hand_qpos(active_qpos, qlimits)
    raise ValueError(f"RM75/RH56 hand qlimits must have 6 or 12 rows, got {qlimits.shape[0]}")


def rm75_native_active_qpos_from_full(hand_qpos: np.ndarray) -> np.ndarray:
    """Collapse native 12-DoF RH56 hand qpos to the six active close coordinates.

    The no-mimic SAPIEN model often responds through the intermediate joints for
    index/middle/ring, while retargeted references may store the same closure in
    the proximal mimic-driver joints.  Use whichever joint is more closed so
    rewards and diagnostics reflect the physical finger curl.
    """

    hand_qpos = np.asarray(hand_qpos, dtype=np.float32)
    if hand_qpos.shape[0] == 6:
        return hand_qpos.copy()
    if hand_qpos.shape[0] != 12:
        raise ValueError(f"Expected 6 or 12 RM75/RH56 hand qpos values, got {hand_qpos.shape[0]}")
    return np.array(
        [
            hand_qpos[0],
            hand_qpos[1],
            max(float(hand_qpos[4]), float(hand_qpos[5])),
            max(float(hand_qpos[6]), float(hand_qpos[7])),
            max(float(hand_qpos[8]), float(hand_qpos[9])),
            max(float(hand_qpos[10]), float(hand_qpos[11])),
        ],
        dtype=np.float32,
    )


def rm75_active_hand_close_state(active_qpos: np.ndarray) -> tuple[float, float, float, float]:
    active_qpos = np.asarray(active_qpos, dtype=np.float32)
    if active_qpos.shape != (6,):
        raise ValueError(f"Expected RM75/RH56 active hand qpos with shape (6,), got {active_qpos.shape}")
    denom = np.maximum(RM75_CLOSED_HAND_QPOS - RM75_PREGRASP_HAND_QPOS, 1e-6)
    close = np.clip((active_qpos - RM75_PREGRASP_HAND_QPOS) / denom, 0.0, 1.0)
    thumb_close = float(np.mean(close[:2]))
    main_close = float(np.mean(close[2:5]))
    pinky_close = float(close[5])
    close_fraction = float(np.mean(close))
    return close_fraction, thumb_close, main_close, pinky_close


def enforce_rm75_coupled_hand_qpos(qpos: np.ndarray, arm_dof: int, qlimits: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    hand_qpos = qpos[arm_dof:]
    qlimits = np.asarray(qlimits, dtype=np.float32)
    if hand_qpos.shape[0] != 12:
        return qpos
    active_qpos = hand_qpos[RM75_ACTIVE_HAND_QPOS_INDICES].copy()
    # Some legacy RM75 seeds stored the main fingers in the proximal joints
    # even though the driven URDF responds through the intermediate joints.
    # Preserve whichever representation is more closed, then expand once into
    # the controlled 12-DoF layout.
    active_qpos[2] = max(float(active_qpos[2]), float(hand_qpos[4]))
    active_qpos[3] = max(float(active_qpos[3]), float(hand_qpos[6]))
    active_qpos[4] = max(float(active_qpos[4]), float(hand_qpos[8]))
    qpos[arm_dof:] = expand_rm75_active_hand_qpos(active_qpos, qlimits)
    return qpos


def enforce_rm75_mimic_hand_qpos(qpos: np.ndarray, arm_dof: int, qlimits: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    hand_qpos = qpos[arm_dof:].copy()
    if hand_qpos.shape[0] != 12:
        return qpos
    qpos[arm_dof:] = _enforce_rm75_mimic_hand_only_qpos(hand_qpos, qlimits)
    return qpos


def project_rm75_velocity_to_object(
    linear_velocity: np.ndarray,
    palm_pos: np.ndarray,
    object_pos: np.ndarray,
    *,
    tangent_scale: float,
    max_approach_speed: float,
    max_retreat_speed: float,
    approach_bias: float = 0.0,
) -> np.ndarray:
    linear_velocity = np.asarray(linear_velocity, dtype=np.float32)
    palm_pos = np.asarray(palm_pos, dtype=np.float32)
    object_pos = np.asarray(object_pos, dtype=np.float32)
    direction = object_pos - palm_pos
    direction = direction.astype(np.float32, copy=True)
    direction[2] *= 0.35
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return linear_velocity.copy()

    unit = direction / norm
    raw_approach = float(np.dot(linear_velocity, unit))
    approach = raw_approach + max(float(approach_bias), 0.0)
    tangent = linear_velocity - raw_approach * unit
    tangent_scale = float(np.clip(tangent_scale, 0.0, 1.0))
    max_approach_speed = max(float(max_approach_speed), 0.0)
    max_retreat_speed = max(float(max_retreat_speed), 0.0)
    return (
        np.clip(approach, -max_retreat_speed, max_approach_speed) * unit
        + tangent_scale * tangent
    ).astype(np.float32)


def should_terminate_for_contact_loss(
    robot_name: str,
    is_contact: bool,
    current_step: int,
    pregrasp_steps: int,
    no_contact_steps: int,
    grace_steps: int | None = None,
) -> tuple[bool, int]:
    if current_step <= pregrasp_steps:
        return False, int(no_contact_steps)

    if is_contact:
        return False, 0

    no_contact_steps = int(no_contact_steps) + 1
    if robot_name == "rm75_inspire_right":
        grace = RM75_NO_CONTACT_GRACE_STEPS if grace_steps is None else int(grace_steps)
        return no_contact_steps > grace, no_contact_steps
    return True, no_contact_steps


def compute_rm75_grasp_scores(
    *,
    object_lift: float,
    contact_count: float,
    stable_grasp_contact: bool,
    contact_hold_steps: int,
    stable_contact_hold_steps: int,
    thumb_contact: bool,
    non_thumb_contact_count: int,
    object_xy_drift: float,
    object_speed: float,
    object_tilt_err: float,
    object_ang_speed: float,
    lift_target: float = 0.08,
) -> tuple[float, float]:
    lift_target = max(float(lift_target), 1e-6)
    lift_score = 45.0 * min(float(object_lift) / lift_target, 1.0)
    contact_score = 20.0 * min(float(contact_count) / 5.0, 1.0)
    stable_hold_score = 10.0 * min(float(stable_contact_hold_steps) / 6.0, 1.0)
    stable_score = 25.0 if bool(stable_grasp_contact) else 0.0
    drift_penalty = 20.0 * min(float(object_xy_drift) / 0.08, 1.0)
    speed_penalty = 5.0 * min(float(object_speed) / 1.0, 1.0)
    tilt_penalty = 18.0 * min(float(object_tilt_err) / 0.45, 1.0)
    ang_penalty = 7.0 * min(float(object_ang_speed) / 5.0, 1.0)
    grasp_score = (
        lift_score + contact_score + stable_score + stable_hold_score
        - drift_penalty - speed_penalty - tilt_penalty - ang_penalty
    )

    precision_contact_score = 6.0 * (1.0 if bool(thumb_contact) else 0.0)
    precision_contact_score += 5.0 * min(float(non_thumb_contact_count), 2.0)
    precision_stable_score = 32.0 if bool(stable_grasp_contact) else 0.0
    precision_hold_score = 18.0 * min(float(stable_contact_hold_steps) / 8.0, 1.0)
    precision_lift_score = 46.0 * min(float(object_lift) / lift_target, 1.0)
    precision_drift_penalty = 26.0 * min(float(object_xy_drift) / 0.07, 1.0)
    precision_tilt_penalty = 22.0 * min(float(object_tilt_err) / 0.36, 1.0)
    precision_speed_penalty = 6.0 * min(float(object_speed) / 1.0, 1.0)
    precision_ang_penalty = 8.0 * min(float(object_ang_speed) / 5.0, 1.0)
    precision_score = (
        precision_contact_score
        + precision_stable_score
        + precision_hold_score
        + precision_lift_score
        - precision_drift_penalty
        - precision_tilt_penalty
        - precision_speed_penalty
        - precision_ang_penalty
    )
    return float(grasp_score), float(precision_score)


def compute_rm75_task_score(
    *,
    obj_com_err: float,
    obj_rot_err: float = 0.0,
    object_lift: float,
    stable_grasp_contact: bool,
    stable_contact_hold_steps: int,
    thumb_contact: bool,
    non_thumb_contact_count: int,
    object_xy_drift: float,
    object_speed: float,
    object_tilt_err: float,
    object_ang_speed: float,
    lift_target: float = 0.08,
) -> float:
    stable_hold_frac = min(max(float(stable_contact_hold_steps), 0.0) / 8.0, 1.0)
    lift_target = max(float(lift_target), 1e-6)
    lift_frac = min(max(float(object_lift), 0.0) / lift_target, 1.0)
    contact_structure_score = 8.0 * (1.0 if bool(thumb_contact) else 0.0)
    contact_structure_score += 6.0 * min(max(float(non_thumb_contact_count), 0.0), 2.0)
    stable_score = 20.0 if bool(stable_grasp_contact) else 0.0
    stable_hold_score = 18.0 * stable_hold_frac
    lift_gate = stable_hold_frac if bool(stable_grasp_contact) else 0.0
    lift_score = 65.0 * lift_frac * lift_gate
    tracking_err = max(float(obj_com_err), 0.0) + 0.1 * max(float(obj_rot_err), 0.0)
    tracking_score = 35.0 * np.exp(-50.0 * tracking_err) * lift_gate
    drift_penalty = 28.0 * min(max(float(object_xy_drift), 0.0) / 0.07, 1.0)
    tilt_penalty = 24.0 * min(max(float(object_tilt_err), 0.0) / 0.36, 1.0)
    speed_penalty = 6.0 * min(max(float(object_speed), 0.0) / 1.0, 1.0)
    ang_penalty = 8.0 * min(max(float(object_ang_speed), 0.0) / 5.0, 1.0)
    return float(
        contact_structure_score
        + stable_score
        + stable_hold_score
        + lift_score
        + tracking_score
        - drift_penalty
        - tilt_penalty
        - speed_penalty
        - ang_penalty
    )
