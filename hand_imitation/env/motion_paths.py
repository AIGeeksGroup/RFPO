from pathlib import Path
import os


def resolve_motion_path(repo_root: Path | str, name: str, norm_traj: bool, robot_name: str) -> Path:
    repo_root = Path(repo_root)
    traj_dir = "norm_trajectories" if norm_traj else "trajectories"
    if robot_name == "rm75_inspire_right":
        override_subdir = os.environ.get("VIVIDEX_RM75_TRAJ_SUBDIR")
        if override_subdir:
            return repo_root / traj_dir / override_subdir / f"{name}.npz"
    if robot_name != "allegro_hand_ur5":
        return repo_root / traj_dir / robot_name / f"{name}.npz"
    return repo_root / traj_dir / f"{name}.npz"


def resolve_existing_motion_path(
    repo_root: Path | str,
    name: str,
    norm_traj: bool,
    robot_name: str,
    allow_legacy_fallback: bool = False,
) -> Path:
    repo_root = Path(repo_root)
    motion_path = resolve_motion_path(repo_root, name, norm_traj, robot_name)
    if motion_path.exists():
        return motion_path

    traj_dir = "norm_trajectories" if norm_traj else "trajectories"
    legacy_path = repo_root / traj_dir / f"{name}.npz"
    if robot_name != "allegro_hand_ur5" and allow_legacy_fallback and legacy_path.exists():
        return legacy_path

    if robot_name != "allegro_hand_ur5":
        raise FileNotFoundError(
            f"Cannot find robot-specific trajectory for {robot_name}: {motion_path}. "
            "Generate it first, e.g. tools/retarget_rm75_inspire_reference.py."
        )
    raise FileNotFoundError(f"Cannot find sequence file: {motion_path}")
