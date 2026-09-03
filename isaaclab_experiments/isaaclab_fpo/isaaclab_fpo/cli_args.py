from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlOnPolicyRunnerCfg


def add_fpo_args(parser: argparse.ArgumentParser):
    """Add FPO arguments to the parser."""
    arg_group = parser.add_argument_group("fpo", description="Arguments for FPO agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Frozen baseline checkpoint for reflow endpoints / teacher KD.",
    )
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--fpo_variant",
        type=str,
        default=None,
        choices={
            "baseline",
            "reflow",
            "reflow_adaptive_lambda",
            "reflow_random_x0",
            "reflow_teacher_kd",
            "reward_aware",
            "adaptive_compute",
            "fpo_operator",
            "theory",
            "all_ideas",
            "all_ideas_teacher_kd",
            "all_ideas_fpo",
        },
        help="FPO config variant for Go2 / Spot / G1 velocity tasks (overrides default task config when set).",
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def _fpo_variants_for_task(task_name: str):
    from isaaclab_fpo.task_cfgs import G1_FPO_VARIANTS, GO2_FPO_VARIANTS, SPOT_FPO_VARIANTS

    if "Spot" in task_name:
        return SPOT_FPO_VARIANTS
    if "Unitree-Go2" in task_name or "Go2" in task_name:
        return GO2_FPO_VARIANTS
    # Velocity Flat-G1 only (Tracking-Flat-G1 has no reflow registry).
    if "G1" in task_name and "Tracking" not in task_name:
        return G1_FPO_VARIANTS
    raise KeyError(
        f"No FPO variant registry for task '{task_name}'. "
        "Use a Go2, Spot, or G1 velocity task with --fpo_variant."
    )


def parse_fpo_cfg(task_name: str, args_cli: argparse.Namespace) -> FpoRslRlOnPolicyRunnerCfg:
    """Parse configuration for FPO agent based on inputs.

    Looks up the task config from the isaaclab_fpo registry instead of gym kwargs.
    """
    from isaaclab_fpo.task_cfgs import TASK_CONFIGS

    if hasattr(args_cli, "fpo_variant") and args_cli.fpo_variant is not None:
        variant_registry = _fpo_variants_for_task(task_name)
        if args_cli.fpo_variant not in variant_registry:
            raise KeyError(
                f"Unknown fpo_variant '{args_cli.fpo_variant}' for task '{task_name}'. "
                f"Available: {sorted(variant_registry.keys())}"
            )
        agent_cfg = variant_registry[args_cli.fpo_variant]()
    elif task_name not in TASK_CONFIGS:
        raise KeyError(
            f"No FPO config registered for task '{task_name}'. "
            f"Available tasks: {sorted(TASK_CONFIGS.keys())}"
        )
    else:
        agent_cfg = TASK_CONFIGS[task_name]()
    return update_fpo_cfg(agent_cfg, args_cli)


def update_fpo_cfg(agent_cfg: FpoRslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for FPO agent based on inputs."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "teacher_checkpoint", None):
        agent_cfg.algorithm.reflow_teacher_checkpoint = args_cli.teacher_checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if hasattr(args_cli, "experiment_name") and args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
