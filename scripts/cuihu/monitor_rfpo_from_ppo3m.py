#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


RUN_NAMES = [
    "rfpo_from_ppo3m_lownoise_s012to006_lr5e6_20m",
    "rfpo_from_ppo3m_tiny_s010to004_lr2e6_20m",
    "rfpo_from_ppo3m_mid_s012to006_lr3e6_clip08_20m",
]


@dataclass
class EvalPoint:
    run_name: str
    step: int
    reward: float
    source_log: Path


def _parse_float(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_training_log(path: Path) -> list[EvalPoint]:
    if not path.exists():
        return []
    current_section = ""
    current_step = 0
    points: list[EvalPoint] = []
    run_name = path.stem
    section_re = re.compile(r"^\|\s*([A-Za-z0-9_./-]+)/\s*\|")
    value_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            section_match = section_re.match(line)
            if section_match:
                current_section = section_match.group(1).strip()
                continue
            value_match = value_re.match(line)
            if not value_match:
                continue
            key = value_match.group(1).strip()
            value = value_match.group(2).strip()
            if key == "total_timesteps":
                parsed = _parse_float(value)
                if parsed is not None:
                    current_step = int(parsed)
            if current_section == "eval" and key == "mean_reward":
                reward = _parse_float(value)
                if reward is not None:
                    points.append(EvalPoint(run_name, current_step, reward, path))
    return points


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def write_record(
    path: Path,
    *,
    root: Path,
    points: list[EvalPoint],
    strict_best_text: str,
    pipeline_status: str,
) -> None:
    best = max(points, key=lambda x: x.reward, default=None)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# RFPO from PPO-3M 当前最佳实验记录",
        "",
        f"更新时间：{now}",
        "",
        "## 目标",
        "",
        "- 从 `PPO-3M` teacher 开始。",
        "- 流程：`PPO-3M -> stage-2 successful BC dataset -> Flow-BC -> RFPO online fine-tuning`。",
        "- 目标参考：`PPO-60M stochastic reward = 82.117`，`PPO-60M deterministic reward = 83.541`。",
        "- 记录规则：训练日志里的 eval 只作为巡检信号，最终 best 以严格 stage-2 headless 100 episodes 复评为准。",
        "",
        "## 当前状态",
        "",
        pipeline_status.strip() or "- 未读取到流水线状态。",
        "",
        "## 当前训练日志最高分",
        "",
    ]
    if best is None:
        lines += [
            "- 暂无 RFPO-from-PPO3M eval 结果。",
            "- 通常表示 PPO-3M、BC 数据或 Flow-BC 仍在前置阶段。",
        ]
    else:
        run_dir = root / "results/vividex_fpo_runtime/results/state_baseline" / best.run_name
        best_ckpt = run_dir / "models/best.pt"
        last_ckpt = run_dir / "models/last.pt"
        lines += [
            f"- run：`{best.run_name}`",
            f"- eval reward：`{best.reward:.6g}`",
            f"- total timesteps：`{best.step}`",
            f"- run dir：`{run_dir}`",
            f"- preferred checkpoint：`{best_ckpt if best_ckpt.exists() else last_ckpt}`",
            f"- source log：`{best.source_log}`",
        ]
    lines += [
        "",
        "## 严格复评最高分",
        "",
        strict_best_text.strip() or "- 暂无严格 stage-2 headless 100 episodes 复评结果。",
        "",
        "## 最近 eval 记录",
        "",
    ]
    for point in sorted(points, key=lambda x: (x.run_name, x.step))[-30:]:
        lines.append(f"- `{point.run_name}` step `{point.step}` reward `{point.reward:.6g}`")
    if not points:
        lines.append("- 暂无。")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def pipeline_status(root: Path) -> str:
    ppo_ckpt = root / "results/vividex_fpo_runtime/results/state_baseline/ppo_mustard_3m_teacher/logs/rl_models_3000000_steps.zip"
    bc_data = root / "data/fpo_bc/mustard_ppo3m_stage2.npz"
    bc_ckpt = root / "results/fpo_bc_mustard_ppo3m_value/flow_bc_step_200000.pt"
    parts = [
        f"- PPO-3M checkpoint：`{'exists' if ppo_ckpt.exists() else 'missing'}`",
        f"- PPO-3M BC dataset：`{'exists' if bc_data.exists() else 'missing'}`",
        f"- Flow-BC checkpoint：`{'exists' if bc_ckpt.exists() else 'missing'}`",
    ]
    ps = subprocess.run(
        ["bash", "-lc", "ps -eo pid,etime,args | grep -E 'run_ppo3m|rfpo_from_ppo3m|tools/train.py|collect_fpo|pretrain_fpo' | grep -v grep || true"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    if ps:
        parts += ["", "运行中的相关进程：", "```text", ps, "```"]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/share/project/liyanjun/rongshanyu")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    logs = root / "logs"
    record = root / "RFPO_from_PPO3M_当前最佳实验记录.md"
    strict = root / "RFPO_from_PPO3M_strict_best.txt"
    while True:
        points: list[EvalPoint] = []
        for run_name in RUN_NAMES:
            points.extend(parse_training_log(logs / f"{run_name}.out"))
        write_record(
            record,
            root=root,
            points=points,
            strict_best_text=read_text(strict),
            pipeline_status=pipeline_status(root),
        )
        if args.once:
            break
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    main()
