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


RUN_GLOB = "rfpo_scratch*.out"


def canonical_run_name(path: Path) -> str:
    name = path.stem
    for suffix in ("_resume_noopt_fixedsched", "_resume_noopt", "_resume"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@dataclass
class EvalPoint:
    run_name: str
    step: int
    reward: float
    stage: float | None
    pregrasp: float | None
    obj_err: float | None
    source_log: Path


def parse_float(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_log(path: Path) -> list[EvalPoint]:
    if not path.exists():
        return []
    points: list[EvalPoint] = []
    run_name = canonical_run_name(path)
    section = ""
    step = 0
    pending_reward: float | None = None
    pending_stage: float | None = None
    pending_pregrasp: float | None = None
    pending_obj_err: float | None = None
    section_re = re.compile(r"^\|\s*([A-Za-z0-9_./-]+)/\s*\|")
    value_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            section_match = section_re.match(line)
            if section_match:
                if section == "eval" and pending_reward is not None:
                    points.append(EvalPoint(run_name, step, pending_reward, pending_stage, pending_pregrasp, pending_obj_err, path))
                section = section_match.group(1).strip()
                pending_reward = None
                pending_stage = None
                pending_pregrasp = None
                pending_obj_err = None
                continue
            value_match = value_re.match(line)
            if not value_match:
                continue
            key = value_match.group(1).strip()
            value = value_match.group(2).strip()
            parsed = parse_float(value)
            if key == "total_timesteps" and parsed is not None:
                step = int(parsed)
            if section != "eval" or parsed is None:
                continue
            if key == "mean_reward":
                pending_reward = parsed
            elif key == "mean_stage":
                pending_stage = parsed
            elif key.startswith("mean_pregrasp_suc"):
                pending_pregrasp = parsed
            elif key == "mean_obj_com_err":
                pending_obj_err = parsed
    if section == "eval" and pending_reward is not None:
        points.append(EvalPoint(run_name, step, pending_reward, pending_stage, pending_pregrasp, pending_obj_err, path))
    return points


def status_text(root: Path) -> str:
    ps = subprocess.run(
        ["bash", "-lc", "ps -eo pid,etime,args | grep 'python tools/train.py' | grep 'rfpo_scratch' | grep -v grep || true"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    gpu_status = subprocess.run(
        ["bash", "-lc", "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    lines = []
    if ps:
        running = []
        for line in ps.splitlines():
            match = re.match(r"\s*(\d+)\s+(\S+)\s+(.*)", line)
            if not match:
                continue
            pid, elapsed, args = match.groups()
            run_match = re.search(r"state_baseline/([^/\s]+)", args)
            run = run_match.group(1) if run_match else "unknown"
            visible_devices = "?"
            try:
                environ = (Path("/proc") / pid / "environ").read_bytes().split(b"\0")
            except OSError:
                environ = []
            for item in environ:
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    visible_devices = item.split(b"=", 1)[1].decode(errors="ignore") or "?"
                    break
            running.append(f"pid={pid} gpu={visible_devices} elapsed={elapsed} run={run}")
        if running:
            lines += ["运行中的 RFPO 训练：", "```text", "\n".join(running), "```"]
    if gpu_status:
        lines += ["GPU 状态：", "```text", gpu_status, "```"]
    return "\n".join(lines) if lines else "- 暂无运行中的 RFPO scratch 进程。"


def trained_points(points: list[EvalPoint]) -> list[EvalPoint]:
    return [point for point in points if point.step > 0]


def is_solved_point(point: EvalPoint) -> bool:
    stage = -math.inf if point.stage is None else point.stage
    pregrasp = -math.inf if point.pregrasp is None else point.pregrasp
    return stage >= 2.0 and pregrasp >= 0.98


def precision_aware_score(point: EvalPoint) -> float:
    score = point.reward
    if not is_solved_point(point):
        score -= 100.0
    if point.obj_err is None:
        score -= 2.0
    else:
        precision_penalty = max(point.obj_err - 0.0011, 0.0) * 500.0
        score -= min(precision_penalty, 2.0)
    return score


def best_point(points: list[EvalPoint]) -> EvalPoint | None:
    ranking_points = trained_points(points)
    return max(
        ranking_points,
        key=lambda point: (precision_aware_score(point), point.reward, -float("inf") if point.obj_err is None else -point.obj_err),
        default=None,
    )


RFPO_EXPERIENCE_LINES = [
    "",
    "## 最佳 RFPO 经验和参数",
    "",
    "这一节是手工总结，目的是以后重新打开项目时能快速恢复当时的判断。monitor 脚本会保留并重写这段内容。",
    "",
    "### 结论",
    "",
    "- 当前 best76 是 RFPO-from-scratch 路线，不加载 PPO checkpoint，不用 PPO teacher，不用 PPO-BC dataset。",
    "- 训练不是一条从零直接跑到 76 的单次配置，而是先从零训练 RFPO 到可抓取水平，再围绕 RFPO best checkpoint 做保守 refinement。",
    "- 最有效的方向不是加大探索，而是低噪声、小学习率、小 residual、强 precision 约束，让已经会抓取的策略继续压低 object COM error。",
    "- 当前结果视觉上已经能稳定完成 stage-2 抓取/移动，但 reward 仍低于原始 PPO-60M 的约 82-83，所以论文表述不能写成已经超过 PPO-60M。",
    "",
    "### 当前最佳配置",
    "",
    "- 关键 run：`rfpo_scratch_best76_prec_v2_lr35_std009_anchor085_rankelite2_s28999680_lr35em7_std0009_clip0018_80m`。",
    "- 起点：从 RFPO scratch 的 best checkpoint 继续训练，`resume_model` 指向上一轮 RFPO best，不是 PPO checkpoint。",
    "- policy head：`action_head_mode=flow_residual`。",
    "- actor objective：`actor_objective=gaussian_ppo`，`trust_region_mode=ppo`，`fpo_objective_coef=0.0`，`gaussian_objective_coef=1.0`。",
    "- 训练规模：`total_timesteps=80000000`，`n_envs=16`，`n_eval_envs=4`，`eval_freq=200000`，`eval_n_episodes=25`。",
    "- rollout/action：`rollout_deterministic=False`，`eval_deterministic=False`，`sampling_steps=8`，`n_samples_per_action=16`，`n_steps=4096`，`batch_size=512`，`n_epochs=1`。",
    "- 学习率和 trust region：`learning_rate=3.5e-7`，`min_learning_rate=3.5e-7`，`max_learning_rate=3.5e-7`，`clip_range=0.018`，`target_kl=0.00065`，`max_clip_fraction=0.08`。",
    "- 探索噪声：`gaussian_action_std=0.009`，固定不训练；`rollout_action_noise_std=0.0`，`action_perturb_std=0.0`。",
    "- residual：`residual_action_scale=0.006`，`residual_head_zero_init=True`。在纯 `flow_residual` 模式下 residual 更像小修正，不适合放大。",
    "- 梯度限制：`max_grad_norm=0.40`，`actor_max_grad_norm=0.050`，`critic_max_grad_norm=0.40`。",
    "- on-policy anchor：`on_policy_action_anchor_coef=0.085`，`on_policy_action_anchor_min_coef=0.085`，不衰减。",
    "- precision reward：`obj_precision_reward_coef=3.8`，`target=0.0010`，`scale=620.0`，`max=2.5`，stage 至少为 2 才启用。",
    "- precision penalty：`obj_precision_penalty_coef=1150.0`，`target=0.0011`，`max=2.6`。",
    "- precision delta：`obj_precision_delta_coef=330.0`，`target=0.0011`，`max=1.2`。",
    "- AWR/action 辅助项：`advantage_weighted_action_coef=0.00045`，`temp=0.75`，`max_weight=2.0`，positive-only。",
    "- elite precision：`precision_elite_action_coef=0.042`，`err_threshold=0.0021`，`top_fraction=0.045`，`min_fraction=0.025`，`reward_quantile=0.86`，`mode=gaussian_nll`。",
    "- best/rollback：`best_metric=eval/rfpo_precision_score`，`rollback_to_best_on_degrade=True`，`rollback_patience=2`，`degrade_threshold=0.8`，`resume_load_optimizer=False`。",
    "",
    "### 为什么这组有效",
    "",
    "- stage 和 pregrasp 已经解决后，主要瓶颈变成最后几毫米的物体位置误差；所以选择指标不能只看 raw reward，要看 `rfpo_precision_score`。",
    "- 低噪声 `std=0.007-0.010` 比大噪声稳定。大噪声能探索，但后期会破坏已学会的抓取和定位。",
    "- 很小的 LR `2.5e-7-4.0e-7` 比 `1e-6` 级别稳定。RFPO 在这个任务里后期更像策略精修，不像 PPO 那样能承受较大更新。",
    "- `on_policy_action_anchor` 很关键，它限制策略漂移；没有 anchor 或 anchor 太弱时，obj_err 会变大，reward 掉到 60-73。",
    "- `rollback_to_best_on_degrade` 很关键。RFPO 后期经常短暂变好后又退化，自动回滚避免把好 checkpoint 训练坏。",
    "- `precision_elite` 只保留 stage=2 且 obj_err 很小的片段，相当于让策略向自己已经做对的动作收缩。",
    "",
    "### 失败经验",
    "",
    "- 直接加大 residual/action head 更新通常不好：`std=0.03-0.06`、`clip=0.07-0.12`、较大 residual 的版本 reward 经常降到 65-73，obj_err 明显变大。",
    "- 只靠 reward shaping 不能自然突破到 80；它主要把策略稳定在 75-76 附近。",
    "- 过强探索适合早期找会抓的行为，但不适合 best76 附近继续提升精度。",
    "- `flow_plus_mlp_residual` 可以继续探索，但目前没有稳定超过 best76；它不是默认继承路线。",
    "- 用 raw reward 选 best 会误选 obj_err 较差的 checkpoint；必须同时看 `stage=2`、`pregrasp=1`、`obj_com_err`。",
    "",
    "### 复现实验时的顺序",
    "",
    "1. 先确认 best checkpoint 存在：`models/best.pt`。",
    "2. 用当前 best checkpoint 做小范围保守 refinement，优先扫 `lr=2.5e-7..4.0e-7`、`std=0.007..0.010`、`anchor=0.08..0.12`、`clip=0.012..0.020`。",
    "3. 每轮用 `eval/rfpo_precision_score` 选 checkpoint，不要只看 `eval/mean_reward`。",
    "4. 出现 reward 变高后，必须做 deterministic/stochastic 复评并渲染视频，确认不是指标偶然波动。",
    "5. 新结果如果超过当前 best，再更新本文件的“当前最高训练 eval”和这段经验。",
    "",
    "### 重要路径",
    "",
    "- 本地记录：`/home/why/桌面/RFPO_from_scratch_当前最佳实验记录.md`。",
    "- 服务器记录：`/share/project/liyanjun/rongshanyu/RFPO_from_scratch_当前最佳实验记录.md`。",
    "- 结果根目录：`/share/project/liyanjun/rongshanyu/results/vividex_fpo_runtime/results/state_baseline`。",
    "- 日志根目录：`/share/project/liyanjun/rongshanyu/logs`。",
    "- 自动迭代脚本：`scripts/cuihu/auto_iterate_rfpo_scratch.py`。",
    "- 自动记录脚本：`scripts/cuihu/monitor_rfpo_scratch.py`。",
]


def write_record(path: Path, *, root: Path, points: list[EvalPoint]) -> None:
    ranking_points = trained_points(points)
    best = best_point(ranking_points)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    sorted_points = sorted(ranking_points, key=lambda p: (p.step, p.run_name))
    lines = [
        "# RFPO from Scratch 当前最佳实验记录",
        "",
        f"更新时间：{now}",
        "",
        "## 目标",
        "",
        "- 不使用 PPO checkpoint，不使用 PPO-BC 数据，从零训练 RFPO/FPO-style flow policy。",
        "- 目标性能：接近原始 PPO-60M，参考 `stochastic reward = 82.117`，`deterministic reward = 83.541`。",
        "- 当前主指标：训练期间 precision-aware score，要求 stage/pregrasp 已解决，并惩罚 object COM error；出现高分后再做严格 stage-2 headless 100 episodes 复评。",
        "",
        "## 当前状态",
        "",
        status_text(root),
        "",
        "## 当前最高训练 eval",
        "",
    ]
    if best is None:
        lines += ["- 暂无 RFPO scratch eval 结果。"]
    else:
        run_dir = root / "results/vividex_fpo_runtime/results/state_baseline" / best.run_name
        best_ckpt = run_dir / "models/best.pt"
        lines += [
            f"- run：`{best.run_name}`",
            f"- reward：`{best.reward:.6g}`",
            f"- precision-aware score：`{precision_aware_score(best):.6g}`",
            f"- total timesteps：`{best.step}`",
            f"- mean_stage：`{best.stage}`",
            f"- mean_pregrasp_success：`{best.pregrasp}`",
            f"- mean_obj_com_err：`{best.obj_err}`",
            f"- checkpoint：`{best_ckpt}`",
            f"- run dir：`{run_dir}`",
            f"- source log：`{best.source_log}`",
        ]
    lines += [
        "",
        *RFPO_EXPERIENCE_LINES,
        "",
        "## 最近 eval 记录",
        "",
    ]
    for point in sorted_points[-60:]:
        lines.append(
            f"- `{point.run_name}` step `{point.step}` reward `{point.reward:.6g}` "
            f"stage `{point.stage}` pregrasp `{point.pregrasp}` obj_err `{point.obj_err}`"
        )
    if not points:
        lines.append("- 暂无。")
    lines.append("")
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/share/project/liyanjun/rongshanyu")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    record = root / "RFPO_from_scratch_当前最佳实验记录.md"
    while True:
        points: list[EvalPoint] = []
        log_paths = sorted((root / "logs").glob(RUN_GLOB))
        for log_path in log_paths:
            points.extend(parse_log(log_path))
        write_record(record, root=root, points=points)
        if args.once:
            break
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    main()
