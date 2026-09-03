# Go2 FPO++ 训练结果汇总

**实验日期：** 2026-08-19  
**日志目录：** `logs/isaaclab_fpo/unitree_go2_flat_flow/2026-08-19_11-24-11/`  
**最终 checkpoint：** `model_1499.pt`

---

## 1. 实验配置

| 项目 | 本次设置 | README 参考 |
|------|----------|-------------|
| 任务 | `Isaac-Velocity-Flat-Unitree-Go2-v0` | 同 |
| 算法 | FPO++ | 同 |
| 训练迭代 | 1500 | 1500 |
| 并行环境数 | 2048 | 4096（常用默认） |
| 每环境步数 / 轮 | 24 | 24 |
| 随机种子 | 42 | — |
| 设备 | 单卡 NVIDIA RTX 4090 | — |
| 训练模式 | `--headless` | — |

---

## 2. 主指标（与官方对照）

| 指标 | 本次结果（iter 1499） | README 参考（Go2） | 说明 |
|------|----------------------|-------------------|------|
| PostEval **zero** 回报 | **41.02 ± 3.58** | ~40 | 流采样从全 0 噪声起步 |
| PostEval **random** 回报 | **40.11 ± 3.62** | ~40 | 流采样从随机噪声起步 |
| PostEval zero episode 长度 | 994.9 ± 48.5 | ~1000 | 接近满长度，很少提前终止 |
| PostEval random episode 长度 | 994.2 ± 58.9 | ~1000 | 同上 |
| Train **mean_reward**（训练日志） | **39.80** | ~40 | 最后一轮 rollout 平均即时奖励 |

---

## 3. 训练曲线关键点

| 迭代 | Train mean_reward | PostEval zero | PostEval random |
|------|-------------------|---------------|-----------------|
| 0 | -0.49 | 9.54 ± 7.12 | -15.51 ± 6.00 |
| 750 | 37.55 | 40.41 ± 0.97 | 37.49 ± 3.22 |
| **1499** | **39.80** | **41.02 ± 3.58** | **40.11 ± 3.62** |

---

## 4. 分项奖励（Train 日志，iter 1499）

| 奖励项 | iter 0 | iter 1499 | 趋势 |
|--------|--------|-----------|------|
| track_lin_vel_xy_exp | 0.004 | **1.442** | 线速度跟踪 ↑ |
| track_ang_vel_z_exp | 0.003 | **0.710** | 角速度跟踪 ↑ |
| lin_vel_z_l2 | -0.015 | -0.010 | 上下蹦惩罚 |
| ang_vel_xy_l2 | -0.007 | -0.013 | 翻滚惩罚 |
| dof_torques_l2 | -0.002 | -0.066 | 力矩惩罚 |
| dof_acc_l2 | -0.005 | -0.023 | 加速度惩罚 |
| action_rate_l2 | -0.003 | -0.020 | 动作突变惩罚 |
| feet_air_time | -0.000 | -0.017 | 抬脚奖励 |
| flat_orientation_l2 | -0.000 | -0.004 | 姿态惩罚 |
| dof_pos_limits | 0.000 | 0.000 | 未启用 |

---

## 5. 结论（可作报告摘要）

| 结论项 | 内容 |
|--------|------|
| 复现状态 | 成功：PostEval 回报 **~41 / ~40**，与 README 给出的 Go2 **~40** 一致 |
| 与论文设定差异 | 并行环境为 **2048**（非 4096），结果仍达标 |
| 行为表现 | episode 长度接近 **1000**，表明策略可稳定走完整局 |
| 可视化 | Viser 回放可观察到正常行走；`num_envs=1` 时速度箭头有已知 bug |

---

## 6. 佐证文件

| 类型 | 路径 |
|------|------|
| TensorBoard | `logs/.../2026-08-19_11-24-11/events.out.tfevents.*` |
| 超参 | `logs/.../2026-08-19_11-24-11/params/agent.yaml` |
| 环境配置 | `logs/.../2026-08-19_11-24-11/params/env.yaml` |
| 模型权重 | `logs/.../2026-08-19_11-24-11/model_*.pt` |
| 官方预期曲线 | `isaaclab_experiments/expected_training_curves_locomotion.png` |
| 官方预期评估 | `isaaclab_experiments/expected_eval_curves_locomotion.png` |

---

*数据来源：TensorBoard 标量日志 + 训练结束 Post-training checkpoint evaluation。*
