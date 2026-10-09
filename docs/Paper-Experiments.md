# Experiments

主结果来自官方 Unitree-Go2 play（与 PPO 教师、域随机化和 1-step 部署同一套 MDP）。IsaacLab Flat Go2 只保留一条先行记录：说明想法先在 FPO++ 上跑通，但该栈缺少可落地的 sim2real 手段，因此主架构改到现在这套。

## Experimental Setup

**Tasks.** 主表：官方 `Unitree-Go2-Velocity` play（256 并行环境 × 10 episodes）。先行对照：IsaacLab Flat Go2，2500 iter，n=2048（仅用于说明 FPO++ 基线上的少步现象，不与主表混绝对值）。Spot / H1 / G1 的官方四档 play 尚未齐。

**Methods.** PPO（官方 Gaussian）；FPO++（无辅助项）；KD-only（仅 PPO 动作蒸馏，从头训）；reflow；reward-aware；reflow+KD；Ours（RA-reflow + 自适应预算 + $p_0=0.25$ 零混 + PPO KD）。

**Eval modes.** `zero` / `random` 只针对流策略的积分初值 $x_0$（零向量 vs $\mathcal{N}(0,I)$），再配欧拉步 $N$。PPO 没有 $x_0$ 也没有 $N$：play 用 `act_inference` 取 Gaussian **均值动作**（确定性单次前向），表里只出现一个数 35.58，不标 zero/random。

**Metrics.** 流策略平均回报（zero / random，$N\in\{64,32,16,8,4,1\}$）；PPO 确定性回报；跟踪误差 $|v_x-\mathrm{cmd}|$（env0）；跌倒比例（基座 $z<0.2$ 时间占比）；HW 指令回放的关节相关与 MAE；onboard 推理时延（mean / p50 / p90 / p99，ms）。

---

## Main Results

**Table 1.** 官方 Go2 play 平均回报。PPO 行的 35.58 是确定性均值动作，与 $x_0$、欧拉步无关；其余行才是 zero / random。

| Method | zero@64 | zero@1 | random@64 | random@1 |
|--------|---------|--------|-----------|----------|
| PPO (det.) | 35.58 | — | — | — |
| FPO++ | 21.29 | −187.7 | 17.66 | −220.1 |
| KD-only | 29.95 | −4919.7 | 27.65 | −5121.7 |
| Ours | 34.78 | 34.25 | 31.56 | 31.50 |

FPO++ 满步低于 PPO，1-step 崩溃。仅 KD 抬高满步至 29.95，1-step 仍崩溃。Ours 满步 34.78，接近 PPO；1-step 为 34.25。少步可用需要 Reflow 与教师对齐同时存在。

**Table 2.** 官方 Go2 play：回报随欧拉步数。

| Method | Mode | 64 | 32 | 16 | 8 | 4 | 1 |
|--------|------|----|----|----|---|---|---|
| FPO++ | zero | 21.29 | 19.06 | 18.78 | 18.56 | 15.43 | −187.7 |
| FPO++ | random | 17.66 | 17.55 | 18.12 | 18.22 | 15.94 | −220.1 |
| KD-only | zero | 29.95 | 29.59 | 29.91 | 26.63 | −398.1 | −4919.7 |
| KD-only | random | 27.65 | 27.70 | 28.39 | 26.50 | −349.6 | −5121.7 |
| Ours | zero | 34.78 | 34.43 | 34.32 | 34.67 | 34.21 | 34.25 |
| Ours | random | 31.56 | 31.58 | 31.73 | 31.82 | 31.96 | 31.50 |
| PPO (det.) | — | 35.58 | — | — | — | — | — |

FPO++ 与 KD-only 在 $N\le 4$ 断崖；Ours 在 zero / random 下全程平坦，支持固定导出 1-step 策略。

**Table 3.** Go2 机载 CPU、ORT 单线程推理时延。Warmup 30，重复 100。对照 50 Hz 控制周期（20 ms）。

| Steps | mean (ms) | p50 | p90 | p99 |
|-------|-----------|-----|-----|-----|
| 64 | 4.39 | 3.16 | 7.93 | 20.06 |
| 32 | 1.50 | 1.49 | 1.52 | 1.56 |
| 16 | 0.79 | 0.77 | 0.84 | 1.07 |
| 8 | 0.41 | 0.40 | 0.42 | 0.44 |
| 4 | 0.23 | 0.22 | 0.23 | 0.26 |
| 1 | 0.08 | 0.08 | 0.08 | 0.10 |

$N\le 32$ 时 p99 远低于 20 ms。$N=64$ 均值 4.39 ms 仍低于周期，但 p99 触及约 20 ms。结合 Table 2，1-step 在回报与时延上同时可部署。

**Table 4.** Go2 Ours@1：钉死真机指令后的仿真–真机关节对齐。

| Run | Command | sim vs HW cmd corr | joint pos corr | joint pos MAE |
|-----|---------|--------------------|----------------|---------------|
| 1 | $v_x$ | 0.984 | 0.992 | 0.085 |
| 2 | $v_y$ | 0.983 | 0.995 | 0.079 |
| 3 | yaw | 0.988 | 0.997 | 0.055 |

指令钉死后关节位置相关 0.98–0.99。关节速度 / gyro 相关低，本协议只支撑位置级对齐，不支撑动力学逐拍复现。

---

## Ablations and Analysis

**Table 5.** 官方 Go2 结构消融（zero）。

| Method | 64 | 32 | 16 | 8 | 4 | 1 |
|--------|----|----|----|---|---|---|
| FPO++ | 21.29 | 19.06 | 18.78 | 18.56 | 15.43 | −187.7 |
| reflow | 33.57 | 33.50 | 33.50 | 33.45 | 33.18 | 32.84 |
| reward-aware | 32.78 | 32.98 | 33.12 | 32.59 | 33.00 | 33.40 |
| KD-only | 29.95 | 29.59 | 29.91 | 26.63 | −398.1 | −4919.7 |
| reflow+KD | 34.65 | 34.44 | 34.77 | 34.77 | 34.92 | 34.39 |
| Ours | 34.78 | 34.43 | 34.32 | 34.67 | 34.21 | 34.25 |

**Table 6.** 同一消融的 random 协议（重跑批次已有，此前未写入正文）。

| Method | 64 | 32 | 16 | 8 | 4 | 1 |
|--------|----|----|----|---|---|---|
| FPO++ | 17.66 | 17.55 | 18.12 | 18.22 | 15.94 | −220.1 |
| reflow | 27.66 | 27.77 | 27.75 | 27.53 | 27.79 | 28.21 |
| reward-aware | 28.52 | 27.84 | 27.41 | 28.09 | 28.06 | 28.17 |
| KD-only | 27.65 | 27.70 | 28.39 | 26.50 | −349.6 | −5121.7 |
| reflow+KD | 31.63 | 31.77 | 31.68 | 31.62 | 31.31 | 32.38 |
| Ours | 31.56 | 31.58 | 31.73 | 31.82 | 31.96 | 31.50 |

仅 Reflow（均匀或 RA）即可把 1-step 回报拉回与 64-step 同量级。仅 KD 不保少步。reflow+KD 与 Ours 接近。

**Table 7.** 固定速度指令跟踪（env0）。含 reward-aware，此前表中漏列。

| Method | $\|v_x-\mathrm{cmd}\|$ @64 / @1 | Fall frac @64 / @1 |
|--------|----------------------------------|--------------------|
| PPO | 0.055 / — | 0.000 / — |
| FPO++ | 0.792 / 0.883 | 0.000 / 0.680 |
| reflow | 0.080 / 0.070 | 0.000 / 0.000 |
| reward-aware | 0.086 / 0.805 | 0.000 / 0.000 |
| KD-only | 0.178 / 0.810 | 0.000 / 0.334 |
| reflow+KD | 0.068 / 0.068 | 0.000 / 0.000 |
| Ours | 0.066 / 0.075 | 0.000 / 0.000 |

Ours@1 跟踪（0.075）接近 PPO（0.055）且无跌倒。reward-aware 的 play 回报在 1-step 仍高（Table 5），但跟踪误差升至 0.805：少步回报不崩不等于指令跟踪可用。reflow 与 reflow+KD 在跟踪上已接近 Ours。

### 先行记录：IsaacLab 上的 FPO++（为何改主架构）

想法先在 IsaacLab Flat Go2、FPO++ 基线上验证。Table 8 只保留这一条：baseline 满步尚可，1-step 掉约 9 分；同一栈上打开 reward-aware Reflow 后 1-step 与 64-step 持平。因此“FPO++ 上少步会塌、Reflow 能托住”不是后来官方 play 才出现的。

该栈没有对齐官方 Unitree 观测/奖励、没有冻结 PPO 教师、也没有按部署协议做域随机化与 HW 指令回放。少步仿真分数保住之后，真机侧仍缺可落地的动作先验和 1-step 导出链路，实际机器效果不好。主实验因此改到现在的架构：官方 Go2 MDP + reward-aware Reflow + PPO 蒸馏 + 1-step ONNX。IsaacLab 的 Spot 全表与 $\lambda_r$ 短消融不再进入正文。

**Table 8.** IsaacLab Flat Go2，2500 iter，n=2048，zero-noise。与主表不可比绝对值。

| Method | 64 | 1 | $\Delta$ |
|--------|----|---|----------|
| FPO++ baseline | 41.89 | 32.60 | −9.29 |
| + reward-aware Reflow | 41.78 | 41.73 | −0.04 |

---

## Qualitative Results

每张图回答一个具体问题，而不是堆曲线。速度图统一为 2×2：前进 0.8 m/s、前进+偏航、横移 0.3 m/s、站立。虚线是指令，实线是实测 $v_x$（蓝）与 $w_z$（橙）。俯视图是同一四条指令下的平面轨迹。

**Figure 1.** 回报随欧拉步数（zero play）。看曲线是否在 $N=64\to 1$ 保持水平。FPO++ 与 KD-only 在 $N\le 4$ 标 × 后崩溃；凡开启 Reflow 的曲线贴着 PPO 横线。这是 Table 2/5 的主图。

![step sweep](paper_figures/summary/step_sweep_zero.png)

**Figure 2.** 机载推理时延。均值与 p99 相对 20 ms（50 Hz）。$N=1$ 约 0.08 ms；$N=64$ 的 p99 顶到周期。和图 1 合读：可以固定 1-step，而不必为保真留 64 步。

![latency](paper_figures/summary/latency.png)

**Figure 3.** 1-step 速度跟踪：谁还能跟上指令。

读图：看实线是否贴住虚线。PPO 与 Ours@1 在前进档能到约 0.8 m/s，偏航有抖动但中心在指令附近。FPO++@1 前进几乎停在 0，$w_z$ 大幅振荡——这就是 Table 2 里 −187 与跌倒率 0.68 的时间域形态。KD-only@1 同样跟不上 $v_x$，对应 −4919。reward-aware@1 回报仍高（Table 5），但 $v_x$ 贴不住指令，对应跟踪误差 0.805。

![PPO velocity](paper_figures/velocity/ppo.png)

![Ours 1-step velocity](paper_figures/velocity/ours1.png)

![FPO++ 1-step velocity](paper_figures/velocity/fpo1.png)

![KD-only 1-step velocity](paper_figures/velocity/kd1.png)

**Figure 4.** 同一四条指令的俯视轨迹。

读图：前进应走出一条长直线；站立/横移应缩在原点附近。PPO 与 Ours@1 前进段接近直线。FPO++@1 与 KD-only@1 前进段短、终点乱绕，站立也不收敛——少步不是“分低一点”，而是走不出指定方向。

![PPO traj](paper_figures/traj/ppo.png)

![Ours 1-step traj](paper_figures/traj/ours1.png)

![FPO++ 1-step traj](paper_figures/traj/fpo1.png)

![KD-only 1-step traj](paper_figures/traj/kd1.png)

**Figure 5.** 先前画过的方形走（Ours，64-step，$|v|=0.3$）。指令按边切换 $v_x/v_y$，检验多轴切换而不是单一步指令。

读图：上两栏虚线是方形边的速度指令，实线应跟着方波走；下栏 $\Delta x,\Delta y$ 应交替增减。当前图里速度大体跟随，但世界坐标 $\Delta x$ 在约 20 s 出现跳变，俯视轨迹被拉成一段水平长线——里程计/记录不连续，不能当成闭环方形几何已经干净。它能说明速度指令可切换，不能单独证明平面方形闭合。

![square tracking](paper_figures/summary/square_tracking.png)

![square traj](paper_figures/summary/square_traj.png)

**Figure 6.** HW 指令回放（Ours@1）。蓝虚线：真机记录的 cmd；橙：仿真侧钉死的同一 cmd；绿：仿真实际速度。

读图：蓝与橙重合说明协议把真机指令钉进了仿真。绿线相对橙线的跟随，是仿真里策略+动力学对这条真机指令的响应，不是真机本体速度。关节对比图是仿真关节目标 vs 真机 `hw_q_cmd`：相关 0.98–0.99 表示 1-step 关节命令与真机执行链路同相位，不表示关节速度或 IMU 已对齐（后两者相关低）。

![run1 cmd](paper_figures/sim2real/run4_cmd.png)

![run1 joints](paper_figures/sim2real/run4_joints.png)

![run2 cmd](paper_figures/sim2real/run5_cmd.png)

![run3 cmd](paper_figures/sim2real/run6_cmd.png)
