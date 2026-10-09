# Method

完整方法（Ours）只使用：FPO++ 主目标、reward-aware 在线 Reflow、自适应积分预算、辅助零初值混合、冻结 PPO 教师蒸馏。下文只形式化这些项。未进入 Ours 的开关（EMA 端点、random-$x_0$ 一致性损失等）不写入总目标。

## Problem Formulation

条件流策略将观测 $o$ 映射为动作 $a \in \mathbb{R}^{d_a}$。Actor 参数化速度场 $v_\theta(o,t,x)$。时间约定与 FPO 一致：$t=1$ 为噪声端，$t=0$ 为动作端。给定噪声 $x_0 \sim \mathcal{N}(0,I)$ 与动作端点 $x_1$，直线路径与目标速度为

$$
x_t = t\, x_0 + (1-t)\, x_1, \qquad u^\star(x_0,x_1) = x_0 - x_1.
$$

推理时从 $t=1$ 积至 $t=0$，采用 $N$ 步欧拉离散（locomotion 默认 $N=64$）：

$$
a = \mathrm{ODE}_{v_\theta}^{(N)}(o, x_0).
$$

记 $T_v(o,x_0)$ 为场 $v$ 下的 $N$ 步端点映射。

部署要求在更小欧拉预算 $k \ll N$（尤其 $k=1$）下仍保持可用控制。独立采样的 $(x_0,x_1)$ 易诱导弯曲传输，少步离散误差放大，表现为满步尚可而少步回报崩溃。本文在 FPO++ 之上叠加在线 Reflow 与 PPO 动作蒸馏，使速度场趋向更直的传输，并把少步动作拉向已可部署的 Gaussian 先验。

## Overall Framework

基线更新：（i）用 $\pi_\theta$ 采集 rollout（默认从 $x_0 \sim \mathcal{N}(0,I)$ 起步积分）；（ii）用 GAE 估计 advantage $A$；（iii）最小化 FPO surrogate 与 value loss：

$$
\mathcal{L}_{\mathrm{FPO}} = \mathcal{L}_{\mathrm{surr}} + \lambda_V \mathcal{L}_V.
$$

FPO++ 仅优化 $\mathcal{L}_{\mathrm{FPO}}$。Ours 的总目标为

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{FPO}}
+ \lambda_r \mathcal{L}_{\mathrm{reflow}}
+ \lambda_a \mathcal{L}_{\mathrm{ada}}
+ \lambda_{\mathrm{KD}} \mathcal{L}_{\mathrm{KD}}.
$$

默认 $\lambda_r=1$，$\lambda_a=0.1$，$\lambda_{\mathrm{KD}}=0.1$。消融通过关掉其中一项得到，而不是另加未使用的损失。

**Algorithm 1。** 采集 rollout 并计算 $A$。对每个 mini-batch：计算 $\mathcal{L}_{\mathrm{FPO}}$，加入 reward-aware $\mathcal{L}_{\mathrm{reflow}}$、$\mathcal{L}_{\mathrm{ada}}$、$\mathcal{L}_{\mathrm{KD}}$，对 online 参数与步数头 $\psi$ 反传。诊断量不进入梯度。

## Online Reflow

弯曲传输使少步欧拉偏离满步端点。本文将 2-Rectified Flow 的“当前场积分得耦合端点、再在直线段上回归”改写为在线 RL 辅助项。

对观测 mini-batch $\{o_i\}_{i=1}^{B}$：对每个 $o_i$ 采样 $n$ 个噪声 $x_0^{i,k} \sim \mathcal{N}(0,I)$（$n=4$）；端点截断梯度 $x_1^{i,k} \leftarrow T_{v_\theta}(o_i, x_0^{i,k})$（由**学生流**积分，PPO 不作端点）；按 CFM 的 Beta 反 CDF 采样 $t^{i,k}$；在直线段上对 online 场回归

$$
\ell_{i,k}
=
\bigl\| v_\theta(o_i, t^{i,k}, x_t^{i,k}) - u^{i,k} \bigr\|^2_{\mathrm{red}},
\qquad
u^{i,k}=x_0^{i,k}-x_1^{i,k}.
$$

均匀平均对应消融变体 reflow。Ours 使用 advantage 加权：

$$
w_i = \max(A_i-\tau, 0),
\qquad
\tilde{w}_i = \frac{w_i}{\sum_j w_j + \varepsilon},
\qquad
\mathcal{L}_{\mathrm{reflow}} = \sum_{i,k} \tilde{w}_i \ell_{i,k}.
$$

若 $\sum_j w_j \approx 0$，退回均匀权重。$\tau=0$。相对均匀 Reflow，容量集中在高 advantage 状态。

## Adaptive Compute

小型 MLP $\phi_\psi(o)$ 在 $\mathcal{K}=\{1,4,8,16,64\}$ 上输出 logits，$\hat{k}(o)=\arg\max \phi_\psi(o)$。延迟代理与零噪声保真为

$$
L_{\mathrm{lat}}(o)
=
\mathbb{E}_{k \sim \mathrm{softmax}(\phi_\psi(o))}
\Bigl[ \frac{k}{\max \mathcal{K}} \Bigr],
$$

$$
a_N = \mathrm{ODE}_{v_\theta}^{(N)}(o, 0)
\;\text{（截断梯度）},
\quad
a_{\hat{k}} = \mathrm{ODE}_{v_\theta}^{(\hat{k})}(o, 0),
$$

$$
\mathcal{L}_{\mathrm{ada}}
=
\| a_{\hat{k}} - a_N \|^2
+ \alpha \mathbb{E}[L_{\mathrm{lat}}(o)]
- \beta \mathbb{E}[\mathrm{relu}(A)],
$$

$\alpha=1$，$\beta=0.1$。该项与 Reflow 同时开启：Reflow 改善可少步性，步数头决定何时减步。

## PPO Teacher Distillation and Zero-Start Mix

真机部署固定 $N=1$。冻结 Gaussian PPO 只提供动作目标 $a_T=\mathrm{PPO}(o)$（stop-gradient）。对学生满步 / 少步做 MSE：

$$
\mathcal{L}_{\mathrm{KD}}
=
\frac{1}{|K|}
\sum_{k \in K}
\| a^{(k)} - a_T \|^2,
\qquad
K=\{64,8,4\}.
$$

辅助项中以概率 $p_0=0.25$ 将 $x_0$ 置零，使 zero@1 与 random@$N$ 同时可部署。Teacher 不对齐仿真满步回报本身，而是把少步动作拉到已可部署流形上。

## Diagnostics and Evaluation Protocol

每个 update 记录路径中点处 $v_\theta$ 相对 $u^\star$ 的偏差，以及对 $k \in \{1,4,8,16\}$ 的 $\|a_N-a_k\|$。二者只监控，不进入 $\mathcal{L}$。

评测协议：zero@$N$（$x_0=0$）与 random@$N$（$x_0\sim\mathcal{N}(0,I)$），$N\in\{64,32,16,8,4,1\}$。方法主张：Ours 满足 $\mathrm{zero@1}\approx\mathrm{zero@64}$，且满步接近 PPO。

## Sim-to-Real Training Setup

环境对齐官方 Unitree Go2 速度跟踪 MDP（45 维 policy 观测、官方奖励与 PD 接口）。部署导出固定欧拉步策略（真机主用 1-step）。观测不含基座线速度。

域随机化：摩擦 $[0.3,1.2]$，restitution $[0,0.15]$（64 buckets）；基座质量 $[-1,3]$ kg；间隔 5–10 s 推扰 $v_x,v_y\in[-0.5,0.5]$；角速度、投影重力、关节位置/速度加性均匀噪声（与官方 PPO 同量级）。真机评测可钉死记录指令以对齐关节轨迹。

## Variants Used in Experiments

| Name | Reflow | Weighting | Adaptive | Zero-mix $p_0$ | PPO KD |
|------|--------|-----------|----------|----------------|--------|
| FPO++ | off | — | off | 0 | off |
| reflow | on | uniform | off | 0 | off |
| reward-aware | on | advantage | off | 0 | off |
| KD-only | off | — | off | 0 | on |
| reflow+KD | on | uniform | off | 0 | on |
| Ours | on | advantage | on | 0.25 | on |

IsaacLab Flat Go2 上曾用同一套 FPO++ 基线做过先行验证（见 Experiments Table 8）。该栈缺少官方 MDP 对齐、PPO 教师、部署域随机化与 HW 回放，真机效果不好，因此不作为主方法；主架构改为官方 Go2 + RA-reflow + PPO KD + 1-step ONNX。
