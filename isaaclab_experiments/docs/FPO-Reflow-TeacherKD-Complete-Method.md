# Method：在 FPO++ 上的在线 Reflow 与教师蒸馏

对应代码变体 **`all_ideas_teacher_kd`**。

本仓库默认已经是 FPO++：Gaussian actor 已被条件 rectified-flow 替换，用 CFM 比值做 trust-region 更新。我们不改任务、不换策略类，只改速度场怎么训。

相对 baseline，真正用上的只有三件事：

1. **在线 Reflow**：把 2-rectified flow 写成 FPO 更新里的直线回归；
2. **冻结教师端点**：Reflow 的动作端点来自满步 FPO++ 教师，不来自学生自己积分；
3. **少步蒸馏**：学生 \(1/4/8\) 步动作去贴教师的 \(64\) 步动作。

环境、奖励、动作范围、重置分布、在线采集循环都与 FPO++ locomotion 协议相同。

---

## Overview

在环境时刻 \(t\)，策略观察 \(o_t\)，输出连续动作 \(a_t\in\mathbb{R}^{d_a}\)。目标仍是最大化折扣回报
\[
\mathbb{E}_\pi\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right], \tag{1}
\]
其中 \(r_t\) 为速度跟踪 locomotion 的标准奖励。FPO++ 已经能在 \(N=64\) 步欧拉下学到可用的流策略；它并不显式要求传输路径足够直，因此少步积分时动作会偏离满步结果。我们在同一套 FPO++ 更新上加入在线 Reflow 与教师蒸馏，使少步欧拉仍接近满步教师动作。

方法分三层，对应消融：

- **baseline**：只有 \(\mathcal{L}_{\mathrm{FPO}}\)，即仓库里的 FPO++；
- **reflow**：再加上均匀在线 Reflow，端点由学生自己的 \(N\) 步积分给出；
- **完整方法**：Reflow 端点改由冻结教师给出，并加上少步 KD。

训练时采集始终从 \(x_0\sim\mathcal{N}(0,I)\) 起步积分。评测同时报 `zero`（\(x_0=0\)）与 `random`，步数扫 \(k\in\{64,32,16,8,4,1\}\)。部署跑学生场本身的少步积分；教师只在训练时提供端点与蒸馏目标。

---

## A. 预备：FPO++ 流策略

符号与本仓库 FPO++ 一致：\(\tau=1\) 为噪声端，\(\tau=0\) 为动作端，默认欧拉预算 \(N=64\)。给定噪声 \(x_0\sim\mathcal{N}(0,I)\) 与动作端点 \(x_1\)，直线路径与目标速度为
\[
x_\tau=\tau\,x_0+(1-\tau)\,x_1,\qquad
u^\star(x_0,x_1)=x_0-x_1. \tag{2}
\]
Actor 预测 \(v_\theta(o,\tau,x_\tau)\)。从 \(\tau=1\) 积到 \(\tau=0\) 共 \(k\) 步，记端点映射为 \(T_v^{(k)}(o,x_0)\)。训练时执行
\[
a_t=T_{v_\theta}^{(N)}(o_t,x_0),\qquad x_0\sim\mathcal{N}(0,I). \tag{3}
\]

对 buffer 里每条动作，用与 CFM 相同的时间采样构造插值点，计算当前与旧策略的 CFM 损失 \(\ell^{(i,t)}_\theta\)、\(\ell^{(i,t)}_{\theta_{\mathrm{old}}}\)，得到流比值
\[
\hat\rho^{(i)}(\theta)=\exp\bigl(\ell^{(i,t)}_{\theta_{\mathrm{old}}}-\ell^{(i,t)}_\theta\bigr). \tag{4}
\]
Actor 用 ASPO 替代目标（正 advantage 走 PPO clip，负 advantage 走 SPO），Critic 用 bootstrap 价值损失。记这一项为 \(\mathcal{L}_{\mathrm{FPO}}\)。下文都是在同一网络上对 \(\mathcal{L}_{\mathrm{FPO}}\) 的叠加；`baseline` 只保留这一项。

---

## B. 在线 Reflow

独立采样的 \((x_0,x_1)\) 容易学出弯曲传输，少步欧拉误差大。经典 2-Rectified Flow 会先用当前场积分得到耦合端点，再在直线上重训。我们把它写成 **FPO minibatch 里的辅助回归**，不做单独离线阶段。

对 minibatch 中每个观测 \(o_i\)，采 \(n=4\) 个噪声 \(x_0^{i,k}\)，用端点场 \(\bar v\) 做无梯度积分
\[
x_1^{i,k}\leftarrow T_{\bar v}^{(N)}(o_i^{\mathrm{end}},x_0^{i,k}). \tag{5}
\]
均匀 Reflow（消融 `reflow`）取 \(\bar v=v_\theta\)、\(o_i^{\mathrm{end}}=o_i\)。完整方法改用冻结教师，见第 C 节。随后按与 CFM 相同的时间表采样 \(\tau\)，在直线
\[
x_\tau=\tau x_0+(1-\tau)x_1,\qquad u^\star=x_0-x_1
\]
上只对学生场回归
\[
\ell_{i,k}=\bigl\|v_\theta(o_i,\tau,x_\tau)-u^\star\bigr\|^2_{\mathrm{red}}. \tag{6}
\]
均匀平均为
\[
\mathcal{L}_{\mathrm{reflow}}=\frac{1}{Bn}\sum_{i,k}\ell_{i,k}. \tag{7}
\]
完整方法里再按正 advantage 对样本归一化加权（阈值 \(0\)；全零则退回均匀）。系数 \(\lambda_r=1\)。

Reflow 不改变采集：执行动作仍是 (3)。\(\mathcal{L}_{\mathrm{FPO}}\) 负责回报，\(\mathcal{L}_{\mathrm{reflow}}\) 负责把路径掰直。

---

## C. 冻结教师作为 Reflow 终点

若端点来自学生自己的 \(N\) 步积分，直线回归会对准一个仍在变、且早期很弱的耦合。完整方法改用冻结的满步 FPO++ 教师 \(v_{\theta_T}\)：
\[
x_1^{i,k}=T_{v_{\theta_T}}^{(N)}(o_i^{T},x_0^{i,k}), \tag{8}
\]
\(\theta_T\) 不更新。学生只在直线上拟合速度，梯度不进入教师。部署时也不跑教师，只用学生场。

教师与学生的观测维可能不同。教师侧用其 checkpoint 的观测归一化；维数相同时，线速度前三维从 privileged critic 观测拼接，避免教师看到学生自己的速度估计。记 \(o^T=\mathrm{map}_T(o,o_{\mathrm{critic}})\)。

---

## D. 少步教师蒸馏

Reflow 是速度场上的直线回归。部署还需要少步 **积分后的动作** 贴近可靠的满步控制器。令教师与学生共用同一个 \(x_0\)，教师目标为
\[
a_T=T_{v_{\theta_T}}^{(N)}(o^T,x_0)\qquad\text{（截断梯度）}. \tag{9}
\]
学生在 \(K=\{1,4,8\}\) 上蒸馏
\[
\mathcal{L}_{\mathrm{KD}}
=\frac{1}{|K|}\sum_{k\in K}
\bigl\|T_{v_\theta}^{(k)}(o,x_0)-a_T\bigr\|^2. \tag{10}
\]
系数 \(\lambda_{\mathrm{KD}}=0.1\)。目标是教师的 \(64\) 步动作，不是学生自己的满步动作。

辅助损失里的噪声与部署协议对齐：Reflow 与 KD 的 \(x_0\) 以概率 \(p_0=0.25\) 取 \(0\)，否则 \(\mathcal{N}(0,I)\)。**采集不变**，仍全部 random。

---

## E. 总目标与评测

完整方法的 minibatch 损失为
\[
\mathcal{L}
=\mathcal{L}_{\mathrm{FPO}}
+\lambda_r\,\mathcal{L}_{\mathrm{reflow}}
+\lambda_{\mathrm{KD}}\,\mathcal{L}_{\mathrm{KD}}. \tag{11}
\]

一次更新：用 (3) 采集并算 GAE；对每个 minibatch 先算 \(\mathcal{L}_{\mathrm{FPO}}\)，再按 (5)–(8) 算教师端点 Reflow，再按 (9)–(10) 算少步 KD，只对学生参数反传。

评测对学生 checkpoint 扫 `zero` / `random` 与步数 \(\{64,32,16,8,4,1\}\)。主主张是：`zero` 下少步相对满步掉分很小，且满步 `zero` 与 FPO++ baseline 打平。

---

*代码：`--fpo_variant all_ideas_teacher_kd`。*
