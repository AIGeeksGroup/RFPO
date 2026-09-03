# Rectified Flow（2-RF Reflow）在本仓库中的实现说明

本文档详细说明：如何在 FPO++（`isaaclab_fpo`）中，基于 Liu et al. (2022) 的 **2-Rectified Flow（Reflow）** 思路，增加一项**可选的辅助训练损失**。

- **参考文献：** Xingchao Liu, Chengyue Gong, Qiang Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, arXiv:2209.03003, 2022.
- **实现状态：** 实验性扩展；**默认关闭**（`reflow_enabled=False`），不影响原版 FPO++ 行为。
- **相关代码：**
  - `isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py` → `get_reflow_loss()`
  - `isaaclab_fpo/isaaclab_fpo/algorithms/fpo.py` → 更新循环
  - `isaaclab_fpo/isaaclab_fpo/rl_cfg.py` → 配置项
  - `isaaclab_fpo/isaaclab_fpo/task_cfgs.py` → Go2 预设 `UnitreeGo2FlatFlowPPORunnerCfgReflow`

---

## 1. 为什么要做这件事

FPO++ 的行动网络是一个 **flow（流模型）**：

- 推理时：从噪声出发，欧拉积分约 **64 步**，得到 12 维关节目标。
- 训练时：用 **CFM（条件流匹配）损失** 配合 FPO 的策略比值更新。

Liu et al. (2022) 指出：普通 flow 从噪声到数据的传输路径可能是**弯曲**的，导致：

1. 推理需要很多步积分；
2. 传输路径不直，采样效率差。

**Rectified Flow** 的核心是：在 **噪声端点 `x0` 与数据端点 `x1` 之间走直线**，并学习常数速度场 `u = x1 - x0`。

**2-Rectified Flow（Reflow）** 进一步：不用「独立采样」的 `(x0, x1)`，而是用**当前已训练 flow 生成耦合对** `(x0, x1')`，再在直线上重训，使路径越来越直。

本实现把 Reflow 作为 **FPO 在线 RL 更新阶段的辅助损失**，在保留 FPO++ 主训练逻辑的同时，鼓励 flow 传输更直。

---

## 2. Liu et al. (2022) 原文在讲什么

### 2.1 1-Rectified Flow（1-RF）

独立采样：

- `x0 ~ π0`（源分布，常为标准高斯噪声）
- `x1 ~ π1`（目标分布，常为目标数据）

在直线路径上插值（原文常用 `t ∈ [0,1]`，`t=0` 为 `x0`，`t=1` 为 `x1`）：

```
x_t = t · x1 + (1 - t) · x0
```

目标速度（常数）：

```
u* = x1 - x0
```

训练 velocity network `vθ(x_t, t)` 去拟合 `u*`（L2 损失）。

### 2.2 2-Rectified Flow（2-RF / Reflow）

1-RF 的端点 `(x0, x1)` 是**独立采样**的，学到的 flow 实际走的路径仍可能弯。

Reflow 的做法：

1. 已有 flow `v1`；
2. 采样 `x0 ~ π0`；
3. 用 `v1` 做 ODE 积分，得到 **耦合端点** `x1' = T_{v1}(x0)`；
4. 在 **直线** `(x0, x1')` 上训练新 flow `v2`；
5. 可迭代多次（2-RF → 3-RF → …），路径越来越直，推理步数可减。

### 2.3 与本仓库 FPO 的关系

| 原文设定 | FPO++ 设定 |
|----------|-----------|
| 生成建模：`x1` 是数据样本 | 控制：`x1` 是 **动作向量**（12 维关节目标） |
| 离线训练 flow | **在线 RL**：FPO 用 CFM 差 + ASPO 更新 |
| 多轮 Reflow 重训新网络 | 本实现：**同一 Actor**，每 mini-batch 加 reflow 辅助项 |
| 无条件或类条件 flow | **状态条件**：`vθ(x_t, t, obs)` |

FPO 论文仅在参考文献 [51] 引用 Rectified Flow，**官方 release 未实现 Reflow**。下文描述的是**本仓库的实验性扩展**。

---

## 3. 本仓库 FPO 的时间参数化（读代码前必看）

FPO 代码里 **时间与 Liu 原文的写法方向相反**，但数学上是同一条直线。

### 3.1 FPO 约定（与 `get_cfm_loss` / `_integrate_flow` 一致）

| 时间 `t` | 含义 |
|----------|------|
| `t = 1` | 噪声端 `x0`（代码里常记为 `eps`） |
| `t = 0` | 动作端 `x1`（数据 / 策略输出） |

直线路径：

```
x_t = t · x0 + (1 - t) · x1        （x0=噪声, x1=动作）
```

常数目标速度（从 `t=1` 积到 `t=0`，`dt < 0`）：

```
u* = x0 - x1
```

推理积分（`act()` / `_integrate_flow`）：

```
从 t=1 的 x_t = x0 出发
重复：x_t ← x_t + uθ(obs, t, x_t) · dt
到 t=0 得到动作
```

默认 `sampling_steps = 64`（Go2）。

### 3.2 与 Liu 原文符号对照

| Liu 原文（t: 0→x0, 1→x1） | 本仓库（t: 1→x0, 0→x1） |
|---------------------------|-------------------------|
| `u* = x1 - x0` | `u* = x0 - x1`（差一个符号，因 t 反向） |
| `x_t = t·x1 + (1-t)·x0` | `x_t = t·x0 + (1-t)·x1` |

**结论：** 都是端点之间的直线插值；实现 Reflow 时必须用 **FPO 的 t 方向**，不能直接照搬原文公式而不改符号。

---

## 4. 本实现「做了什么 / 没做什么」

### 4.1 已实现

| 内容 | 说明 |
|------|------|
| **2-RF 耦合端点** | `x0 ~ N(0,I)`；`x1' = ODE_integrate(当前 Actor, x0)`，且 **`stop_gradient`** |
| **直线路径 CFM** | 在 `(x0, x1')` 上算 `x_t` 与 `u* = x0 - x1'`，MSE 训练 Actor |
| **嵌入 FPO 更新** | 每个 mini-batch：`L = L_FPO + λ · L_reflow` |
| **可配置开关** | 默认关；Go2 有预设 config class |
| **日志** | TensorBoard / 终端输出 `reflow_loss` |

### 4.2 未实现（与原文 / 理想 Reflow 的差距）

| 内容 | 说明 |
|------|------|
| **多轮独立 Reflow 模型** | 原文可训 v1→v2→v3；这里是**同一网络**在线加损失 |
| **离线 Reflow 数据集** | 原文可先大规模生成 `(x0,x1')` 再训；这里每 batch 即时采样 |
| **替换 FPO 主 CFM** | **未替换**；rollout 上的 FPO ratio 仍用 `get_cfm_loss` + rollout 动作 |
| **自动减少 `sampling_steps`** | 需手动做 ablation（如 64 vs 16 步 eval） |
| **Manip / BC 分支** | 当前只在 `isaaclab_fpo` locomotion 的 `FPO.update()` 中 |

---

## 5. 标准 FPO CFM vs Reflow：核心区别

两者都在训练 **同一个 Actor MLP**，但 **端点怎么来** 不同。

### 5.1 标准 FPO：`get_cfm_loss()`

**何时：** 环境交互采集时 + PPO 更新时重算。

**端点：**

- `x0 = eps`：随机高斯噪声（每个 CFM 样本一个）
- `x1 = action`：rollout 里 **环境交互得到的真实动作**（来自当前策略 `act()`）

**损失用途：** 进入 FPO ratio `exp(L_old - L_new)`，驱动 RL。

```text
耦合方式： (obs, rollout_action, eps) — 动作来自仿真轨迹
```

### 5.2 Reflow 辅助：`get_reflow_loss()`

**何时：** 仅 PPO **更新阶段**，每个 mini-batch。

**端点：**

- `x0 ~ N(0,I)`：新采样噪声
- `x1' = Integrate(Actor, x0)`：**当前 Actor 自己积分出来的动作**（detach，不当标签反传）

**损失用途：** 辅助项 `λ · mean(MSE)`，**不进入 FPO ratio**。

```text
耦合方式： (obs, x0) → ODE → x1' — 端点由 flow 自身定义
```

### 5.3 对照表

| | FPO CFM | Reflow 辅助 |
|--|---------|-------------|
| 函数 | `get_cfm_loss` | `get_reflow_loss` |
| 动作端点 x1 | rollout 真实动作 | flow 生成 `x1'` |
| 是否 detach x1' | N/A | **是** |
| 与 RL 关系 | 主策略梯度 | 正则 / 拉直 |
| 每 obs 样本数 | `n_samples_per_action=16` | `reflow_n_samples_per_obs=4`（默认） |

**直觉：** FPO CFM 问「你对 **实际采取的动作** 拟合得好不好」；Reflow 问「你对 **自己从噪声推出来的动作** 能否用 **直线** 拟合」。

---

## 6. 数学形式（与代码一一对应）

给定 mini-batch 观测 `obs`，维度 `(B, obs_dim)`。

### Step 1：采样噪声

```
x0 ~ N(0, I)     shape: (B, S, action_dim)
```

`S = reflow_n_samples_per_obs`（默认 4）。

### Step 2：用当前 Actor 生成耦合端点（无梯度）

对每个 `(obs_i, x0_{i,s})`，展平后调用 `_integrate_flow`：

```
x1'_{i,s} = ODE_{θ}( obs_i, x0_{i,s} )     （stop_gradient）
```

ODE 与推理相同：`sampling_steps` 步欧拉，从 `t=1→0`。

### Step 3：在直线上采样时间 t

```python
uniform_t ~ U(0,1)
t = 0.005 + 0.99 * (1 - (1 - uniform_t)^(1/beta))
```

`beta = cfm_loss_t_inverse_cdf_beta`（与 CFM 相同，默认 1.0 即近似均匀）。  
边界 `0.005 / 0.995` 避免 t 贴边数值问题（与 FPO 一致）。

### Step 4：构造插值点与目标速度

```
x_t = t ⊙ x0 + (1 - t) ⊙ x1'
u*  = x0 - x1'
```

### Step 5：网络预测与损失

```
u_pred = ActorMLP( concat(obs, embed(t), x_t) )
L_reflow = mean( ||u_pred - u*||^2 )    # reduction 同 cfm_loss_reduction
```

### Step 6：并入 FPO 总损失

```
L_total = L_surrogate + c_v · L_value + λ · L_reflow
```

- `λ = reflow_loss_coef`（默认 1.0）
- 然后 `backward()` + 梯度裁剪 + `optimizer.step()`
- **与 FPO surrogate 共用同一个 Adam 优化器**

---

## 7. 训练流程（完整一轮 iteration）

```text
┌─────────────────────────────────────────────────────────────┐
│ A. Rollout（与原版 FPO 完全相同）                              │
│    obs → act() → action → env.step → reward                  │
│    存 initial_cfm_loss（基于 rollout action + eps）           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ B. GAE + advantage（与原版相同）                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ C. Mini-batch 更新（reflow_enabled=True 时多一步）              │
│    1. 重算 CFM → FPO ratio → L_surrogate                      │
│    2. L_value                                                 │
│    3. 【新增】get_reflow_loss(obs_batch) → L_reflow            │
│    4. L_total = L_surrogate + c_v·L_value + λ·L_reflow       │
│    5. backward / clip / step / EMA                            │
└─────────────────────────────────────────────────────────────┘
```

**重要：** Reflow **不参与** rollout 采集，也 **不改变** FPO ratio 的定义。

---

## 8. 代码走读

### 8.1 `ActorCritic.get_reflow_loss()`（核心）

文件：`isaaclab_fpo/modules/actor_critic.py`

```python
# 1) 采样噪声 x0
x0 = torch.randn(batch_size, n_samples_per_obs, action_dim)

# 2) 展平 (B,S) → (B*S)，用 _integrate_flow 得 x1'（no_grad）
with torch.no_grad():
    x1_prime = self._integrate_flow(flat_obs, flat_x0, ...)

# 3) 采样 t，构造 x_t 与 u*
x_t = t * x0 + (1.0 - t) * x1_prime
target_velocity = x0 - x1_prime

# 4) Actor 前向，MSE
velocity_pred = self.actor(cat(obs, embed(t), x_t))
return per_sample_loss.mean()
```

**为何 `x1'` 要 detach？**

- 与 Reflow 论文一致：直线目标端的生成映射视为**固定**，当前步只学「在已知端点对之间走直线」；
- 若不对 `x1'` 停梯度，辅助 loss 会通过 ODE 反传，与 FPO 主目标纠缠，易不稳定。

**为何用 `_integrate_flow` 而非 `_compiled_integrate_flow`？**

- 编译版为推理 CUDA graph 优化；reflow 在训练图内、batch 形状多变，用未编译版更稳妥。

### 8.2 `FPO.update()`（接入点）

文件：`isaaclab_fpo/algorithms/fpo.py`

在每个 mini-batch 算完 `surrogate_loss` 和 `value_loss` 后：

```python
if self.reflow_enabled:
    reflow_loss = self.policy.get_reflow_loss(
        obs_batch, self.reflow_n_samples_per_obs
    )
loss = surrogate_loss + value_loss_coef * value_loss
if self.reflow_enabled:
    loss = loss + self.reflow_loss_coef * reflow_loss
loss.backward()
```

日志字典增加 `reflow_loss`。

### 8.3 配置项

文件：`isaaclab_fpo/rl_cfg.py` → `FpoRslRlPpoAlgorithmCfg`

| 字段 | 默认 | 含义 |
|------|------|------|
| `reflow_enabled` | `False` | 总开关 |
| `reflow_loss_coef` | `1.0` | λ，辅助 loss 权重 |
| `reflow_n_samples_per_obs` | `4` | 每个 obs 上采几组 `(x0, x1')` |

Go2 预设：`UnitreeGo2FlatFlowPPORunnerCfgReflow`（`task_cfgs.py`），`experiment_name = unitree_go2_flat_flow_reflow`。

---

## 9. 如何使用

### 9.1 开启 Reflow 训练

```bash
cd isaaclab_experiments
source source_env.sh

python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --num_envs 2048 \
  --max_iterations 1500 \
  --run_name reflow_v1 \
  agent.experiment_name=unitree_go2_flat_flow_reflow \
  agent.algorithm.reflow_enabled=True \
  agent.algorithm.reflow_loss_coef=1.0 \
  agent.algorithm.reflow_n_samples_per_obs=4
```

### 9.2 关闭（原版 FPO++）

不传上述 `agent.algorithm.reflow_*` 即可；或显式 `reflow_enabled=False`。

### 9.3 建议对比 baseline

你已有的 baseline（2026-08-19，PostEval ~41）：

| 实验 | 日志目录 | 关键开关 |
|------|----------|----------|
| Baseline FPO++ | `.../unitree_go2_flat_flow/2026-08-19_11-24-11/` | reflow 关 |
| Reflow 实验 | `.../unitree_go2_flat_flow_reflow/...` | reflow 开 |

对比指标：

1. **PostEval zero / random 回报**（训练结束自动评）
2. **`Train/mean_reward` 曲线**
3. **`reflow_loss` 是否下降**
4. **（可选）少步推理：** `play_with_viser.py` + 改 `sampling_steps` 或 policy config

---

## 10. 设计选择与取舍

### 10.1 为何做成「辅助 loss」而不是替换 FPO CFM？

- FPO 的理论与实现围绕 **rollout 动作的 CFM ratio** 构建；
- 直接替换会破坏与 ASPO 的配套关系；
- Reflow 作为正则，风险更小，便于 A/B。

### 10.2 为何在 mini-batch 上即时算 `x1'`，而不是离线缓存？

- 在线 RL 的 Actor 每步都在变，离线 couples 很快过期；
- 即时计算与 Reflow「用当前 transport 定义直线」的精神一致；
- 代价：每个更新步多一次 `_integrate_flow`（`S × B` 次），训练稍慢。

### 10.3 `reflow_n_samples_per_obs=4` vs CFM 的 16

- Reflow 仅辅助，默认更小以控算力；
- 可扫：`2 / 4 / 8`。

### 10.4 `reflow_loss_coef`

- 太大：可能压制 FPO surrogate，回报掉；
- 太小：拉直效果弱；
- 建议扫：`0.1, 0.5, 1.0, 2.0`。

---

## 11. 已知局限

1. **不是论文完整 Reflow 流水线**（无 v1→v2 分阶段新模型）。
2. **未证明** 一定提升 PostEval；需实验对比 baseline。
3. **推理步数** 训练仍用 64；Reflow 的价值需通过 **减步 eval** 验证。
4. **仅 locomotion FPO**；`manipulation_experiments` 未接。
5. Reflow 积分在 `get_reflow_loss` 里走 **未 compile** 的 ODE，大 batch 时增加 wall-clock。

---

## 12. 后续可扩展方向

| 方向 | 描述 |
|------|------|
| 多步 Reflow | 每隔 K iteration 用 EMA Actor 生成 couples，单独 reflow phase |
| 减步联合训练 | 同一 loss 内用 `training_sampling_steps` 少步积分生成 `x1'` |
| 与 EMA 联动 | 用 EMA 权重积分得 `x1'`，目标更平滑 |
| Manipulation | 在 `flow_model.py` / `finetune_online_rl.py` 接同样逻辑 |
| 指标 | 显式 log「路径弯曲度」或不同 `sampling_steps` 下的 eval 曲线 |

---

## 13. 附录：伪代码

```python
# === 推理（不变） ===
def act(obs):
    x = noise_or_zero()
    for t from 1 down to 0:
        x = x + actor(obs, t, x) * dt
    return x

# === FPO 主损失（不变） ===
def fpo_update(batch):
    ratio = exp(old_cfm(rollout_action) - new_cfm(rollout_action))
    L_sur = aspo(ratio, advantage)
    L_val = mse(V, returns)
    L = L_sur + cv * L_val
    if reflow_enabled:
        L += lam * reflow_loss(batch.obs)
    L.backward()

# === Reflow 辅助（新增） ===
def reflow_loss(obs):
    x0 = randn()
    x1p = act_integrate(obs, x0).detach()
    t = sample_t()
    xt = t * x0 + (1-t) * x1p
    u_star = x0 - x1p
    u_pred = actor(obs, t, xt)
    return mse(u_pred, u_star)
```

---

## 14. 参考文献

1. Liu, Gong, Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, arXiv:2209.03003, 2022.
2. Yi et al., *Flow Policy Gradients for Robot Control* (FPO++), arXiv:2602.02481, 2026.
3. McAllister et al., *Flow Matching Policy Gradients*, arXiv:2507.21053, 2025.

---

*文档版本：对应本仓库 `isaaclab_fpo` Reflow 实验性实现。若代码变更，以 `get_reflow_loss()` 与 `FPO.update()` 为准。*
