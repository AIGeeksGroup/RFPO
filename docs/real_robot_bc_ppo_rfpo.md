# 真机bc+ppo/rfpo流程

本文说明如果要把当前 RFPO / PPO 训练流程接到真实机器人上，需要准备什么、BC 数据应该是什么格式、如何接入现有 pipeline，以及如果要做真机在线 RL 还需要额外实现哪些模块。

## 总体结论

当前仓库里的 PPO / RFPO 算法代码可以复用，但真机部署不是把模拟器代码原封不动搬过去就能跑。

最推荐的路线是：

```text
真机采集示教数据
-> 转成当前环境一致的 observations/actions NPZ
-> 用 NPZ 做 BC warm start
-> PPO / RFPO 主要在仿真里继续训练
-> 最后部署或小步微调到真机
```

如果要在真机上做在线 PPO / RFPO，则还必须实现一个真实机器人 Gym 环境，提供 `reset()`、`step(action)`、`observation_space`、`action_space`、reward、done、success、安全保护和复位机制。

## 必须对齐的东西

真机不要求和模拟器在物理外观上完全一样，但 policy 的接口必须一致。

必须完全一致的是：

- `observation` 每一维的含义、顺序、单位、坐标系。
- `action` 每一维的含义、顺序、范围、控制频率。
- 机器人 DOF、关节方向、关节限位、手指开合语义。
- 物体位姿、目标位姿、机器人状态是在什么 frame 下表达的。
- 策略推理时使用的 obs/action scale 和归一化方式。

当前 `GymWrapper` 的动作空间是连续 `[-1, 1]`：

```python
spaces.Box(low=-1, high=1, shape=(base_env.action_dim,), dtype=np.float32)
```

动作维度来自：

```python
action_dim = 6 + robot_info.hand_dof
```

因此常见维度是：

- `allegro_hand_ur5`: `6 + 16 = 22`
- `rm75_inspire_right` native12: `6 + 12 = 18`
- `rm75_inspire_right` active6: `6 + 6 = 12`

注意：前 6 维不是 6 个或 7 个机械臂关节角，而是当前环境定义下的掌心/末端控制动作；后面的维度才是手部动作。真机采集到的底层电机指令、关节角、笛卡尔速度不能直接塞进 `actions`，需要先转换成这套环境 action space 中的归一化动作。

## 相机和坐标系

如果策略是 state-based policy，也就是当前这套训练主要使用的低维状态输入，那么真机相机不需要和模拟器视角完全一样。

但是必须保证最终喂给 policy 的状态语义一致：

```text
真机相机或其他传感器检测 object pose
-> 通过标定外参转换到 robot/world frame
-> 拼成和模拟器 get_observation() 一致的 observation
-> 输入 policy
```

也就是说，相机可以换位置，但转换后的物体位姿、目标位姿、机器人状态必须和训练时的 observation 定义一致。

如果以后要训练 raw image / ego-centric 视觉策略，要求会更高。相机内参、外参、分辨率、裁剪方式、光照、手和物体在画面里的尺度都会影响 sim2real。那时需要尽量对齐相机设置，或者做 domain randomization、多视角训练、真机视觉适配。

## 真机 BC 数据需要准备什么

BC 数据不是原始视频，也不是原始机器人日志，而是已经对齐到当前环境观测空间和动作空间的监督学习数据。

必须字段：

- `observations`: `float32`，shape `[N, obs_dim]`
- `actions`: `float32`，shape `[N, action_dim]`

建议字段：

- `episode_ends`: `int64`，每条 episode 结束时的累计 transition 下标。
- `rewards`: `float32`，如果真机上能算 reward，可以保存。
- `robot_name`: 字符串，例如 `rm75_inspire_right`。
- `metadata/config_json`: 采集配置、坐标系、控制频率、相机标定版本等。

最小保存格式如下：

```python
import numpy as np

np.savez_compressed(
    "rm75_real_bc.npz",
    observations=observations.astype(np.float32),  # [N, obs_dim]
    actions=actions.astype(np.float32),            # [N, action_dim]
    episode_ends=episode_ends.astype(np.int64),    # optional
    rewards=rewards.astype(np.float32),            # optional
    robot_name=np.asarray("rm75_inspire_right"),
)
```

现有代码读取 BC 数据时只强制要求 `observations` 和 `actions` 两个数组。PPO BC、FPO BC、在线 BC anchor 都是读这两个字段。

## 如何采集和转换真机数据

真机采集时，建议每一步保存原始信息和转换后的训练信息。

原始日志建议包括：

- 时间戳。
- 机器人关节位置、速度、力矩或电流。
- 末端/掌心位姿。
- 手指关节位置。
- 控制器实际下发的命令。
- 物体 6D pose 或至少物体中心位置。
- 目标物体 pose / target pose。
- 相机图像、检测结果、相机外参版本。
- 当前 episode id、是否成功、是否失败、失败原因。

然后写一个 adapter，把原始日志转换成训练数据：

```text
real_robot_log
-> real state adapter
-> observation, shape [obs_dim]

real control command
-> action adapter
-> normalized action in [-1, 1], shape [action_dim]
```

转换时需要重点检查：

- `observations.shape[1]` 必须等于训练环境的 `env.observation_space.shape[0]`。
- `actions.shape[1]` 必须等于训练环境的 `env.action_space.shape[0]`。
- 所有数值必须 finite，不能有 NaN / Inf。
- action 必须已经 clip 或 normalize 到 `[-1, 1]`。
- 坐标单位一般使用米和弧度，不能混入毫米或角度制。
- 坐标 frame 必须和仿真 observation 一致。

建议写一个检查脚本：

```python
import numpy as np

data = np.load("rm75_real_bc.npz")
obs = np.asarray(data["observations"], dtype=np.float32)
act = np.asarray(data["actions"], dtype=np.float32)

assert obs.ndim == 2
assert act.ndim == 2
assert obs.shape[0] == act.shape[0]
assert np.isfinite(obs).all()
assert np.isfinite(act).all()
assert (act >= -1.0001).all() and (act <= 1.0001).all()

print(obs.shape, act.shape)
```

## 接入 PPO 的 BC warm start

先用真机 BC 数据预训练 PPO actor：

```bash
python tools/pretrain_ppo_bc.py \
  --dataset /path/to/rm75_real_bc.npz \
  --output-dir /path/to/ppo_bc \
  --robot-name rm75_inspire_right
```

输出通常是：

```text
/path/to/ppo_bc/bc_ppo_last.zip
```

然后接 PPO 在线训练：

```bash
python tools/train.py agent=ppo \
  resume_model=/path/to/ppo_bc/bc_ppo_last.zip \
  agent.params.bc_anchor_dataset=/path/to/rm75_real_bc.npz \
  agent.params.bc_anchor_coef=1.0 \
  agent.params.bc_anchor_min_coef=0.2
```

这里 `resume_model` 是 PPO 的 BC 初始化权重，`bc_anchor_dataset` 是在线 PPO 训练期间额外加的 BC anchor loss，用于防止策略很快漂离示教分布。

## 接入 RFPO / FPO 的 BC warm start

先用真机 BC 数据预训练 FPO state policy：

```bash
python tools/pretrain_fpo_bc.py \
  --dataset /path/to/rm75_real_bc.npz \
  --output_dir /path/to/flow_bc
```

输出通常是：

```text
/path/to/flow_bc/flow_bc_last.pt
```

然后接 RFPO / FPO 在线训练：

```bash
python tools/train.py agent=fpo \
  agent.params.bc_checkpoint=/path/to/flow_bc/flow_bc_last.pt \
  agent.params.bc_anchor_dataset=/path/to/rm75_real_bc.npz \
  agent.params.action_anchor_coef=0.2 \
  agent.params.action_anchor_min_coef=0.0
```

这里 `bc_checkpoint` 是 RFPO/FPO 的 BC 初始化权重，`bc_anchor_dataset` 可以继续作为在线训练时的动作监督约束。

## 真机在线 PPO / RFPO 需要额外实现什么

如果在线训练发生在 SAPIEN 里，当前代码已经有对应环境。

如果在线训练发生在真机上，则需要自己实现真实机器人环境，形式上类似：

```python
class RealRobotRelocateEnv:
    observation_space = ...
    action_space = ...

    def reset(self):
        # 复位机器人、物体、目标
        # 可以自动复位，也可以等待人工复位
        return obs

    def step(self, action):
        # 1. action [-1, 1] -> 真机控制命令
        # 2. 执行一个控制周期或若干控制周期
        # 3. 读取传感器和机器人状态
        # 4. 计算 obs, reward, done, info
        return obs, reward, done, info
```

真机在线 RL 的循环是：

```text
policy 输出 action
-> 真机执行 action
-> 传感器读回 observation
-> 计算 reward / done / info
-> PPO / RFPO 用 rollout 更新
-> 成功、失败或超时后 reset
-> 进入下一条 episode
```

## Reward、value 和 done 怎么来

`value` 不需要人工标注。PPO / RFPO 里的 value function 是 critic 网络根据 rollout 里的 reward 序列自己学习出来的。

需要人工设计和实现的是：

- `reward`
- `done`
- `success`
- `failure`
- `reset`
- 安全停止逻辑

真机上可以沿用仿真 reward 的思想，但必须能被传感器测出来。

一个最小 reward 可以是：

```text
reward = - object_to_target_distance
       + lift_bonus
       + success_bonus
       - safety_penalty
```

常见组成：

- 物体离目标位置越近，reward 越高。
- 物体被 lift 到一定高度，给 bonus。
- 物体稳定在目标区域若干帧，给 success bonus。
- 掉落、推歪、超出工作区、接近关节限位、力矩过大，给 penalty 或直接 done。
- 超过最大步数，done。

success / done 可以这样定义：

```text
success:
  object 到 target 距离 < 阈值，并稳定若干帧

failure:
  物体掉落
  物体被推出有效区域
  机器人接近限位
  碰撞或力矩超过安全阈值
  episode 超时
```

如果没有可靠的物体 6D pose、目标 pose、lift 高度估计，就很难做真机在线 RL。真机通常需要 AprilTag、外部相机、腕部相机、motion capture、可靠 6D pose 模型或其他传感器来计算 reward。

## Reset 是真机在线训练的主要难点

真机在线训练不是只调用算法就行。每条 episode 结束后，环境必须能恢复到下一次训练的初始状态。

reset 方式可以是：

- 人工摆回物体，效率最低但实现简单。
- 半自动：机器人回安全位，等待人确认物体已复位。
- 自动 reset：机器人或额外机构把物体放回初始区域。
- 训练任务设计成不需要严格复位，例如目标区域随机但物体当前位置可作为下一次初始状态，不过 reward 和数据分布要重新设计。

如果 reset 不稳定，在线 RL 的数据会非常脏，训练结果通常不可控。

## 安全要求

真机 PPO / RFPO 会探索。探索意味着策略可能输出不合理动作，所以必须有安全层。

至少需要：

- action clip。
- 速度、加速度、力矩、电流限制。
- 工作空间限制。
- 关节限位保护。
- 碰撞检测或力控保护。
- 急停。
- 人员安全区域。
- 失败检测后立即停止当前 episode。

建议不要让 RL policy 直接控制最底层电机。更稳的方式是：

```text
policy action [-1, 1]
-> safety/action adapter
-> 末端位姿增量、速度命令或手指开合目标
-> 真机低层控制器
```

## 推荐实施顺序

1. 在仿真里固定最终要部署的 obs/action 定义。
2. 写真机 observation adapter，确保真机 obs 和仿真 obs 逐维一致。
3. 写真机 action adapter，确保 policy action 和真机控制命令一一对应。
4. 采少量真机示教，保存原始日志和转换后的 `observations/actions`。
5. 用检查脚本验证 NPZ 维度、范围、坐标系、NaN/Inf。
6. 用 `tools/pretrain_ppo_bc.py` 和 `tools/pretrain_fpo_bc.py` 分别做 BC。
7. 在仿真里用真机 BC checkpoint 做 PPO/RFPO 训练，先确认流程能跑。
8. 真机只做离线 replay 或短 episode 评测。
9. 如果需要真机在线 RL，再实现 `RealRobotRelocateEnv`、reward、done、reset、安全层。
10. 从极小 action scale、短 horizon、人工监督开始小步 fine-tune。

## 最重要的判断

这套代码里，算法部分可以复用；真正需要为真机补齐的是 environment adapter。

一句话总结：

```text
不要求真机和模拟器长得完全一样；
但必须让 policy 看到的 observation 和输出的 action 在语义上与训练时完全一样。
value 由算法自己学；
reward / done / reset / safety 必须在真机环境中自己实现。
```
