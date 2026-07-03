# Vividex/RFPO 两个本体实验总结

日期：2026-06-29

本文总结当前工作区里围绕两个机器人本体的一系列实验：

- 原 benchmark 本体：`allegro_hand_ur5`，也就是 UR5 + Allegro hand。
- 新本体：`rm75_inspire_right`，也就是 RM75 机械臂 + Inspire/RH56 右手。当前正式线使用 native 12 手指自由度，action_dim=18，粉指动作禁用/忽略，URDF 主要是 `assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_nocyl.urdf`。

## 总结结论

1. 原 benchmark 本体现在已经补齐了同口径成对实验。使用同一个 PPO teacher 采集的 20 条成功示教、固定 stage2、deterministic、n=100 评测，`BC+RFPO/FPO` 比 `BC+PPO` 更好：RFPO best `reward=82.79, SR10=1.00, obj_err=0.0011`，保守 PPO best `reward=79.63, SR10=1.00, obj_err=0.0032`；同样看 200k 级别在线 checkpoint，RFPO `reward=77.23, SR10=0.97`，保守 PPO last `reward=73.54, SR10=0.99`。默认 PPO 训到 500k 会明显退化到 `reward=49.78, SR10=0.87`。

2. RM75 不是完全不可达。手工/先验控制和 BC 预训练都能在 RM75 上做出比较稳定的抓取和上抬，stage2 通常能到 `0.6-0.7` 左右。这说明主要问题不是“机械臂 IK 完全不可解”，而是抓取包络、接触稳定性、轨迹分布和 RL 微调稳定性。

3. RM75 scratch RL 基本失败。2M 步纯 PPO 和无 BC 的 RFPO/FPO 都是 stage2 `SR10=0.00`。

4. 在 RM75 上，公平的比较应该是 `BC+PPO` vs `BC+RFPO/FPO`。当前固定 stage 的独立复评结果显示，`BC+RFPO/FPO` 的最后 checkpoint 比 `BC+PPO` 更稳：stage2 `SR10=0.74` vs `0.32`。如果只比较训练中途最优 checkpoint，二者差距较小：stage2 `0.64` vs `0.60`。

5. 因此当前可以写成：在两个本体上，当前实验都支持 `BC+RFPO/FPO pipeline` 优于 `BC+PPO`。但要加两个 caveat：第一，RFPO 长训 last 会 drift，实际比较应看 best/early-stop 或相同训练步的 checkpoint；第二，RM75 线上日志显示 online objective 含 PPO-style 项，后续若论文要写“纯 RFPO objective”，还需要再做 ablation。

## 你真正关心的主对比：BC+RFPO vs BC+PPO

目标命题是：在两个本体上，都用同类 BC 初始化，然后比较在线 RL 阶段，`BC+RFPO` 应该优于 `BC+PPO`。

| 本体 | 当前成对证据 | 是否支持 `BC+RFPO > BC+PPO` | 说明 |
|---|---|---|---|
| `allegro_hand_ur5` | 同一 teacher 数据、固定 stage2 n=100：PPO-BC 初始 `78.98`，Flow-BC/RFPO 初始 `82.77`；保守 PPO best `79.63`，RFPO best `82.79`；200k 级别 checkpoint RFPO `77.23`，保守 PPO last `73.54`。 | 支持。RFPO 的 reward 和目标精度更高；成功率两者接近天花板，所以主要看 reward/obj_err/稳定性。 | RFPO 500k last 会 drift 到 `65.15`，但仍高于默认 PPO 500k final `49.78`。实际应使用 best checkpoint 或 early-stop。 |
| `rm75_inspire_right` native12 | scratch PPO/RFPO 都失败；BC 后可出分。固定 stage n=50 复评中，`BC+RFPO/FPO last@2M` stage2 `SR10=0.74`，`BC+PPO final@2M` stage2 `SR10=0.32`；best checkpoint 则是 `0.64` vs `0.60`。 | 支持，尤其最后 checkpoint 稳定性。 | RM75 还没有达到 Allegro 的 100%；且当前 RFPO/FPO pipeline 需要继续确认纯 RFPO objective ablation。 |

所以目前可以诚实地写成：

> 在 Allegro 原本体和 RM75 native12 上，当前实验都支持 `BC+RFPO/FPO pipeline` 比 `BC+PPO` 更好；Allegro 主要体现在 reward/目标精度和 early-stop best，RM75 主要体现在固定 stage2 成功率和最后 checkpoint 稳定性。

## 指标说明

| 指标 | 含义 |
|---|---|
| `SR10` / `norm_success_10` | 成功率主指标。RM75 里是 contact-gated 的成功判断，通常要求抓住且达到 lift/target 条件。 |
| `SR3` | 更严格的目标距离成功率。很多 RM75 run 主要看 `SR10`。 |
| `contact_success` | 是否满足接触门槛。RM75 中非常关键，没有接触的“碰飞/假 lift”不算真正成功。 |
| `lift_success_5cm` / `lift_success_target` | 是否达到抬升阈值。 |
| `obj_lift` | 物体抬升高度，单位 m。 |
| `obj_err` / `obj_tgt_dist` | 物体到目标位置的距离，单位 m。 |
| `reward` | 当前 reward 设计下的平均 return。不同实验阶段 reward shape 不完全等价，主要用于同一批实验内部比较。 |

## 本体与配置对照

| 本体 | 路径/名称 | 主要特征 | 当前判断 |
|---|---|---|---|
| 原 benchmark | `allegro_hand_ur5` | UR5 + Allegro hand，benchmark 自带本体和轨迹分布。 | 最容易出分，已有 checkpoint 基本 100%。 |
| RM75 active6 | RM75 + Inspire hand active 6 控制 | 手自由度低，接触包络和独立手指控制不足。 | 多轮搜索后 contact-gated success 仍为 0，不建议作为主线。 |
| RM75 driven12 | RM75 + driven/mimic 12 控制 | 比 active6 更容易出现个别成功 case。 | 能抽到成功，但整体成功率低，不如 native12 稳。 |
| RM75 native12 | `rm75_inspire_right` + `rm75_inspire_hand_right_nocyl.urdf` | 6 维臂动作 + 12 维手动作，粉指禁用/忽略，物体 scale=0.6，接触门控成功指标。 | 当前最值得继续推进的 RM75 主线。 |

正式 RM75 RL run 里的关键设置：

- `robot_name: rm75_inspire_right`
- `object_scale: 0.6`
- `rm75_native_hand_control: true`
- `rm75_disable_pinky_action: true`
- `rm75_ignore_pinky_contact: true`
- `rm75_success_min_non_thumb_contacts: 2`
- `rm75_norm_success_lift_thresh: 0.08`
- `rm75_done_on_norm_success_10: true`
- BC anchor dataset: `.local_runs/rm75_native_formal_scale060_stage2_ref512_liftfilter_20260629_195200.npz`
- BC dataset summary：`obs_dim=364`，`action_dim=18`，`samples=54177`，episode best lift 平均 `0.0847 m`。

## 原 benchmark 本体结果

新增成对实验目录：`.local_runs/allegro_bc_compare_20260629_2303`。

实验设置：

- teacher checkpoint：`07_baselines_and_logs/ppo_baselines/ppo_mustard_baseline/restore_checkpoint.zip`
- robot/env：`allegro_hand_ur5`，`ycb-006_mustard_bottle-20200709-subject-01-20200709_143211`，`norm_traj=True`
- BC 数据：20 条 teacher success episode，1200 transitions
- 评测协议：固定 stage2，deterministic，n=100，`--no-render --seed 0`

核心固定评测结果：

| Eval | n | Reward | SR3 | SR10 | Obj err | Final lift | Max lift | Contact | Contact frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `eval_ppo_bc_initial_stage2_100ep` | 100 | 78.98 | 0.99 | 1.00 | 0.0040 | 0.1167 | 0.1169 | 3.99 | 47.0 |
| `eval_flow_bc_initial_stage2_100ep` | 100 | 82.77 | 1.00 | 1.00 | 0.0011 | 0.1170 | 0.1173 | 4.00 | 46.9 |
| `eval_bcppo_conservative_stage2_200k_best_stage2_100ep` | 100 | 79.63 | 1.00 | 1.00 | 0.0032 | 0.1164 | 0.1166 | 3.98 | 47.2 |
| `eval_bcppo_conservative_stage2_200k_last_stage2_100ep` | 100 | 73.54 | 0.99 | 0.99 | 0.0102 | 0.1123 | 0.1132 | 3.96 | 46.2 |
| `eval_bcrfpo_conservative_stage2_100k_stage2_100ep` | 100 | 78.81 | 0.97 | 0.97 | 0.0035 | 0.1161 | 0.1165 | 3.88 | 46.3 |
| `eval_bcrfpo_conservative_stage2_200k_stage2_100ep` | 100 | 77.23 | 0.97 | 0.97 | 0.0018 | 0.1120 | 0.1125 | 3.88 | 46.4 |
| `eval_bcrfpo_conservative_stage2_500k_best_stage2_100ep` | 100 | 82.79 | 1.00 | 1.00 | 0.0011 | 0.1171 | 0.1175 | 4.00 | 46.8 |
| `eval_bcrfpo_conservative_stage2_500k_last_stage2_100ep` | 100 | 65.15 | 0.91 | 0.91 | 0.0102 | 0.1017 | 0.1023 | 3.64 | 47.4 |
| `eval_bcppo_online_500k_stage2_100ep` | 100 | 49.78 | 0.52 | 0.87 | 0.0312 | 0.1226 | 0.1244 | 3.46 | 41.8 |

训练中 eval 曲线的关键信息：

| Run | Eval steps | Mean reward 序列 | 说明 |
|---|---:|---|---|
| 默认 `BC+PPO 500k` | 0, 50k, 100k, 150k, 200k, 250k, 300k, 350k, 400k, 450k, 500k | 80.5, 55.6, 47.7, 53.6, 42.7, 42.1, 51.0, 50.2, 52.6, 52.3, 50.4 | PPO 从强 BC 出发后很快破坏策略。 |
| 保守 `BC+PPO 200k` | 0, 50k, 100k, 150k, 200k | 78.8, 75.3, 75.1, 75.1, 74.9 | 低 lr、stage2 起跑、强 BC anchor 后仍缓慢退化。 |
| 保守 `BC+RFPO/FPO 500k` | 0, 50k, 100k, 150k, 200k, 250k, 300k, 350k, 400k, 450k, 500k | 82.8, 81.1, 81.0, 78.0, 79.6, 75.7, 71.5, 60.8, 69.0, 65.6, 64.4 | 前 20 万步明显优于 PPO；长训会 drift，所以 best/early-stop 很重要。 |

旧的 Allegro/old robot 结果仍可作为补充证据：

| 实验/来源 | 类型 | Episodes | Stage | SR10 | Reward | Obj lift | Obj err | Contact | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `.local_runs/allegro_ppo_mustard_baseline_eval_20ep_20260626_185226/summary.csv` | PPO checkpoint eval | 20 | 2 | 1.00 | 83.09 | 0.1168 | 0.0008 | 4.00 | 原 benchmark PPO 复评。 |
| `/home/robot/QianSiyuan/RFPO/qsy-test-video/allegro_ppo_mustard_highscore_20fps_20260626/summary.csv` | PPO highscore render | 5 | 2 | 1.00 | 82.38 | 0.1166 | 0.0009 | 4.00 | 正常速度渲染视频导出。 |
| `/home/robot/QianSiyuan/RFPO/ref/04_videos_old_robot/oldrobot_rfpo16_best5_20260528/summary.csv` | old robot RFPO examples | 2 | 2 | 1.00 | 73.65 | 0.1373 | N/A | 4.00 | 老 RFPO 成功样例，不是完整训练曲线。 |
| `/home/robot/QianSiyuan/RFPO/ref/04_videos_old_robot/current_best76_render/summary.txt` | old robot RFPO examples | 3 | 2 | 1.00 | 约 75 | 约 0.113 | 约 0.0035 | 4.00 | 成功渲染样例。 |

结论：原本体成功率接近天花板，所以不能只看 `SR10`。在同一 BC 数据、固定 stage2、n=100 下，`BC+RFPO/FPO` 的 best reward 和目标精度明显优于 `BC+PPO`，而 PPO 在线更新更容易把好 BC 策略拉坏。

## RM75 变体探索

| 变体/来源 | 类型 | Episodes | SR10 | Contact success | Lift5 | Obj lift | Obj err | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `/home/robot/QianSiyuan/RFPO/qsy-test-video/active6_mountfix_probe/summary.csv` | active6 probe | 4 | 0.00 | 0.00 | 0.00 | 0.0000 | N/A | active6 初始探测不出分。 |
| `.local_runs/rm75_active6_base*_*/**/summary.csv` | active6 XY/yaw sweep | 156 个 summary | 0.00 | 0.00 | 最高有假 lift | 不稳定 | N/A | 大范围移动底座后仍没有 contact-gated success。部分 lift 是碰飞/非抓取，不算成功。 |
| `/home/robot/QianSiyuan/RFPO/qsy-test-video/driven12_success_scan_20260626_224829/summary.csv` | driven12 scan | 120 | 0.04 | 0.04 | 0.14 | 0.0177 | 0.2582 | driven12 能抽到少量成功，但不稳定。 |
| `/home/robot/QianSiyuan/RFPO/qsy-test-video/driven12_success_cases/selected_manifest.csv` | driven12 selected cases | 5 success + 3 near miss | selected | selected | selected | 0.051-0.085 | N/A | 有成功样例，说明方向可行，但不是高成功率策略。 |
| `/home/robot/QianSiyuan/RFPO/qsy-test-video/rm75_allegro_like_stop_on_target_20260629_165010/summary.csv` | native12 scripted/prior feasibility | 5 | 1.00 | 1.00 | 1.00 | 0.0874 | 0.0288 | 工程先验/脚本可行性演示，不是最终 RL 结果。 |
| `/home/robot/QianSiyuan/RFPO/qsy-test-video/rm75_native_z119_slow_wait_20260628_0205/summary.csv` | native12 scripted slow wait | 3 | 1.00 | 1.00 | 1.00 | 0.1163 | 0.0009 | 说明 native12 通过合适轨迹能达到 Allegro 类似效果。 |

变体结论：

- active6 的主要问题不是简单 XY 底座偏移。底座搜索后依然没有稳定接触门控成功，说明 6 个手自由度很难形成 Allegro 那种拇指内侧 + 三指外侧的稳定包络。
- driven12 比 active6 强，可以出现成功 case，但总体 `SR10=0.04`，还不足以作为正式训练主线。
- native12 最有希望。它能通过手工先验或 BC 学到较稳定抓取，所以后续正式比较都转到 native12。

## RM75 BC 预训练与数据质量

| BC/数据来源 | Episodes | Stage | SR10 | Contact | Lift5 | Reward | Obj lift | Obj err | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `rm75_formal_stage2_ref128_bc_eval_stage1_no_prior_20260629_194200` | 100 | 1 | 0.91 | 0.91 | 0.91 | 164.37 | 0.0773 | 0.0571 | ref128 BC，stage1 较稳。 |
| `rm75_formal_stage2_ref128_bc_eval_stage2_no_prior_20260629_194300` | 100 | 2 | 0.63 | 0.63 | 0.63 | 50.95 | 0.0575 | 0.0868 | 同一 BC，stage2 中等。 |
| `rm75_formal_stage2_ref512_bc50k_eval_stage1_no_prior_20260629_200700` | 100 | 1 | 0.97 | 0.97 | 0.97 | 195.50 | 0.0828 | 0.0355 | ref512 lift-filter，stage1 很稳。 |
| `rm75_formal_stage2_ref512_bc50k_eval_stage2_no_prior_20260629_200800` | 100 | 2 | 0.62 | 0.62 | 0.62 | 61.08 | 0.0530 | 0.0919 | stage2 仍有掉分。 |
| `rm75_dense_flowbc_step40000_formal_eval_stage1_100ep_20260629_204300` | 100 | 1 | 0.88 | 0.88 | 0.88 | 159.02 | 0.0864 | 0.0579 | dense flow BC。 |
| `rm75_dense_flowbc_step40000_formal_eval_stage2_100ep_20260629_204300` | 100 | 2 | 0.68 | 0.68 | 0.68 | 71.44 | 0.0733 | 0.0844 | 当前 BC 类里 stage2 较好。 |
| `rm75_1024_flowbc_step50000_formal_eval_stage1_100ep_20260629_211500` | 100 | 1 | 0.91 | 0.91 | 0.91 | 163.93 | 0.0811 | 0.0510 | ref1024 seed42。 |
| `rm75_1024_flowbc_step50000_formal_eval_stage2_100ep_20260629_211500` | 100 | 2 | 0.67 | 0.67 | 0.68 | 67.23 | 0.0682 | 0.0885 | stage2 也在 0.67 左右。 |

BC 结论：

- RM75 的成功并不是从 RL scratch 学出来的，而是很依赖高质量 reference/BC 初始策略。
- BC 单独已经能在 stage2 达到 `0.62-0.68`，所以后续 RL 的目标应是保持抓取包络并提升目标跟踪，而不是破坏已有抓取。
- 这也解释了为什么纯 PPO/RFPO scratch 都不行：探索空间太大，稳定抓取接触太难随机撞出来。

## RM75 正式 RL 训练日志

这些结果来自 2M 步训练日志中的 eval 表格。注意：FPO/RFPO run 带 curriculum/rollback，训练日志最后一条不一定是 stage2。

| Run | 初始化 | Steps | 日志状态 | Stage | SR10 | Contact | Reward | Obj lift | Obj dist | 说明 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `.local_runs/rm75_native_ppo_baseline_2m_20260629_202300` | scratch | 2.0M | final | 2 | 0.00 | 0.00 | -31.6 | 0.0000 | N/A | 纯 PPO scratch 失败。 |
| `.local_runs/rm75_native_rfpo_scratch_nobc_noprior_curriculum_2m_20260629_210539` | scratch | 2.0M | final | 2 | 0.00 | 0.00 | 5.9 | 0.0000 | 0.116 | 无 BC 的 FPO/RFPO scratch 失败。 |
| `.local_runs/rm75_native_bcppo_anchor_ref512_lowstd_2m_20260629_205500` | BC PPO pretrain + anchor | 1.30M | best stage2 in log | 2 | 0.917 | 0.917 | 238.0 | 0.0774 | N/A | 训练中途非常高，但不稳定。 |
| `.local_runs/rm75_native_bcppo_anchor_ref512_lowstd_2m_20260629_205500` | BC PPO pretrain + anchor | 2.0M | final | 2 | 0.167 | 0.167 | -128.0 | 0.0144 | N/A | 继续训到 2M 后明显崩。 |
| `.local_runs/rm75_native_bcrfpo_anchor_ref512_lowstd_curriculum_2m_20260629_210539` | flow BC + BC anchor | 1.40M | best stage2 in log | 2 | 0.667 | 0.667 | 123.0 | 0.0555 | 0.149 | 日志 stage2 peak 不如 BC+PPO peak。 |
| `.local_runs/rm75_native_bcrfpo_anchor_ref512_lowstd_curriculum_2m_20260629_210539` | flow BC + BC anchor | 2.0M | final/rollback | 1 | 0.917 | 0.917 | 162.0 | 0.0765 | 0.0464 | 末尾 rollback 到 stage1，但策略没有像 PPO final 那样崩掉。 |

训练日志结论：

- scratch 训练没有价值：2M 步仍然抓不到。
- `BC+PPO` 能在中途 peak 到非常高，但最后 checkpoint 明显退化。
- `BC+RFPO/FPO` 日志里的 stage2 peak 没超过 PPO peak，但末尾保持性更好。
- 因为 curriculum 会改变 eval stage，最终比较不能只看训练日志最后一行，必须固定 stage 重评。

## RM75 固定 stage 独立复评

下面是更公平的比较：固定 stage1/stage2，deterministic，n=50。使用同一 RM75 native12 环境设置，分别加载对应 checkpoint。这组数值来自本轮固定 stage 独立复评记录；表内保留 checkpoint 路径，便于之后用 `tools/render_rfpo_rollouts.py --no-render` 复跑并落盘成 `summary.csv`。

| Policy/checkpoint | Stage1 SR10 | Stage1 reward | Stage1 lift | Stage1 dist | Stage2 SR10 | Stage2 reward | Stage2 lift | Stage2 dist | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `BC+PPO best-stage2@1.3M`，`rl_models_1300000_steps.zip` | 0.80 | 158.3 | 0.0671 | 0.0622 | 0.60 | 78.92 | 0.0533 | 0.0927 | 中途最优可用，但 stage2 固定复评只有 0.60。 |
| `BC+PPO final@2M`，`models/model.zip` | 0.02 | -236.1 | 0.0018 | 0.1782 | 0.32 | -41.52 | 0.0271 | 0.1466 | 最后模型退化明显。 |
| `BC+RFPO best-stage2@1.4M`，`fpo_step_1400832.pt` | 0.90 | 135.6 | 0.0762 | 0.0580 | 0.64 | -37.96 | 0.0535 | 0.0874 | 比 PPO best 固定 stage2 略好，但 reward 不高。 |
| `BC+RFPO last@2M`，`models/last.pt` | 0.84 | 113.5 | 0.0707 | 0.0720 | 0.74 | 24.07 | 0.0627 | 0.0826 | 当前 RM75 正式训练里最稳的结果。 |

固定复评结论：

- 如果比较最后模型：`BC+RFPO/FPO last` 明显优于 `BC+PPO final`，stage2 `0.74` vs `0.32`。
- 如果比较各自 best checkpoint：`BC+RFPO/FPO best` 略优于 `BC+PPO best`，stage2 `0.64` vs `0.60`，差距不大。
- `BC+PPO` 的训练日志 peak 很漂亮，但固定复评和最终稳定性不如 `BC+RFPO/FPO`。
- 因此当前可以说：在 RM75 native12 这条线上，RFPO/FPO pipeline 的稳定性更好；但还不能说已经达到 Allegro 的 `100%`，也不能说严格纯 RFPO objective 已经被单独验证。

## 为什么 RM75 比 Allegro 难

1. 接触几何不同。Allegro 成功时通常是拇指在内侧固定，另外三个手指在外侧固定，4 个接触点包络稳定。RM75 active6/早期 native 配置经常出现物体滑动、晃动、碰歪，说明手指包络和接触法向不如 Allegro 天然合适。

2. 自由度和控制分配不同。active6 自由度太少，难以独立对齐四指；driven12 能产生个别成功，但泛化差；native12 最接近可用，但 action space 变成 arm+hand 18 维，RL 从零探索很难。

3. 参考轨迹不再原生匹配。原 benchmark 的轨迹和 Allegro 手是同分布的；RM75 需要 retarget。我们看到过手腕翻转、斜向抬升、提前碰瓶子等问题，这些都说明 reference/IK/末端姿态和新本体并非天然一致。

4. RM75 成功依赖稳定接触门控。单纯把物体碰起来、碰飞、或者短暂 lift，并不会算 `norm_success_10`。active6 sweep 里一些 lift 数字很高，但 contact success 仍是 0，就是这个原因。

5. RL 微调容易破坏 BC 的抓取结构。BC 已经能达到 stage2 `0.6-0.7`，但 `BC+PPO final` 掉到 `0.32`，说明在线 RL 的探索/梯度会把稳定包络打散。`BC+RFPO/FPO` 在当前设置下更能保住 BC 初始化。

## 当前最稳定的 RM75 方案

当前推荐继续推进的 RM75 方案是：

- 本体：`rm75_inspire_right` native12，而不是 active6。
- URDF：`assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_nocyl.urdf`。
- 手策略：粉指禁用/忽略，主要用拇指 + 三指包络，对齐 Allegro 的成功模式。
- 物体/任务：formal run 使用 `object_scale: 0.6`，success lift target 为 `0.08 m`。
- 数据：使用 lift-filter 的 successful reference dataset 作为 BC anchor。
- 训练：先做 flow BC 或 PPO BC，再在线 RL。不要 scratch。
- 比较：主要看固定 stage eval，不只看训练日志里的 best/final。

## 对“RFPO 是否比 PPO nb”的当前回答

在 Allegro 原本体上：现在已经有同口径成对证据支持 `BC+RFPO/FPO > BC+PPO`。固定 stage2 n=100 下，RFPO best `reward=82.79, SR10=1.00, obj_err=0.0011`，强 PPO 对照 best `reward=79.63, SR10=1.00, obj_err=0.0032`；默认 PPO 500k final 是 `reward=49.78, SR10=0.87`。由于成功率接近天花板，Allegro 上主要看 reward、目标误差和在线微调是否破坏 BC。

在 RM75 native12 上：如果比较最后模型，`BC+RFPO/FPO` 明显好于 `BC+PPO`，stage2 `0.74` vs `0.32`。如果比较各自 best checkpoint，`BC+RFPO/FPO` 也略好，stage2 `0.64` vs `0.60`，但优势不算大。

最严谨的总说法是：

> 当前两个本体都支持“BC+RFPO/FPO pipeline 比 BC+PPO 更好”。Allegro 上优势体现在 reward/目标精度和 early-stop best；RM75 上优势体现在固定 stage2 成功率和最终稳定性。但 RM75 还没有像 Allegro 一样 100%，且纯 RFPO objective 仍应单独 ablation。

## 后续建议

1. 固定评测协议：每个 checkpoint 都跑 `stage1/stage2, deterministic, n>=100`，统一输出 `summary.csv`，避免训练日志被 curriculum rollback 误导。

2. 做严格 ablation：保持 BC dataset、seed、学习率、batch、env 完全一致，只切换 `BC+PPO` 和真正启用的 `BC+RFPO objective`。RM75 当前日志里的 `train/fpo_objective_coef=0` 需要先解释/修正。

3. 继续提升 RM75 到 100%：优先从 BC/reference 质量和接触包络入手，而不是盲目加 RL 步数。重点检查稳定接触帧、拇指-非拇指对向接触、物体 XY drift、手腕姿态是否在 lift 阶段保持一致。

4. 对齐 Allegro 成功模式：四指包络、接触点位置、目标物体尺寸、摩擦和接触刚度都应继续作为对标项。当前 evidence 显示物体大小不是唯一根因，但合适尺寸和接触参数会显著影响 BC 成功率。

5. 建议把下一轮目标设为：RM75 native12 `BC+RFPO/FPO last` 在 stage2 n=100 固定复评达到 `SR10 >= 0.85`，然后再冲 `0.95-1.00`。
