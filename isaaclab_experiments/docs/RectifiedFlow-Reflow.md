# Rectified Flow Reflow（索引）

- **Method（中文，论文写法 / 算法过程）：** [FPO-Reflow-Ideas-Method.md](./FPO-Reflow-Ideas-Method.md)
- **实现说明（旧版、偏工程）：** [RectifiedFlow-Reflow-实现说明.md](./RectifiedFlow-Reflow-实现说明.md)

---

## 一句话

在 FPO 更新时加 **Reflow 辅助损失**：用当前 flow 生成 `(噪声, 动作)` 耦合对，在直线上训练速度场，使传输路径更直；可选 advantage 加权、EMA 端点、自适应步数头、random-\(x_0\) 一致性。

## 快速开启

```bash
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless \
  --fpo_variant reflow \
  --run_name reflow_v1
```

## 代码入口

- `modules/actor_critic.py` → `get_reflow_loss` / adaptive / rx0
- `modules/reflow_extensions.py`
- `algorithms/fpo.py` → `FPO.update()`
- `task_cfgs.py` → `--fpo_variant`
