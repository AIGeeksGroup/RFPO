# ViViDex Server Runbook

这份文档对应本仓库的 `state-based PPO baseline` 服务器训练流程，写法遵循桌面的“本地项目通过 VSCode 远程服务器运行指南”。

目标是把服务器流程拆成三步：

1. 登录节点 `preflight`
2. 短交互式 GPU `smoke test`
3. `sbatch` 正式训练

## 0. 推荐服务器目录

假设你的服务器正式存储根目录是：

```bash
/data/group/<lab>/$USER
```

建议至少准备这些目录：

```bash
mkdir -p /data/group/<lab>/$USER/{projects,envs,cache,tmp,logs,results}
```

然后把仓库放到：

```bash
/data/group/<lab>/$USER/projects/vividex_sapien
```

## 1. 建环境

第一次在服务器上执行：

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien
bash scripts/server/setup_conda_env.sh /data/group/<lab>/$USER/envs/vividex
```

这一步会：

- 建立 conda 环境
- 安装 README 里要求的 PyTorch CUDA 版本
- 把 `mkl / intel-openmp` 固定到与本地验证一致的版本，避免 `torch` 出现 `undefined symbol: iJIT_NotifyEvent`
- 使用显式 channel 安装核心框架，尽量避免服务器默认 `conda-forge` 配置把求解结果带偏
- 安装 `requirements.txt`
- 做一次最基础 import 检查

建议在执行前先看服务器当前 conda 配置：

```bash
which conda
conda info
conda config --show channels
conda env list
```

如果服务器默认 channel 被改过，优先相信这一步输出，不要假设它和你本地一样。

## 2. 每次新开终端先设置这两个变量

```bash
export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex
```

可选：

```bash
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU
export WANDB_MODE=offline
export SLURM_PARTITION=day
export SLURM_TIME=23:59:00
export SLURM_CPUS=8
export SLURM_MEM=32G
```

## 3. 登录节点静态检查

先不要直接训练，先做静态检查：

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien
bash scripts/server/preflight.sh ycb-006_mustard_bottle-20200709-subject-01-20200709_143211
```

这一步会检查：

- conda 环境能否激活
- `torch / hydra / stable_baselines3 / wandb / sapien` 是否能 import
- `tools/train.py --help` 能否正常启动
- 指定的序列文件是否存在

如果这一步不过，不要继续排 GPU 作业。

常见情况：

- 如果报 `import failed for hydra`、`stable_baselines3`、`sapien` 之类的错误，说明目标 conda 环境没有装完整，先回到第 1 步重新执行 `setup_conda_env.sh`。
- 如果 `tools/train.py --help` 失败，通常也是环境缺包，不是训练脚本本身有问题。
- 如果报 `EnvironmentLocationNotFound`，说明 `VIVIDEX_CONDA_ENV` 指向的环境根本不存在。
- 如果报 `MKL_INTERFACE_LAYER: unbound variable`，说明环境激活脚本和当前 shell 的 `set -u` 冲突，先补 `MKL_INTERFACE_LAYER` 和 `MKL_THREADING_LAYER`。
- 如果报 `LibMambaUnsatisfiableError`，优先怀疑服务器 conda channel 和包求解，不要先怀疑项目逻辑。

## 4. 短交互式 GPU 验证

先申请短作业：

```bash
srun -p short -t 0:30:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
```

进入计算节点后：

```bash
hostname
nvidia-smi
cd /data/group/<lab>/$USER/projects/vividex_sapien
export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex
bash scripts/server/smoke_test.sh ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 smoke_mustard
```

`smoke_test.sh` 会做两件事：

1. 构造环境、`reset`、`step` 一次  
2. 跑一个超小的 PPO 训练 smoke run

只有这一步通过后，才提交正式训练。

## 5. 单任务正式训练

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien
export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex
bash scripts/server/submit_state_job.sh \
  ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 \
  ppo_mustard_baseline \
  total_timesteps=2000000
```

这条命令会：

- 提交到单卡作业
- 自动把日志写到 `${VIVIDEX_RUNTIME_ROOT}/logs`
- 自动把训练结果写到 `${VIVIDEX_RUNTIME_ROOT}/results/state_baseline/<run_name>`

## 6. 一次提交 3 个 Phase 1 baseline

仓库已经内置了一个 3 任务列表：

- mustard bottle
- sugar box
- extra large clamp

直接提交：

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien
export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex
bash scripts/server/submit_phase1_baselines.sh scripts/server/phase1_tasks.tsv total_timesteps=2000000
```

## 7. 如何监控

查看队列：

```bash
squeue -u $USER
```

查看历史状态：

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,Nodelist
```

查看日志：

```bash
tail -f /data/group/<lab>/$USER/results/vividex/logs/<run_name>_<jobid>.out
```

如果作业 `RUNNING` 但没输出：

```bash
scontrol show job <jobid>
sstat -j <jobid>.batch --format=JobID,AveCPU,MaxRSS,MaxVMSize
```

## 8. 你现在最该执行的顺序

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien

export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex

bash scripts/server/preflight.sh ycb-006_mustard_bottle-20200709-subject-01-20200709_143211
```

通过后：

```bash
srun -p short -t 0:30:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
```

进入节点后：

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien
export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex
bash scripts/server/smoke_test.sh ycb-006_mustard_bottle-20200709-subject-01-20200709_143211 smoke_mustard
```

通过后正式提交：

```bash
cd /data/group/<lab>/$USER/projects/vividex_sapien
export VIVIDEX_CONDA_ENV=/data/group/<lab>/$USER/envs/vividex
export VIVIDEX_RUNTIME_ROOT=/data/group/<lab>/$USER/results/vividex
bash scripts/server/submit_phase1_baselines.sh scripts/server/phase1_tasks.tsv total_timesteps=2000000
```

## 9. 当前这套脚本的边界

这套脚本解决的是：

- baseline PPO state policy 训练
- 服务器环境搭建
- 静态检查
- 交互式 smoke test
- 单卡 A100 正式训练

它还没有做：

- FPO/FPO++ 训练
- visual policy 训练
- 多卡训练
- 下游 rollout 数据导出 pipeline

先把这一版 baseline 跑通，再进入 FPO/FPO++ 改造。

## 10. 这次服务器踩坑得到的通用结论

- 本地 smoke test 通过，只能证明代码主路径正确，不能证明服务器环境也一定正确。
- 服务器第一次部署时，应该先查 `conda`、channel、环境路径，再查项目依赖，再查训练代码。
- 登录节点只做 `preflight` 这类静态检查；GPU 相关问题要进短交互式作业里确认。
- 长任务提交前，一定先有一次可复现的短 GPU smoke run。


## RM75 + Inspire/RH56 Simulation Port

This fork keeps `allegro_hand_ur5` as the default robot.  To run the real-hardware-matched robot in simulation, use `env.robot_name=rm75_inspire_right`.

The RM75/RH56 robot does not use the original Allegro reference file directly.  Generate a robot-specific reference first:

```bash
python tools/retarget_rm75_inspire_reference.py \
  --src norm_trajectories/ycb-006_mustard_bottle-20200709-subject-01-20200709_143211.npz \
  --dst norm_trajectories/rm75_inspire_right/ycb-006_mustard_bottle-20200709-subject-01-20200709_143211.npz \
  --overwrite
```

Then run the server smoke test:

```bash
sbatch scripts/server/rm75_smoke.slurm
```

For a first mustard-bottle training run:

```bash
sbatch scripts/server/rm75_train_mustard.slurm
```

Important: the current retargeter is shape-compatible, not a calibrated real-robot retargeter.  It creates 13-DoF RM75/RH56 qpos and palm+five-fingertip references so the environment and reward no longer silently consume Allegro's 22-DoF, four-finger data.  To reach paper-level sim-to-real quality, replace this with calibrated flange mounting and IK retargeting for the RM75 arm and RH56 hand.
