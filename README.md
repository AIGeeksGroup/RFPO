# rfpo_new_method

FPO++ experiment code (Isaac Lab locomotion + manipulation), based on the
local `fgo_test` tree with Go2/G1 method changes. Deployment packages are not
included.

## Layout

- `isaaclab_experiments/` — velocity-conditioned locomotion (Go2/G1/…)
- `manipulation_experiments/` — manipulation pretrain / fine-tune (FPO++)

## Setup

```bash
git submodule update --init --recursive

# Locomotion
cd isaaclab_experiments
bash setup_env.sh && source source_env.sh

# Manipulation
cd ../manipulation_experiments
bash setup_env.sh && source source_env.sh
```

## Train (locomotion)

```bash
cd isaaclab_experiments
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```

## Eval (locomotion main-table)

Use `isaaclab_fpo/scripts/eval_sampling_steps.py` with steps `64 32 16 8 4 1`
and modes `zero` / `random`.

## Manipulation

See `manipulation_experiments/README.md` (`pretrain_flow_bc.py`,
`finetune_online_rl.py`, `eval_checkpoint.py`).
