# rfpo_new_method

Isaac Lab training code for reproducing the Go2 / G1 main locomotion experiments
(baseline, reflow, and stacked / teacher-KD variants). Extracted from a local
`fgo_test` working tree; deployment and manipulation code are not included.

## Layout

- `isaaclab_experiments/` — FPO++ / reflow training package and setup scripts

## Setup

```bash
git submodule update --init --recursive
cd isaaclab_experiments
bash setup_env.sh
source source_env.sh
```

## Train

```bash
cd isaaclab_experiments
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```

Task hyperparameters live in `isaaclab_fpo/isaaclab_fpo/task_cfgs.py`.

## Eval (main-table protocol)

Use `eval_sampling_steps.py` with steps `64 32 16 8 4 1` and noise modes
`zero` / `random`. See `isaaclab_experiments/README.md` and package scripts for
details.
