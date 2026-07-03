#!/usr/bin/env python3

"""Train PPO on the RM75/RH56 relocation simulation path."""

import argparse
import os
import time

import yaml
import _init_paths

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from algos import ActorCriticPolicy
from hand_imitation.utils.eval import EvalCallback
from hand_imitation.utils.rm75_train_util import make_rm75_env, make_rm75_eval_env
from hand_imitation.utils.util import FallbackCheckpoint, InfoCallback


DEFAULT_SEQ = "ycb-006_mustard_bottle-20200709-subject-01-20200709_143211"
DEFAULT_INFO_KEYWORDS = [
    "pregrasp_success",
    "pregrasp_steps",
    "imitate_steps",
    "hand_jpos_err",
    "obj_com_err",
    "obj_rot_err",
    "obj_lift",
    "contact_count",
    "stable_grasp_contact",
    "hand_mjpos_err",
    "stage",
    "control_error",
    "obj_tgt_dist",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-name", default=DEFAULT_SEQ)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-eval-envs", type=int, default=2)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-n-episodes", type=int, default=25)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--restore-checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-proc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-traj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--log-std-init", type=float, default=-1.60)
    parser.add_argument("--pregrasp-success-thresh", type=float, default=0.075)
    parser.add_argument("--obj-com-done-thresh", type=float, default=0.18)
    parser.add_argument("--no-contact-grace-steps", type=int, default=30)
    parser.add_argument("--stable-contact-bonus", type=float, default=0.75)
    parser.add_argument("--required-non-thumb-contacts", type=int, default=1)
    parser.add_argument("--contact-reward-scale", type=float, default=0.5)
    parser.add_argument("--no-contact-penalty", type=float, default=0.25)
    parser.add_argument("--object-reward-requires-contact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-reward-scale", type=float, default=10.0)
    parser.add_argument("--obj-err-scale", type=float, default=50.0)
    parser.add_argument("--lift-bonus-thresh", type=float, default=0.02)
    parser.add_argument("--lift-bonus-mag", type=float, default=2.5)
    return parser.parse_args()


def build_env_kwargs(args):
    reward_kwargs = {
        "pregrasp_success_thresh": args.pregrasp_success_thresh,
        "obj_com_done_thresh": args.obj_com_done_thresh,
        "no_contact_grace_steps": args.no_contact_grace_steps,
        "stable_contact_bonus": args.stable_contact_bonus,
        "required_non_thumb_contacts": args.required_non_thumb_contacts,
        "contact_reward_scale": args.contact_reward_scale,
        "no_contact_penalty": args.no_contact_penalty,
        "object_reward_requires_contact": args.object_reward_requires_contact,
        "object_reward_scale": args.object_reward_scale,
        "obj_err_scale": args.obj_err_scale,
        "lift_bonus_thresh": args.lift_bonus_thresh,
        "lift_bonus_mag": args.lift_bonus_mag,
    }
    return {
        "name": args.seq_name,
        "robot_name": "rm75_inspire_right",
        "norm_traj": args.norm_traj,
        "task_kwargs": {"action": "relocate", "reward_kwargs": reward_kwargs},
        "info_keywords": DEFAULT_INFO_KEYWORDS,
        "n_envs": args.n_envs,
        "vid_freq": None,
        "vid_length": 100,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    env_kwargs = build_env_kwargs(args)
    run_config = {"args": vars(args), "env": env_kwargs}
    with open(os.path.join(args.output_dir, "rm75_train_config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    n_steps = max(int(args.n_steps // args.n_envs), 1)
    eval_freq = max(int(args.eval_freq // args.n_envs), 1)
    save_freq = max(int(args.save_freq // args.n_envs), 1)
    restore_freq = max(int(args.restore_checkpoint_freq // args.n_envs), 1)

    env = make_rm75_env(multi_proc=args.multi_proc, is_eval=False, **env_kwargs)
    eval_env = make_rm75_eval_env(args.multi_proc, args.n_eval_envs, **env_kwargs)

    policy_kwargs = {
        "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
        "log_std_init": args.log_std_init,
    }
    if args.resume_model:
        model = PPO.load(args.resume_model, env)
        model._last_obs = None
    else:
        model = PPO(
            ActorCriticPolicy,
            env,
            verbose=1,
            tensorboard_log=os.path.join(args.output_dir, "tb_logs"),
            n_steps=n_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            clip_range=args.clip_range,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
        )

    train_logger = configure(os.path.join(args.output_dir, "logs"), ["stdout", "log"])
    model.set_logger(train_logger)

    callbacks = [
        InfoCallback(),
        EvalCallback(args.output_dir, eval_freq, eval_env, n_eval_episodes=args.eval_n_episodes),
        CheckpointCallback(save_freq=save_freq, save_path=os.path.join(args.output_dir, "logs"), name_prefix="rl_models"),
        FallbackCheckpoint(args.output_dir, restore_freq),
    ]

    start = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, reset_num_timesteps=args.resume_model is None)
    model.save(os.path.join(args.output_dir, "models", "last"))
    print(f"[rm75-train] done in {time.time() - start:.1f}s")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
