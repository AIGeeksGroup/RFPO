#!/usr/bin/env bash
set -euo pipefail

ROOT="${CUIHU_ROOT:-/share/project/liyanjun/rongshanyu}"
REPO="${ROOT}/vividex_sapien_server_upload_fpo_control_full"
LOG_ROOT="${ROOT}/logs"
DATA_ROOT="${ROOT}/data/rm75_rfpo_bc"
BC_OUT_ROOT="${ROOT}/results/rm75_flow_bc"
mkdir -p "${LOG_ROOT}" "${DATA_ROOT}" "${BC_OUT_ROOT}"

cd "${REPO}"
source "${REPO}/scripts/cuihu/common.sh"

export VIVIDEX_HEADLESS_NO_RENDER=1
export HYDRA_FULL_ERROR=0
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"

TAG="${RM75_RUN_TAG:-reference_bc_fpo2_v2_contactbc}"
RM75_TRAJ="norm_trajectories/rm75_inspire_right/${SEQ}.npz"
BC_DATASET="${RM75_BC_DATASET:-${DATA_ROOT}/mustard_rm75_reference_${TAG}.npz}"
BC_OUT_DIR="${RM75_BC_OUT_DIR:-${BC_OUT_ROOT}/mustard_${TAG}}"
BC_CKPT="${BC_OUT_DIR}/flow_bc_last.pt"

echo "[rm75-rfpo-refbc] retarget reference"
python tools/retarget_rm75_inspire_reference.py \
  --src "norm_trajectories/${SEQ}.npz" \
  --dst "${RM75_TRAJ}" \
  --overwrite

if [[ ! -f "${BC_DATASET}" || "${FORCE_COLLECT_BC:-0}" == "1" ]]; then
  echo "[rm75-rfpo-refbc] collect RM75 reference BC: ${BC_DATASET}"
  python tools/collect_rm75_reference_bc_dataset.py \
    --seq-name "${SEQ}" \
    --norm-traj \
    --stage 2 \
    --mode approach_lift \
    -n "${RM75_BC_TRAJS:-256}" \
    --max-attempts "${RM75_BC_MAX_ATTEMPTS:-1024}" \
    --filter-success \
    --min-contact-steps "${RM75_BC_MIN_CONTACT_STEPS:-8}" \
    --min-lift "${RM75_BC_MIN_LIFT:-0.0}" \
    --close-val "${RM75_BC_CLOSE_VAL:-0.92}" \
    --thumb-yaw "${RM75_BC_THUMB_YAW:-0.64}" \
    --thumb-pitch "${RM75_BC_THUMB_PITCH:-0.72}" \
    --finger-close "${RM75_BC_FINGER_CLOSE:-0.92}" \
    --approach-delta 0.0 -0.045 0.0 \
    --approach-steps 12 \
    --close-steps 18 \
    --hold-steps 8 \
    --lift-steps 35 \
    --lift-height 0.075 \
    --settle-steps 8 \
    -o "${BC_DATASET}"
else
  echo "[rm75-rfpo-refbc] reuse BC dataset: ${BC_DATASET}"
fi

if [[ ! -f "${BC_CKPT}" || "${FORCE_PRETRAIN_BC:-0}" == "1" ]]; then
  echo "[rm75-rfpo-refbc] Flow-BC pretrain: ${BC_OUT_DIR}"
  mkdir -p "${BC_OUT_DIR}"
  CUDA_VISIBLE_DEVICES="${BC_GPU:-0}" python tools/pretrain_fpo_bc.py \
    -d "${BC_DATASET}" \
    -o "${BC_OUT_DIR}" \
    --steps "${RM75_FLOW_BC_STEPS:-200000}" \
    --batch_size "${RM75_FLOW_BC_BATCH:-512}" \
    --learning_rate "${RM75_FLOW_BC_LR:-1.0e-4}" \
    --weight_decay 1.0e-6 \
    --save_freq "${RM75_FLOW_BC_SAVE_FREQ:-50000}" \
    --actor_hidden_dims "[512,512,512]" \
    --critic_hidden_dims "[512,512]" \
    --activation elu \
    --timestep_embed_dim 32 \
    --sampling_steps 10 \
    --n_samples_per_action 1 \
    --cfm_loss_use_huber \
    --cfm_loss_huber_delta 0.5
else
  echo "[rm75-rfpo-refbc] reuse Flow-BC checkpoint: ${BC_CKPT}"
fi

launch_rm75_rfpo_refbc() {
  local gpu="$1"
  local run="$2"
  local clip="$3"
  local anchor_coef="$4"
  local anchor_min="$5"
  local lr="$6"
  local actor_objective="$7"

  echo "[rm75-rfpo-refbc] launch ${run} on GPU${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash scripts/server/train_state_single.sh "${SEQ}" "${run}" \
      "agent=fpo" \
      "env.robot_name=rm75_inspire_right" \
      "env.norm_traj=True" \
      "+env.task_kwargs.rm75_arm_action_scale=0.205" \
      "+env.task_kwargs.rm75_hand_action_scale=1.0" \
      "+env.task_kwargs.rm75_hand_close_bias=0.010" \
      "+env.task_kwargs.rm75_hand_dynamic_close_bias=0.18" \
      "+env.task_kwargs.rm75_hand_dynamic_close_start=0.245" \
      "+env.task_kwargs.rm75_hand_dynamic_close_full=0.080" \
      "+env.task_kwargs.rm75_hand_dynamic_close_requires_contact=False" \
      "+env.task_kwargs.rm75_hand_dynamic_close_gate_mode=blend" \
      "+env.task_kwargs.rm75_hand_dynamic_close_warmup_steps=8" \
      "+env.task_kwargs.rm75_hand_dynamic_close_ramp_steps=18" \
      "+env.task_kwargs.rm75_hand_dynamic_close_alpha_smooth=0.50" \
      "+env.task_kwargs.rm75_hand_dynamic_close_group_mode=ratio" \
      "+env.task_kwargs.rm75_hand_dynamic_close_thumb_ratio=0.72" \
      "+env.task_kwargs.rm75_hand_dynamic_close_pinky_ratio=0.78" \
      "+env.task_kwargs.rm75_hand_precontact_close_cap=0.38" \
      "+env.task_kwargs.rm75_hand_precontact_thumb_cap=0.18" \
      "+env.task_kwargs.rm75_hand_precontact_pinky_cap=0.28" \
      "+env.task_kwargs.rm75_hand_close_bias_weights=[0.28,0.36,1.0,1.0,0.96,0.62]" \
      "env.info_keywords=[pregrasp_success,pregrasp_steps,imitate_steps,hand_jpos_err,obj_com_err,obj_rot_err,obj_lift,contact_count,stable_grasp_contact,rm75_grasp_score,fingertip_obj_dist,min_fingertip_obj_dist,palm_obj_dist,hand_close_fraction,hand_thumb_close_fraction,hand_main_close_fraction,hand_pinky_close_fraction,dynamic_close_alpha,contact_hold_steps,object_xy_drift,object_speed,hand_mjpos_err,stage,control_error,obj_tgt_dist]" \
      "+env.task_kwargs.reward_kwargs.pregrasp_success_thresh=0.075" \
      "+env.task_kwargs.reward_kwargs.no_contact_grace_steps=58" \
      "+env.task_kwargs.reward_kwargs.obj_com_done_thresh=0.24" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_contact=True" \
      "+env.task_kwargs.reward_kwargs.object_reward_requires_stable_contact=False" \
      "+env.task_kwargs.reward_kwargs.no_contact_penalty=0.08" \
      "+env.task_kwargs.reward_kwargs.finger_approach_reward_scale=1.4" \
      "+env.task_kwargs.reward_kwargs.finger_approach_scale=24.0" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_reward_scale=0.8" \
      "+env.task_kwargs.reward_kwargs.min_finger_approach_scale=16.0" \
      "+env.task_kwargs.reward_kwargs.palm_approach_reward_scale=0.8" \
      "+env.task_kwargs.reward_kwargs.palm_approach_scale=7.0" \
      "+env.task_kwargs.reward_kwargs.contact_reward_scale=0.65" \
      "+env.task_kwargs.reward_kwargs.thumb_contact_reward_scale=0.9" \
      "+env.task_kwargs.reward_kwargs.non_thumb_contact_reward_scale=0.65" \
      "+env.task_kwargs.reward_kwargs.stable_contact_bonus=2.0" \
      "+env.task_kwargs.reward_kwargs.contact_hold_bonus_scale=1.5" \
      "+env.task_kwargs.reward_kwargs.contact_hold_bonus_steps=6" \
      "+env.task_kwargs.reward_kwargs.required_non_thumb_contacts=1" \
      "env.task_kwargs.reward_kwargs.object_reward_scale=10.0" \
      "env.task_kwargs.reward_kwargs.obj_err_scale=50.0" \
      "+env.task_kwargs.reward_kwargs.hand_mimic_reward_scale=4.0" \
      "+env.task_kwargs.reward_kwargs.hand_close_reward_scale=0.8" \
      "+env.task_kwargs.reward_kwargs.hand_close_near_dist=0.112" \
      "+env.task_kwargs.reward_kwargs.hand_close_palm_near_dist=0.225" \
      "+env.task_kwargs.reward_kwargs.hand_close_min_finger_near_dist=0.118" \
      "+env.task_kwargs.reward_kwargs.hand_close_contact_only=False" \
      "+env.task_kwargs.reward_kwargs.hand_close_target=0.64" \
      "+env.task_kwargs.reward_kwargs.hand_open_penalty_scale=0.45" \
      "+env.task_kwargs.reward_kwargs.early_close_penalty_scale=0.75" \
      "+env.task_kwargs.reward_kwargs.early_close_palm_dist=0.215" \
      "+env.task_kwargs.reward_kwargs.early_close_min_finger_dist=0.125" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_reward_scale=0.8" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_penalty_scale=0.6" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_balance_penalty_scale=0.55" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_min_main_close=0.18" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_thumb_target_ratio=0.72" \
      "+env.task_kwargs.reward_kwargs.hand_synergy_pinky_target_ratio=0.78" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_free_thresh=0.014" \
      "+env.task_kwargs.reward_kwargs.object_xy_drift_penalty_scale=24.0" \
      "+env.task_kwargs.reward_kwargs.object_speed_penalty_scale=0.055" \
      "+env.task_kwargs.reward_kwargs.lift_reward_scale=58.0" \
      "+env.task_kwargs.reward_kwargs.lift_reward_cap=0.09" \
      "env.task_kwargs.reward_kwargs.lift_bonus_thresh=0.010" \
      "env.task_kwargs.reward_kwargs.lift_bonus_mag=7.0" \
      "agent.params.bc_checkpoint=${BC_CKPT}" \
      "agent.params.bc_anchor_dataset=${BC_DATASET}" \
      "agent.params.bc_anchor_coef=${anchor_coef}" \
      "agent.params.bc_anchor_min_coef=${anchor_min}" \
      "agent.params.bc_anchor_decay_steps=800000" \
      "agent.params.bc_anchor_batch_size=512" \
      "agent.params.action_anchor_coef=0.005" \
      "agent.params.action_anchor_batch_size=512" \
      "agent.params.actor_objective=${actor_objective}" \
      "agent.params.trust_region_mode=ppo" \
      "agent.params.gamma=0.999" \
      "agent.params.gae_lambda=0.99" \
      "agent.params.learning_rate=${lr}" \
      "agent.params.min_learning_rate=${lr}" \
      "agent.params.max_learning_rate=${lr}" \
      "agent.params.clip_range=${clip}" \
      "agent.params.target_kl=0.003" \
      "agent.params.max_clip_fraction=0.20" \
      "agent.params.log_ratio_scale=1.0" \
      "agent.params.average_cfm_loss_in_chunk=True" \
      "agent.params.cfm_diff_clip=2.0" \
      "agent.params.cfm_loss_clamp=4.0" \
      "agent.params.cfm_loss_use_huber=True" \
      "agent.params.cfm_loss_huber_delta=0.5" \
      "agent.params.cfm_loss_huber_style=fpo_control" \
      "agent.params.rollout_action_noise_std=0.006" \
      "agent.params.rollout_deterministic=False" \
      "agent.params.eval_deterministic=False" \
      "agent.params.n_steps=4096" \
      "agent.params.batch_size=512" \
      "agent.params.n_epochs=3" \
      "agent.params.n_samples_per_action=8" \
      "agent.params.sampling_steps=10" \
      "agent.params.actor_hidden_dims=[512,512,512]" \
      "agent.params.critic_hidden_dims=[512,512]" \
      "agent.params.activation=elu" \
      "agent.params.max_grad_norm=1.0" \
      "agent.params.actor_max_grad_norm=0.5" \
      "agent.params.critic_max_grad_norm=2.0" \
      "agent.params.best_metric=eval/mean_rm75_grasp_score" \
      "agent.params.degrade_metric=eval/mean_rm75_grasp_score" \
      "agent.params.save_best_model=True" \
      "agent.params.rollback_to_best_on_degrade=True" \
      "agent.params.rollback_patience=4" \
      "agent.params.degrade_threshold=5.0" \
      "n_envs=16" \
      "n_eval_envs=4" \
      "eval_n_episodes=25" \
      "total_timesteps=${RM75_RFPO_TIMESTEPS:-1200000}" \
      "eval_freq=50000" \
      "save_freq=100000" \
      "restore_checkpoint_freq=100000" \
      "wandb.group=rm75_rfpo_reference_bc_2gpu" \
      "wandb.sweep_name_prefix=${run}"
  ) > "${LOG_ROOT}/${run}.out" 2>&1 &
  echo "$!" > "${LOG_ROOT}/${run}.pid"
}

launch_rm75_rfpo_refbc 0 "rm75_rfpo_${TAG}_fpo_clip002_anchor005_gpu0" 0.02 0.05 0.01 2.0e-6 fpo
launch_rm75_rfpo_refbc 1 "rm75_rfpo_${TAG}_hybrid_clip003_anchor003_gpu1" 0.03 0.03 0.005 2.5e-6 hybrid_fpo

echo "[rm75-rfpo-refbc] launched jobs:"
jobs -l
wait
