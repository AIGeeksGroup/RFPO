# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

#########################
# Policy configurations #
#########################


@configclass
class FpoRslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = 1.0
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    actor_scale: float = 1.0
    """Scaling factor applied to actor network output."""

    actor_mlp_output_scale: float = 1.0
    """Scaling factor applied to actor MLP output."""

    actor_final_layer_weight_scale: float | None = None
    """Scaling factor applied to initial weights of actor's final layer. Default is None (no scaling).

    When set, multiplies the weights of the actor's final linear layer by this value during
    initialization. Can help stabilize training by reducing initial action magnitudes.
    """

    timestep_embed_dim: int = 8
    """Dimension of the timestep embedding for the flow network. Default is 8.

    When > 0, adds a learned embedding of the flow timestep t to the actor network input.
    This allows the network to condition its output on the current denoising step.
    All canonical locomotion baselines (Nov 2025) used timestep_embed_dim=8.
    Set to 0 to disable timestep conditioning.
    """

    training_sampling_steps: int | None = None
    """Override sampling_steps for training CFM loss computation. Default is None (use sampling_steps).

    When set, uses a different number of discretization steps for computing CFM training loss
    than for inference. This allows using fewer steps during training for efficiency while
    using more steps during inference for quality.
    """

    cfm_loss_t_inverse_cdf_beta: float = 1.0
    """Beta parameter for Beta(1, beta) distribution used in timestep sampling.

    Controls the distribution of timesteps t sampled during CFM training:
    - beta = 1.0: Uniform sampling (default)
    - beta > 1.0: Favors sampling timesteps near t=0 (closer to actions)
      - e.g., beta = 2.0 moderately emphasizes action refinement
      - e.g., beta = 3.0 matches DDPM schedule (standard in flow matching)
      - e.g., beta = 4.0 strongly emphasizes action reconstruction
    - beta < 1.0: Favors sampling timesteps near t=1 (closer to noise)
      - e.g., beta = 0.5 moderately emphasizes exploration phase
      - e.g., beta = 0.25 strongly emphasizes early flow matching

    Math: Given uniform u ~ U(0,1), timesteps are sampled as:
    t = 0.005 + 0.99 * (1 - (1-u)^(1/beta))

    This implements the inverse CDF of Beta(1, beta) distribution scaled to [0.005, 0.995].
    """

    sampling_steps: int = 64
    """Number of sampling steps for flow matching inference. Default is 64."""

    cfm_loss_reduction: Literal["mean", "sum", "sqrt"] = "sqrt"
    """Reduction method for CFM loss across action dimensions. Default is "sqrt".

    - "mean": Average loss across action dimensions (divides by action_dim)
    - "sum": Sum loss across action dimensions (no division)
    - "sqrt": Variance-preserving reduction (divides by sqrt(action_dim))

    The "sqrt" option provides variance-preserving scaling that maintains similar
    gradient magnitudes across robots with different action dimensions.
    Empirically outperforms both "mean" and "sum" across all tasks.
    """

    action_perturb_std: float = 0.02
    """Standard deviation of Gaussian noise added to actions during training.
    Perturbs actions with random noise, which can be interpreted as an entropy
    regularizer."""

    train_flow_x0_mode: Literal["random", "zero", "mix"] = "random"
    """How to initialize flow noise x0 during on-policy ``act()`` rollouts.

    - ``random``: x0 ~ N(0, I) (default; matches RF training assumption)
    - ``zero``: x0 = 0 (matches common PostEval ``zero`` mode)
    - ``mix``: per-env Bernoulli mix of random/zero using ``train_flow_x0_random_prob``
    """

    train_flow_x0_random_prob: float = 0.5
    """Probability of sampling random x0 when ``train_flow_x0_mode='mix'``."""

    adaptive_compute_enabled: bool = False
    """Enable step predictor head on the actor for state-adaptive compute."""

    adaptive_step_bins: list[int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.adaptive_step_bins is None:
            self.adaptive_step_bins = [1, 4, 8, 16, 64]


############################
# Algorithm configurations #
############################


@configclass
class FpoRslRlPpoAlgorithmCfg:  # Keeping name for backwards compatibility
    """Configuration for the FPO (Flow Policy Optimization) algorithm."""

    class_name: str = "FPO"
    """The algorithm class name. Default is FPO (Flow Policy Optimization)."""

    num_learning_epochs: int = 16
    """The number of learning epochs per update. Default is 16.

    Quadruped locomotion (Go2, A1, Anymal-B/C/D) uses 16. Humanoid locomotion (H1, G1)
    and Spot use 32. Override in per-robot configs as needed.
    """

    num_mini_batches: int = 4
    """The number of mini-batches per update. Default is 4."""

    learning_rate: float = 1e-4
    """The learning rate for the policy. Default is 1e-4."""

    weight_decay: float = 1e-4
    """Weight decay coefficient for AdamW optimizer. Default is 1e-4."""

    adam_betas: tuple[float, float] = (0.9, 0.999)
    """Beta parameters (beta1, beta2) for Adam/AdamW optimizer. Default is (0.9, 0.999).

    - beta1: Exponential decay rate for first moment estimates (momentum)
    - beta2: Exponential decay rate for second moment estimates (RMSprop-like)
    """

    schedule: str = "fixed"
    """The learning rate schedule. Default is 'fixed'.

    - 'fixed': Constant learning rate throughout training
    - 'adaptive': Adjusts learning rate based on KL divergence (requires desired_kl > 0)

    The canonical locomotion baseline (Nov 2025) uses 'fixed'.
    """

    gamma: float = 0.99
    """The discount factor. Default is 0.99."""

    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE). Default is 0.95."""

    knn_entropy_coef: float = 0.0
    """Coefficient for kNN entropy bonus in the policy loss."""

    knn_entropy_k: int = 1
    """Number of nearest neighbors for kNN entropy bonus in the policy loss. Default is 1.

    Separate from knn_k which is used for the entropy regularization term.
    This parameter controls the k used when computing an additional entropy-based
    exploration bonus. k=1 gives the sharpest entropy estimate and empirically
    outperforms higher k values for locomotion tasks.
    """

    desired_kl: float = 1e-4
    """The desired KL divergence. Default is 1e-4.

    When schedule='adaptive', the learning rate is adjusted to keep KL divergence
    near this target. Ignored when schedule='fixed' (the default).
    """

    max_grad_norm: float = 1.0
    """The maximum gradient norm. Default is 1.0."""

    value_loss_coef: float = 1.0
    """The coefficient for the value loss. Default is 1.0.

    Spot uses 0.5 (override in per-robot config). All other locomotion robots use 1.0.
    """

    use_clipped_value_loss: bool = False
    """Whether to use clipped value loss. Default is False.

    PPO typically clips the value function loss, but with FPO's tighter policy clipping,
    value clipping adds instability.
    """

    clip_param: float = 0.05
    """The clipping parameter for the policy. Default is 0.05.

    FPO needs much tighter clipping than standard PPO (0.2). Locomotion uses 0.05.
    Also used as the SPO epsilon in 'spo' and 'aspo' modes.
    """

    trust_region_mode: Literal["ppo", "spo", "aspo"] = "aspo"
    """Trust region method to use. Default is 'aspo'.

    - 'ppo': Standard PPO with hard clipping constraint
    - 'spo': Structured Policy Optimization with quadratic penalty
    - 'aspo': Asymmetric SPO - uses PPO for positive advantages, SPO for negative advantages

    SPO uses a smoother trust region constraint:
    policy_loss = -mean(ratio * advantage - |advantage| / (2*epsilon) * (ratio - 1)^2)

    This provides more gradual policy updates compared to PPO's hard clipping.
    """

    normalize_advantage: bool = True
    """Whether to normalize advantages at all. Default is True.

    If False, advantages are used as-is without any normalization.
    This can be useful when the agent is near optimal and normalization
    artificially amplifies small differences in returns.
    """

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    Note: This only applies if normalize_advantage is True.
    """

    advantage_clamp: tuple[float, float] = (100.0, 100.0)
    """Symmetric clamp bounds for advantages as (positive_max, negative_max).

    Clamps advantages to [-negative_max, positive_max] before using them in the policy loss.
    Prevents large advantages from causing unstable updates."""

    n_samples_per_action: int = 16
    """Number of samples per action for CFM loss computation. Default is 16.

    The canonical locomotion baseline (Nov 2025, run 03yjn5lj) uses 16.
    Gains plateau beyond 16.
    """

    cfm_diff_clamp_max: float = 10.0
    """Upper bound for CFM loss difference clamping. Default is 10.0.

    The loss difference (old_cfm_loss - new_cfm_loss) is clamped to this upper bound
    using straight-through estimator (STE) before exp().
    """

    cfm_loss_clamp: float = 20.0
    """Maximum value to clamp CFM losses (both old and current). Default is -1.0 (disabled).

    When > 0, clamps both the old (stored) and current (recomputed) CFM loss values
    to this upper bound. This prevents extremely large losses from producing extreme ratios
    that destabilize training. Applied symmetrically to both old and current CFM losses.
    """

    cfm_loss_clamp_negative_advantages: bool = True
    """Clamp current CFM loss when advantage is negative. Default is True.

    When enabled, clamps the current (recomputed) CFM loss to cfm_loss_clamp_negative_advantages_max
    for transitions where the advantage is negative (bad actions). This prevents the policy from
    being destabilized by extreme ratios when aggressively avoiding bad actions.
    Critical for training stability with 32+ learning epochs (H1, G1, Spot).
    """

    cfm_loss_clamp_negative_advantages_max: float = 20.0
    """Maximum CFM loss value for negative advantage clamping.

    Only used when cfm_loss_clamp_negative_advantages is True. The current CFM loss is clamped
    to this value for transitions with negative advantages.
    """

    storage_action_noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to stored actions. Default is 0.0 (no noise).

    This noise is added to actions before storing them in the rollout buffer, affecting
    the CFM loss computation in the PPO ratio. Acts as implicit entropy regularization
    by forcing the policy to be robust to action perturbations. Unlike action_perturb_std
    in the actor, this noise affects the policy gradient computation.

    Typical values: 0.01-0.05 depending on action scale and desired regularization strength.
    """

    ema_decay: float = 0.95
    """EMA decay rate for exponential moving average of flow model weights. Default is 0.95.

    When > 0, maintains a smoothed copy of the actor (flow model) parameters that is updated
    after each PPO update. The EMA weights are saved in checkpoints and typically produce better
    samples than the currently-being-trained weights.

    The canonical locomotion baseline (Nov 2025, run 03yjn5lj) uses 0.95.

    Recommended values:
    - 0.0: Disabled (no EMA)
    - 0.95: Fast adaptation, suitable for short training runs (~1500-2000 steps)
    - 0.99: Slower adaptation, suitable for longer training runs
    - 0.999: Very slow adaptation, only for very long training runs

    The effective averaging window is approximately 1/(1-decay) updates.
    """

    ema_warmup_steps: int = 500
    """Number of PPO updates before starting EMA. Default is 500.

    Prevents early noisy weights from contaminating the EMA by waiting for the policy
    to stabilize before starting exponential averaging. Only applies when ema_decay > 0.
    Set to 0 to start EMA immediately from the first update.
    """

    reflow_enabled: bool = False
    """Enable Rectified Flow reflow auxiliary loss (Liu et al., 2022).

    When True, adds a straight-path CFM loss on coupled (noise, flow-generated action) pairs.
    This encourages straighter transport and can reduce required inference steps. Default is False.
    """

    reflow_loss_coef: float = 1.0
    """Coefficient for the reflow auxiliary loss when reflow_enabled is True."""

    reflow_n_samples_per_obs: int = 4
    """Number of reflow (noise, endpoint) pairs sampled per observation in each mini-batch."""

    reflow_mode: Literal["uniform", "reward_aware", "fpo_operator"] = "uniform"
    """How to weight reflow samples: uniform, advantage-weighted, or FPO++ operator-style."""

    reflow_advantage_threshold: float = 0.0
    """Minimum advantage for reward-aware reflow weighting."""

    reflow_use_ema_endpoint: bool = False
    """Use EMA actor weights (detached) to generate reflow endpoints (policy improvement operator)."""

    reflow_adaptive_lambda_enabled: bool = False
    """Adapt ``reflow_loss_coef`` from the 1-vs-N discretization gap (random x0).

    Collection protocol stays ``train_flow_x0_mode=random`` (same as baseline).
    Large gap → λ near ``reflow_lambda_max``; already-straight → shrink toward min
    so full-step / random@N return is not taxed forever.
    """

    reflow_lambda_min: float = 0.1
    """Floor for adaptive reflow λ."""

    reflow_lambda_max: float = 1.0
    """Ceiling for adaptive reflow λ (same default as plain reflow)."""

    reflow_lambda_gap_low: float = 0.05
    """1-step RMS gap below this maps to ``reflow_lambda_min``."""

    reflow_lambda_gap_high: float = 0.4
    """1-step RMS gap above this maps to ``reflow_lambda_max``."""

    reflow_lambda_ema: float = 0.9
    """EMA on the adaptive λ across PPO updates (0 = no smoothing)."""

    reflow_lambda_warmup_iters: int = 50
    """Keep λ at ``reflow_lambda_max`` for this many PPO updates.

    Untrained 1-step and N-step actions can look similar (small gap) even when
    the path is not straight; adapting immediately would collapse λ to the floor.
    """

    theory_metrics_enabled: bool = False
    """Log path straightness and discretization error proxies during training."""

    adaptive_compute_enabled: bool = False
    """Train a step predictor for state-adaptive integration budgets."""

    adaptive_compute_loss_coef: float = 0.1
    """Overall coefficient for adaptive compute loss."""

    adaptive_latency_penalty_coef: float = 1.0
    """Trade-off between action fidelity and predicted integration steps inside adaptive loss."""

    adaptive_step_bins: list[int] = None  # type: ignore[assignment]
    """Discrete integration budgets for the step predictor. Default set in __post_init__."""

    random_x0_consistency_enabled: bool = False
    """Auxiliary loss: match few-step flow actions to full-step actions under shared random x0.

    Targets PostEval ``random`` robustness (RF assumes x0 ~ N(0,I)) while keeping
    few-step fidelity. Intended to stack on plain reflow without other idea heads.
    """

    random_x0_consistency_coef: float = 0.1
    """Coefficient for ``random_x0_consistency`` when enabled."""

    random_x0_consistency_steps: list[int] = None  # type: ignore[assignment]
    """Few-step budgets compared against full ``sampling_steps`` under the same x0."""

    reflow_teacher_checkpoint: str = ""
    """Frozen baseline checkpoint whose 64-step actor supplies reflow endpoints / KD targets.

    Empty disables the frozen teacher. Collection stays ``train_flow_x0_mode=random``.
    """

    teacher_kd_enabled: bool = False
    """Distill student few-step Euler onto stopgrad teacher 64-step actions (same x0)."""

    teacher_kd_coef: float = 0.1
    """Coefficient for ``teacher_kd`` when enabled."""

    teacher_kd_steps: list[int] = None  # type: ignore[assignment]
    """Student Euler budgets matched to the teacher 64-step action."""

    teacher_aux_zero_x0_prob: float = 0.25
    """Fraction of auxiliary (reflow/KD) samples that use x0=0 (deploy protocol).

    Does not change on-policy collection, which stays fully random.
    """

    def __post_init__(self):
        if self.adaptive_step_bins is None:
            self.adaptive_step_bins = [1, 4, 8, 16, 64]
        if self.random_x0_consistency_steps is None:
            self.random_x0_consistency_steps = [1, 4, 8]
        if self.teacher_kd_steps is None:
            self.teacher_kd_steps = [1, 4, 8]


#########################
# Runner configurations #
#########################


@configclass
class FpoRslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = 24
    """The number of steps per environment per update. Default is 24."""

    max_iterations: int = 1500
    """The maximum number of iterations. Default is 1500.

    Quadrupeds (Go2, A1, Spot, Anymal-B/C/D) use 1500. Humanoids (H1, G1) use 2000.
    """

    empirical_normalization: bool = True
    """Whether to use empirical normalization. Default is True."""

    randomize_reset_episode_progress: float = 0.0
    """Randomize episode progress on reset to prevent synchronization. Default is 0.0 (disabled).

    When > 0, environments that reset will have their episode_length_buf randomized to a value
    between 0 and randomize_reset_episode_progress * max_episode_length. For example, 0.25 means
    episodes will start at a random point between 0-25% completion.
    """

    policy: FpoRslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: FpoRslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = 2.0
    """The clipping value for actions. If ``None``, then no clipping is done.
    Default is 2.0.

    .. note::
        This clipping is performed inside the :class:`FpoRslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = 50
    """The number of iterations between saves. Default is 50."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    # Evaluation configuration
    eval_episodes: int = 10
    """Number of episodes to run per evaluation mode. Default is 10."""

    flow_eval_modes: list[str] = ["zero", "random"]
    """Evaluation modes for flow matching deterministic sampling. Default is ["zero", "random"].

    Available modes:
    - "zero": Use zeros for initial noise
    - "fixed_seed": Use fixed random seed for reproducible noise
    - "random": Use random noise (different each time)
    """

    flow_eval_fixed_seed: int = 12345
    """Random seed for fixed_seed evaluation mode. Default is 12345."""

    enable_post_training_eval: bool = True
    """Whether to evaluate all checkpoints after training completes. Default is True.

    When enabled, automatically evaluates all saved checkpoints using the same eval configuration
    (flow_eval_modes, eval_episodes) after training finishes. Results are logged to WandB with
    a custom 'eval_iteration' step metric for comparison with training metrics.
    """

    post_eval_checkpoint_interval: int = 1
    """Evaluate every Nth checkpoint during post-training evaluation. Default is 1 (all checkpoints).

    Set to 2 to evaluate every other checkpoint, 3 for every third, etc. This can significantly
    reduce post-training evaluation time for experiments with many checkpoints.
    """

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
