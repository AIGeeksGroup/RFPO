from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FPOTransition:
    obs: torch.Tensor
    actions: torch.Tensor
    flow_actions: torch.Tensor | None
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    old_cfm_loss: torch.Tensor
    old_x1_pred: torch.Tensor
    cfm_eps: torch.Tensor
    cfm_t: torch.Tensor
    env_rewards: torch.Tensor | None = None
    old_log_prob: torch.Tensor | None = None
    policy_actions: torch.Tensor | None = None
    obj_com_err: torch.Tensor | None = None
    hand_mjpos_err: torch.Tensor | None = None
    control_error: torch.Tensor | None = None
    stage: torch.Tensor | None = None


@dataclass
class FPOMiniBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    policy_actions: torch.Tensor
    flow_actions: torch.Tensor
    obj_com_err: torch.Tensor
    hand_mjpos_err: torch.Tensor
    control_error: torch.Tensor
    env_rewards: torch.Tensor
    stage: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_cfm_loss: torch.Tensor
    old_x1_pred: torch.Tensor
    cfm_eps: torch.Tensor
    cfm_t: torch.Tensor
    old_log_prob: torch.Tensor


@dataclass
class FPOChunkMiniBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    policy_actions: torch.Tensor
    flow_actions: torch.Tensor
    obj_com_err: torch.Tensor
    hand_mjpos_err: torch.Tensor
    control_error: torch.Tensor
    env_rewards: torch.Tensor
    stage: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_cfm_loss: torch.Tensor
    old_x1_pred: torch.Tensor
    cfm_eps: torch.Tensor
    cfm_t: torch.Tensor
    valid_mask: torch.Tensor
    old_log_prob: torch.Tensor


class FPORolloutBuffer:
    def __init__(
        self,
        *,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        n_cfm_samples: int,
        device: str | torch.device,
    ):
        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_cfm_samples = int(n_cfm_samples)
        self.device = torch.device(device)
        self.step = 0

        self.obs = torch.zeros(self.num_steps, self.num_envs, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.num_steps, self.num_envs, self.action_dim, device=self.device)
        self.policy_actions = torch.zeros(self.num_steps, self.num_envs, self.action_dim, device=self.device)
        self.flow_actions = torch.zeros(self.num_steps, self.num_envs, self.action_dim, device=self.device)
        self.obj_com_err = torch.full((self.num_steps, self.num_envs, 1), float("inf"), device=self.device)
        self.hand_mjpos_err = torch.full((self.num_steps, self.num_envs, 1), float("inf"), device=self.device)
        self.control_error = torch.full((self.num_steps, self.num_envs, 1), float("inf"), device=self.device)
        self.stage = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.rewards = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.env_rewards = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.dones = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.values = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.returns = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)
        self.old_cfm_loss = torch.zeros(
            self.num_steps,
            self.num_envs,
            self.n_cfm_samples,
            device=self.device,
        )
        self.old_x1_pred = torch.zeros(
            self.num_steps,
            self.num_envs,
            self.n_cfm_samples,
            self.action_dim,
            device=self.device,
        )
        self.cfm_eps = torch.zeros(
            self.num_steps,
            self.num_envs,
            self.n_cfm_samples,
            self.action_dim,
            device=self.device,
        )
        self.cfm_t = torch.zeros(
            self.num_steps,
            self.num_envs,
            self.n_cfm_samples,
            1,
            device=self.device,
        )
        self.old_log_prob = torch.zeros(self.num_steps, self.num_envs, 1, device=self.device)

    def clear(self) -> None:
        self.step = 0

    def add(self, transition: FPOTransition) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("FPO rollout buffer is full")
        self.obs[self.step].copy_(transition.obs)
        self.actions[self.step].copy_(transition.actions)
        policy_actions = transition.actions if transition.policy_actions is None else transition.policy_actions
        self.policy_actions[self.step].copy_(policy_actions)
        flow_actions = transition.actions if transition.flow_actions is None else transition.flow_actions
        self.flow_actions[self.step].copy_(flow_actions)
        if transition.obj_com_err is not None:
            self.obj_com_err[self.step].copy_(transition.obj_com_err.view(self.num_envs, 1))
        else:
            self.obj_com_err[self.step].fill_(float("inf"))
        if transition.hand_mjpos_err is not None:
            self.hand_mjpos_err[self.step].copy_(transition.hand_mjpos_err.view(self.num_envs, 1))
        else:
            self.hand_mjpos_err[self.step].fill_(float("inf"))
        if transition.control_error is not None:
            self.control_error[self.step].copy_(transition.control_error.view(self.num_envs, 1))
        else:
            self.control_error[self.step].fill_(float("inf"))
        if transition.stage is not None:
            self.stage[self.step].copy_(transition.stage.view(self.num_envs, 1))
        else:
            self.stage[self.step].zero_()
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        env_rewards = transition.rewards if transition.env_rewards is None else transition.env_rewards
        self.env_rewards[self.step].copy_(env_rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.old_cfm_loss[self.step].copy_(transition.old_cfm_loss)
        self.old_x1_pred[self.step].copy_(transition.old_x1_pred)
        self.cfm_eps[self.step].copy_(transition.cfm_eps)
        self.cfm_t[self.step].copy_(transition.cfm_t)
        if transition.old_log_prob is not None:
            self.old_log_prob[self.step].copy_(transition.old_log_prob.view(self.num_envs, 1))
        self.step += 1

    def compute_returns_and_advantages(
        self,
        *,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        normalize_advantage: bool,
    ) -> None:
        if self.step != self.num_steps:
            raise RuntimeError(f"Buffer has {self.step}/{self.num_steps} steps")
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[step] = advantage
            self.returns[step] = advantage + self.values[step]

        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def iter_minibatches(self, *, num_minibatches: int, num_epochs: int):
        if self.step != self.num_steps:
            raise RuntimeError(f"Buffer has {self.step}/{self.num_steps} steps")
        batch_size = self.num_steps * self.num_envs
        if num_minibatches < 1:
            raise ValueError("num_minibatches must be positive")
        num_minibatches = min(int(num_minibatches), batch_size)
        base = batch_size // num_minibatches
        remainder = batch_size % num_minibatches
        sizes = [base + (1 if i < remainder else 0) for i in range(num_minibatches)]

        flat_obs = self.obs.flatten(0, 1)
        flat_actions = self.actions.flatten(0, 1)
        flat_policy_actions = self.policy_actions.flatten(0, 1)
        flat_flow_actions = self.flow_actions.flatten(0, 1)
        flat_obj_com_err = self.obj_com_err.flatten(0, 1)
        flat_hand_mjpos_err = self.hand_mjpos_err.flatten(0, 1)
        flat_control_error = self.control_error.flatten(0, 1)
        flat_env_rewards = self.env_rewards.flatten(0, 1)
        flat_stage = self.stage.flatten(0, 1)
        flat_values = self.values.flatten(0, 1)
        flat_advantages = self.advantages.flatten(0, 1)
        flat_returns = self.returns.flatten(0, 1)
        flat_old_cfm_loss = self.old_cfm_loss.flatten(0, 1)
        flat_old_x1_pred = self.old_x1_pred.flatten(0, 1)
        flat_cfm_eps = self.cfm_eps.flatten(0, 1)
        flat_cfm_t = self.cfm_t.flatten(0, 1)
        flat_old_log_prob = self.old_log_prob.flatten(0, 1)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            start = 0
            for size in sizes:
                end = start + size
                batch_idx = indices[start:end]
                start = end
                yield FPOMiniBatch(
                    obs=flat_obs[batch_idx],
                    actions=flat_actions[batch_idx],
                    policy_actions=flat_policy_actions[batch_idx],
                    flow_actions=flat_flow_actions[batch_idx],
                    obj_com_err=flat_obj_com_err[batch_idx],
                    hand_mjpos_err=flat_hand_mjpos_err[batch_idx],
                    control_error=flat_control_error[batch_idx],
                    env_rewards=flat_env_rewards[batch_idx],
                    stage=flat_stage[batch_idx],
                    old_values=flat_values[batch_idx],
                    advantages=flat_advantages[batch_idx],
                    returns=flat_returns[batch_idx],
                    old_cfm_loss=flat_old_cfm_loss[batch_idx],
                    old_x1_pred=flat_old_x1_pred[batch_idx],
                    cfm_eps=flat_cfm_eps[batch_idx],
                    cfm_t=flat_cfm_t[batch_idx],
                    old_log_prob=flat_old_log_prob[batch_idx],
                )

    def iter_chunk_minibatches(self, *, chunk_steps: int, num_minibatches: int, num_epochs: int):
        if self.step != self.num_steps:
            raise RuntimeError(f"Buffer has {self.step}/{self.num_steps} steps")
        chunk_steps = int(chunk_steps)
        if chunk_steps <= 1:
            raise ValueError("chunk_steps must be greater than 1")
        if self.num_steps % chunk_steps != 0:
            raise ValueError("num_steps must be divisible by chunk_steps")
        num_chunks = self.num_steps // chunk_steps
        batch_size = num_chunks * self.num_envs
        if num_minibatches < 1:
            raise ValueError("num_minibatches must be positive")
        num_minibatches = min(int(num_minibatches), batch_size)
        base = batch_size // num_minibatches
        remainder = batch_size % num_minibatches
        sizes = [base + (1 if i < remainder else 0) for i in range(num_minibatches)]

        def chunk(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape[2:]
            return (
                x.reshape(num_chunks, chunk_steps, self.num_envs, *shape)
                .permute(0, 2, 1, *range(3, 3 + len(shape)))
                .reshape(batch_size, chunk_steps, *shape)
            )

        obs = chunk(self.obs)
        actions = chunk(self.actions)
        policy_actions = chunk(self.policy_actions)
        flow_actions = chunk(self.flow_actions)
        obj_com_err = chunk(self.obj_com_err)
        hand_mjpos_err = chunk(self.hand_mjpos_err)
        control_error = chunk(self.control_error)
        env_rewards = chunk(self.env_rewards)
        stage = chunk(self.stage)
        old_values = chunk(self.values)
        advantages = chunk(self.advantages)
        returns = chunk(self.returns)
        old_cfm_loss = chunk(self.old_cfm_loss)
        old_x1_pred = chunk(self.old_x1_pred)
        cfm_eps = chunk(self.cfm_eps)
        cfm_t = chunk(self.cfm_t)
        old_log_prob = chunk(self.old_log_prob)
        dones = chunk(self.dones).squeeze(-1)

        prior_dones = torch.cat([torch.zeros_like(dones[:, :1]), dones[:, :-1]], dim=1)
        valid_mask = torch.where(
            torch.cumsum(prior_dones, dim=1) > 0,
            torch.zeros(batch_size, chunk_steps, device=self.device),
            torch.ones(batch_size, chunk_steps, device=self.device),
        )
        valid_mask[:, 0] = 1.0

        if chunk_steps > 1:
            cfm_t = cfm_t[:, :1].expand_as(cfm_t).clone()

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            start = 0
            for size in sizes:
                end = start + size
                batch_idx = indices[start:end]
                start = end
                yield FPOChunkMiniBatch(
                    obs=obs[batch_idx],
                    actions=actions[batch_idx],
                    policy_actions=policy_actions[batch_idx],
                    flow_actions=flow_actions[batch_idx],
                    obj_com_err=obj_com_err[batch_idx],
                    hand_mjpos_err=hand_mjpos_err[batch_idx],
                    control_error=control_error[batch_idx],
                    env_rewards=env_rewards[batch_idx],
                    stage=stage[batch_idx],
                    old_values=old_values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_cfm_loss=old_cfm_loss[batch_idx],
                    old_x1_pred=old_x1_pred[batch_idx],
                    cfm_eps=cfm_eps[batch_idx],
                    cfm_t=cfm_t[batch_idx],
                    valid_mask=valid_mask[batch_idx],
                    old_log_prob=old_log_prob[batch_idx],
                )
