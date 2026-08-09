from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn, optim

from src.discounted_success_critic import spearman_rank_correlation


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("critic input dimension must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs).squeeze(-1)


def discounted_success_labels(
    durations: Sequence[int], rewards: Sequence[float], discount: float
) -> Tensor:
    if len(durations) != len(rewards) or not durations:
        raise ValueError("durations and rewards must be nonempty and have matching lengths")
    if not 0.0 < discount <= 1.0 or min(durations) <= 0:
        raise ValueError("discount and durations must be positive")
    labels = torch.empty(len(durations), dtype=torch.float32)
    future = 0.0
    for index in reversed(range(len(durations))):
        future = float(rewards[index]) + discount ** int(durations[index]) * future
        labels[index] = future
    return labels


def stratified_episode_folds(success_by_seed: dict[int, bool]) -> tuple[set[int], set[int]]:
    if not success_by_seed:
        raise ValueError("episode outcomes cannot be empty")
    folds = (set(), set())
    for outcome in (False, True):
        seeds = sorted(seed for seed, success in success_by_seed.items() if success is outcome)
        for index, seed in enumerate(seeds):
            folds[index % 2].add(seed)
    if not folds[0] or not folds[1] or folds[0] & folds[1]:
        raise ValueError("stratified folds must be nonempty and disjoint")
    if folds[0] | folds[1] != set(success_by_seed):
        raise RuntimeError("stratified folds do not cover every episode")
    return folds


def normalize_from_train(train: Tensor, evaluation: Tensor) -> tuple[Tensor, Tensor]:
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("critic inputs must be compatible matrices")
    mean = train.mean(0)
    std = train.std(0, unbiased=False).clamp_min(1e-6)
    return (train - mean) / std, (evaluation - mean) / std


def fit_and_evaluate(
    train_inputs: Tensor,
    train_labels: Tensor,
    evaluation_inputs: Tensor,
    evaluation_labels: Tensor,
    *,
    seed: int,
    epochs: int = 10,
    num_minibatches: int = 8,
    learning_rate: float = 1e-4,
) -> dict[str, float]:
    if min(epochs, num_minibatches) <= 0 or learning_rate <= 0:
        raise ValueError("critic optimization settings must be positive")
    train_inputs, evaluation_inputs = normalize_from_train(train_inputs, evaluation_inputs)
    device = train_inputs.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ValueNetwork(train_inputs.shape[1]).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, eps=1e-5, weight_decay=1e-6
    )
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(train_labels.numel(), generator=generator)
        for indices_cpu in torch.tensor_split(permutation, num_minibatches):
            if indices_cpu.numel() == 0:
                continue
            indices = indices_cpu.to(device)
            loss = (model(train_inputs[indices]) - train_labels[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(evaluation_inputs)
        mse = (predictions - evaluation_labels).square().mean()
        spearman = spearman_rank_correlation(predictions.cpu(), evaluation_labels.cpu())
    return {
        "mse": float(mse.item()),
        "spearman": float(spearman),
        "prediction_mean": float(predictions.mean().item()),
        "prediction_std": float(predictions.std(unbiased=False).item()),
    }


def validate_collection(records: Sequence[dict], expected_seeds: set[int]) -> dict[int, bool]:
    seeds = {int(record["seed"]) for record in records}
    if seeds != expected_seeds:
        raise RuntimeError(f"collection seeds differ: expected {expected_seeds}, got {seeds}")
    outcomes: dict[int, bool] = {}
    for seed in sorted(seeds):
        episode = [record for record in records if int(record["seed"]) == seed]
        terminal_count = sum(bool(record["terminal"]) for record in episode)
        if terminal_count != 1 or not bool(episode[-1]["terminal"]):
            raise RuntimeError(f"seed {seed} does not contain exactly one terminal episode")
        outcomes[seed] = bool(episode[-1]["success"])
        for record in episode:
            for key in ("visual", "privileged"):
                value = torch.as_tensor(record[key])
                if value.ndim != 1 or not torch.isfinite(value).all():
                    raise RuntimeError(f"seed {seed} contains invalid {key} features")
    successes = sum(outcomes.values())
    failures = len(outcomes) - successes
    if min(successes, failures) < 8:
        raise RuntimeError(
            f"collection lacks outcome support: {successes} successes and {failures} failures"
        )
    return outcomes
