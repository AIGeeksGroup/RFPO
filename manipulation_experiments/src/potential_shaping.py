from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def potential_shaping_term(
    potential_before: float,
    potential_next: float,
    *,
    discount: float,
    terminal: bool,
) -> float:
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    values = np.asarray([potential_before, potential_next], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("potentials must be finite and nonnegative")
    effective_next = 0.0 if terminal else float(potential_next)
    return discount * effective_next - float(potential_before)


def audit_potential_records(
    records: Sequence[dict],
    expected_seeds: set[int],
    *,
    discount: float,
    activity_threshold: float = 1e-4,
) -> dict:
    if not records or not expected_seeds:
        raise ValueError("records and expected seeds must be nonempty")
    observed_seeds = {int(record["seed"]) for record in records}
    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"collection seeds differ: expected {expected_seeds}, got {observed_seeds}"
        )

    episode_results = []
    failure_nonterminal_terms = []
    for seed in sorted(expected_seeds):
        episode = sorted(
            (record for record in records if int(record["seed"]) == seed),
            key=lambda record: int(record["step"]),
        )
        if [int(record["step"]) for record in episode] != list(range(len(episode))):
            raise RuntimeError(f"seed {seed} has non-contiguous transition steps")
        terminals = [bool(record["terminal"]) for record in episode]
        if sum(terminals) != 1 or not terminals[-1]:
            raise RuntimeError(f"seed {seed} does not contain one complete episode")

        sparse_return = 0.0
        shaped_return = 0.0
        for step, record in enumerate(episode):
            values = np.asarray(
                [
                    record["sparse_reward"],
                    record["potential_before"],
                    record["potential_next"],
                    record["shaping_term"],
                    record["shaped_reward"],
                ],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"seed {seed} contains non-finite reward records")
            if not 0.0 <= record["potential_before"] <= 0.7 + 1e-7:
                raise RuntimeError(f"seed {seed} has an out-of-range current potential")
            if not 0.0 <= record["potential_next"] <= 0.7 + 1e-7:
                raise RuntimeError(f"seed {seed} has an out-of-range next potential")
            expected_term = potential_shaping_term(
                record["potential_before"],
                record["potential_next"],
                discount=discount,
                terminal=record["terminal"],
            )
            if abs(expected_term - float(record["shaping_term"])) > 1e-10:
                raise RuntimeError(f"seed {seed} has an inconsistent shaping term")
            if abs(
                float(record["sparse_reward"])
                + expected_term
                - float(record["shaped_reward"])
            ) > 1e-10:
                raise RuntimeError(f"seed {seed} has an inconsistent shaped reward")
            weight = discount**step
            sparse_return += weight * float(record["sparse_reward"])
            shaped_return += weight * float(record["shaped_reward"])

        residual = shaped_return - sparse_return + float(
            episode[0]["potential_before"]
        )
        success = bool(episode[-1]["success"])
        episode_results.append(
            {
                "seed": seed,
                "steps": len(episode),
                "success": success,
                "initial_potential": float(episode[0]["potential_before"]),
                "sparse_return": sparse_return,
                "shaped_return": shaped_return,
                "telescoping_residual": residual,
            }
        )
        if not success:
            failure_nonterminal_terms.extend(
                float(record["shaping_term"])
                for record in episode
                if not bool(record["terminal"])
            )

    if not failure_nonterminal_terms:
        raise RuntimeError("collection has no nonterminal transitions from failed episodes")
    terms = np.asarray(failure_nonterminal_terms, dtype=np.float64)
    active_fraction = float(np.mean(np.abs(terms) > activity_threshold))
    positive_fraction = float(np.mean(terms > activity_threshold))
    negative_fraction = float(np.mean(terms < -activity_threshold))
    max_residual = max(abs(item["telescoping_residual"]) for item in episode_results)
    passed = (
        max_residual <= 1e-5
        and active_fraction >= 0.20
        and positive_fraction >= 0.05
        and negative_fraction >= 0.05
    )
    return {
        "episodes": len(episode_results),
        "successes": sum(item["success"] for item in episode_results),
        "failures": sum(not item["success"] for item in episode_results),
        "transitions": len(records),
        "failure_nonterminal_transitions": int(terms.size),
        "max_abs_telescoping_residual": float(max_residual),
        "active_fraction": active_fraction,
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "term_mean": float(terms.mean()),
        "term_std": float(terms.std()),
        "term_min": float(terms.min()),
        "term_max": float(terms.max()),
        "episode_results": episode_results,
        "passed": passed,
    }
