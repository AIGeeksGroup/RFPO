#!/usr/bin/env python3
"""Generate publication figures from committed experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GRAY = "#777777"
LIGHT_GRAY = "#D0D0D0"


def load(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def generate_main_results() -> None:
    policy = load("experiments/go2-antithetic-cross-policy-summary/results/analysis.json")
    mechanism = load("experiments/go2-antithetic-affine-mechanism/results/analysis.json")
    spot = load("experiments/spot-antithetic-cross-task/results/analysis.json")

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.25))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    estimand_keys = [
        "antithetic32_minus_random64_equal_total_nfe",
        "antithetic64_minus_iid_pair64_equal_pair_nfe",
        "antithetic32_minus_zero32_deployment_control",
    ]
    labels = [
        "Symmetric32 - random64",
        "Symmetric64 - IID-pair64",
        "Symmetric32 - zero32",
    ]
    y = np.arange(3)[::-1]
    for row, key in enumerate(estimand_keys):
        item = policy["estimands"][key]
        ypos = y[row]
        effects = np.array(list(item["seed_effects"].values()))
        interval = item["student_t_95"]
        ax_a.plot(
            effects,
            np.full_like(effects, ypos, dtype=float),
            "o",
            color=GRAY,
            markerfacecolor="white",
            markeredgewidth=1,
            markersize=4,
            zorder=3,
        )
        ax_a.errorbar(
            item["mean"],
            ypos,
            xerr=[[item["mean"] - interval[0]], [interval[1] - item["mean"]]],
            fmt="D",
            color=VERMILLION if row < 2 else BLUE,
            capsize=3,
            markersize=4.5,
            linewidth=1.5,
            zorder=4,
        )
    ax_a.axvline(0, color="black", linewidth=0.8)
    ax_a.set_yticks(y, labels)
    ax_a.set_xlabel("Paired return difference")
    ax_a.set_title("Go2: effects across four trained policies")
    ax_a.grid(axis="y", visible=False)
    ax_a.plot([], [], "o", color=GRAY, markerfacecolor="white", label="Policy seed")
    ax_a.plot([], [], "D", color=VERMILLION, label="Mean and 95% t interval")
    ax_a.legend(loc="lower right")

    seeds = ["42", "43", "44", "45"]
    x = np.arange(len(seeds))
    anti = [mechanism["policies"][seed]["antithetic_residual_ratio"] for seed in seeds]
    iid = [mechanism["policies"][seed]["iid_residual_ratio"] for seed in seeds]
    width = 0.34
    ax_b.bar(
        x - width / 2,
        anti,
        width,
        label="Symmetric pair",
        color=BLUE,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
    )
    ax_b.bar(
        x + width / 2,
        iid,
        width,
        label="IID pair",
        color=ORANGE,
        edgecolor="black",
        linewidth=0.5,
        hatch="...",
    )
    ax_b.set_xticks(x, seeds)
    ax_b.set_xlabel("Go2 training seed")
    ax_b.set_ylabel("Residual RMS / single-source RMS")
    ax_b.set_ylim(0, 0.92)
    ax_b.set_title("Symmetry cancels source displacement")
    ax_b.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))

    methods = ["zero64", "zero32", "random64", "iid_pair32", "antithetic32"]
    method_labels = ["Zero64", "Zero32", "Random64", "IID-pair32", "Symmetric32"]
    means = [spot["mean_returns"][key] for key in methods]
    colors = [GRAY, LIGHT_GRAY, ORANGE, GREEN, BLUE]
    bars = ax_c.bar(
        np.arange(len(methods)),
        means,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        hatch=["", "//", "..", "xx", "///"],
    )
    ax_c.set_xticks(np.arange(len(methods)), method_labels, rotation=24, ha="right")
    ax_c.set_ylabel("Mean episode return")
    ax_c.set_ylim(300, 336)
    ax_c.set_title("Spot: descriptive method means")
    for bar, value in zip(bars, means):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.65,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    contrast_keys = [
        "antithetic32_minus_random64",
        "antithetic32_minus_iid_pair32",
        "antithetic32_minus_zero32",
    ]
    contrast_labels = ["Symmetric - random", "Symmetric - IID pair", "Symmetric - zero"]
    cy = np.arange(3)[::-1]
    for ypos, key, color in zip(cy, contrast_keys, [VERMILLION, GREEN, BLUE]):
        item = spot["comparisons"][key]
        interval = item["bootstrap_95"]
        ax_d.errorbar(
            item["gain"],
            ypos,
            xerr=[[item["gain"] - interval[0]], [interval[1] - item["gain"]]],
            fmt="D",
            color=color,
            capsize=3,
            markersize=5,
            linewidth=1.5,
        )
    ax_d.axvline(0, color="black", linewidth=0.8)
    ax_d.set_yticks(cy, contrast_labels)
    ax_d.set_xlabel("Paired return difference")
    ax_d.set_title("Spot: paired bootstrap 95% intervals")
    ax_d.grid(axis="y", visible=False)

    for label, ax in zip("ABCD", axes.flat):
        panel_label(ax, label)
    fig.subplots_adjust(hspace=0.48, wspace=0.44)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"fig_main_results.{suffix}", **kwargs)
    plt.close(fig)


def generate_efficiency_and_boundary() -> None:
    throughput = load("experiments/go2-antithetic-batched-throughput/results/benchmark.json")
    rollout = load("experiments/go2-mirrored-rollout-variance/results/analysis.json")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.35))
    timing_keys = ["random64", "antithetic32_sequential", "antithetic32_batched"]
    labels = ["Random64", "Symmetric32\nsequential", "Symmetric32\nbatched"]
    milliseconds = [
        1000 * throughput["timings"][key]["median_seconds_per_call"]
        for key in timing_keys
    ]
    bars = ax_a.bar(
        np.arange(3),
        milliseconds,
        color=[ORANGE, LIGHT_GRAY, BLUE],
        edgecolor="black",
        linewidth=0.5,
        hatch=["..", "//", "///"],
    )
    ax_a.set_xticks(np.arange(3), labels)
    ax_a.set_ylabel("Policy latency (ms / 4096 actions)")
    ax_a.set_ylim(0, 5.0)
    ax_a.set_title("H200 policy-only throughput")
    for bar, value in zip(bars, milliseconds):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ratio = rollout["statistics"]["variance_ratio"]
    interval = rollout["statistics"]["variance_ratio_bootstrap_95"]
    ax_b.errorbar(
        ratio,
        0,
        xerr=[[ratio - interval[0]], [interval[1] - ratio]],
        fmt="D",
        color=VERMILLION,
        capsize=4,
        markersize=6,
        linewidth=1.6,
    )
    ax_b.axvline(1.0, color="black", linestyle="--", linewidth=1, label="No reduction")
    ax_b.axvline(0.8, color=BLUE, linestyle=":", linewidth=1.2, label="Locked gate")
    ax_b.set_yticks([0], ["Mirrored / IID pair"])
    ax_b.set_xlim(0.3, 1.55)
    ax_b.set_xlabel("Closed-loop residual variance ratio")
    ax_b.set_title("Mirrored-rollout variance boundary")
    ax_b.grid(axis="y", visible=False)
    ax_b.legend(loc="upper right")

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    fig.subplots_adjust(wspace=0.45)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"fig_efficiency_boundary.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    style()
    generate_main_results()
    generate_efficiency_and_boundary()


if __name__ == "__main__":
    main()
