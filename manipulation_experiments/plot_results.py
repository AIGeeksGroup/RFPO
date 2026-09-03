"""
Consolidated plotting script for FPO experiment results.

Usage:
    python plot_results.py --mode main_benchmark
    python plot_results.py --mode fpoplusplus_ablation
    python plot_results.py --mode base_policy_ablation
    python plot_results.py --mode main_benchmark --output my_plot.pdf
"""

import argparse

import pandas as pd
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm.auto import tqdm


# =============================================================================
# Shared utility functions
# =============================================================================

def fetch_data(api, configs, name):
    """Fetch data for all groups given a configuration dict."""
    all_data = []

    for group_label, config in configs.items():
        print(f"Fetching {name} data for: {group_label}...")

        for run_id in tqdm(config["run_ids"]):
            try:
                run_path = f"{config['project']}/runs/{run_id}"
                run = api.run(run_path)

                if config["use_merge_asof"]:
                    # Fetch sparse eval metric and dense training step separately
                    hist_metric = run.history(
                        keys=[config["metric_key"], "_step"],
                        samples=5000
                    )
                    hist_step = run.history(
                        keys=[config["x_axis_key"], "_step"],
                        samples=5000
                    )

                    hist_metric = hist_metric.dropna(subset=[config["metric_key"]])
                    hist_step = hist_step.dropna(subset=[config["x_axis_key"]])

                    hist_metric = hist_metric.sort_values(by="_step")
                    hist_step = hist_step.sort_values(by="_step")

                    hist = pd.merge_asof(
                        hist_metric,
                        hist_step,
                        on="_step",
                        direction="backward"
                    )
                    hist = hist.dropna(subset=[config["x_axis_key"]])
                else:
                    # Simple fetch with _step
                    hist = run.history(
                        keys=[config["metric_key"], config["x_axis_key"]],
                        samples=5000
                    )
                    hist = hist.dropna(subset=[config["metric_key"]])

                # Add initial point (if initial_rate is provided)
                if config["initial_rate"] is not None:
                    initial_point = pd.DataFrame({
                        config["x_axis_key"]: [0],
                        config["metric_key"]: [config["initial_rate"]]
                    })
                    hist = pd.concat([initial_point, hist], ignore_index=True)

                hist['Group'] = group_label
                hist = hist.rename(columns={
                    config["metric_key"]: "Success Rate",
                    config["x_axis_key"]: "total env step"
                })

                all_data.append(hist)

            except Exception as e:
                print(f"Error fetching run {run_id}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Trim to minimum max step across groups
        min_max_step = combined_df.groupby(['Group'])['total env step'].max().min()
        combined_df = combined_df[combined_df['total env step'] <= min_max_step]
        return combined_df
    return None


def downsample_data(df, factor):
    """Downsample data by keeping every Nth point per Group."""
    if factor <= 1:
        return df

    downsampled_dfs = []
    for group in df['Group'].unique():
        group_df = df[df['Group'] == group].copy()
        group_df = group_df.sort_values('total env step')
        group_df = group_df.iloc[::factor]
        downsampled_dfs.append(group_df)

    return pd.concat(downsampled_dfs, ignore_index=True)


def smooth_data(df, smoothing):
    """Apply exponential moving average smoothing to Success Rate per Group."""
    if smoothing <= 0:
        return df

    weight = smoothing / (smoothing + 1)
    smoothed_dfs = []

    for group in df['Group'].unique():
        group_df = df[df['Group'] == group].copy()
        group_df = group_df.sort_values('total env step')
        group_df['Success Rate'] = group_df['Success Rate'].ewm(alpha=1 - weight).mean()
        smoothed_dfs.append(group_df)

    return pd.concat(smoothed_dfs, ignore_index=True)


def fetch_and_process(api, configs, name, downsample_factor, smoothing_factor):
    """Fetch, downsample, and smooth data in one call."""
    df = fetch_data(api, configs, name)
    if df is not None:
        df = downsample_data(df, downsample_factor)
        df = smooth_data(df, smoothing_factor)
    return df


# =============================================================================
# Mode: fpoplusplus_ablation (FPO++ ablation: FPO++ vs ASPO vs per-action ratio)
# =============================================================================

def run_fpoplusplus_ablation(api, output):
    # --- Plot Configuration ---
    SQUARE_Y_LOWER = 0.0
    SQUARE_Y_UPPER = 0.65
    THREADING_Y_LOWER = 0.1
    THREADING_Y_UPPER = 1.0
    SQUARE_RANDOM_Y_LOWER = 0.0
    SQUARE_RANDOM_Y_UPPER = 0.6
    THREADING_RANDOM_Y_LOWER = 0.0
    THREADING_RANDOM_Y_UPPER = 1.0

    SQUARE_DOWNSAMPLE = 1
    THREADING_DOWNSAMPLE = 1

    SQUARE_SMOOTHING = 0.95
    THREADING_SMOOTHING = 0.95

    COLOR_PALETTE = {
        "FPO++": "#00BFFF",           # Deep sky blue
        "ASPO": "#A9A9A9",            # Dark gray
        "Per-action ratio": "#D3D3D3", # Light gray
    }

    GROUP_ORDER = ["ASPO", "Per-action ratio", "FPO++"]
    legends = {
        "Per-action ratio": "FPO++ with per-action ratio",
        "ASPO": "FPO++ with ASPO",
        "FPO++": "FPO++ with PPO",
    }

    # =============================================================================
    # Square Configuration (Zero Sampling)
    # =============================================================================
    square_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["z2u9ryms", "wzyv707a", "rsmunbo4"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
        "ASPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["whhhg3oc", "safs53cm", "069ligh5"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
        "Per-action ratio": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["eikmrvae", "kf0ywqzj", "4x5cxo0w"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Threading Configuration (Zero Sampling)
    # =============================================================================
    threading_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["bt3sl4ex", "fu8wpmbd", "g2ldgoss"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
        "ASPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4pwr3dzu", "ir471vi8", "nx0ktwru"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
        "Per-action ratio": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["l5hdijzn", "pcu57hf1", "xmklzaca"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Square Configuration (Random Sampling)
    # =============================================================================
    square_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["z2u9ryms", "wzyv707a", "rsmunbo4"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,
            "use_merge_asof": False
        },
        "ASPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["whhhg3oc", "safs53cm", "069ligh5"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,
            "use_merge_asof": False
        },
        "Per-action ratio": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["eikmrvae", "kf0ywqzj", "4x5cxo0w"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Threading Configuration (Random Sampling)
    # =============================================================================
    threading_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["bt3sl4ex", "fu8wpmbd", "g2ldgoss"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,
            "use_merge_asof": False
        },
        "ASPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4pwr3dzu", "ir471vi8", "nx0ktwru"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,
            "use_merge_asof": False
        },
        "Per-action ratio": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["l5hdijzn", "pcu57hf1", "xmklzaca"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,
            "use_merge_asof": False
        },
    }

    # --- Fetch all data ---
    print("=" * 50)
    print("Fetching Square data (Zero Sampling)...")
    print("=" * 50)
    square_df = fetch_and_process(api, square_configs, "Square", SQUARE_DOWNSAMPLE, SQUARE_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Threading data (Zero Sampling)...")
    print("=" * 50)
    threading_df = fetch_and_process(api, threading_configs, "Threading", THREADING_DOWNSAMPLE, THREADING_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Square data (Random Sampling)...")
    print("=" * 50)
    square_random_df = fetch_and_process(api, square_random_configs, "Square Random", SQUARE_DOWNSAMPLE, SQUARE_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Threading data (Random Sampling)...")
    print("=" * 50)
    threading_random_df = fetch_and_process(api, threading_random_configs, "Threading Random", THREADING_DOWNSAMPLE, THREADING_SMOOTHING)

    # --- Create combined figure ---
    zero_data_available = (square_df is not None and threading_df is not None)
    random_data_available = (square_random_df is not None and threading_random_df is not None)

    if zero_data_available or random_data_available:
        print("\nGenerating combined ablation plot...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

        # =========================================================================
        # ROW 1: Zero Sampling
        # =========================================================================

        # Plot Square (Zero Sampling)
        if square_df is not None:
            sns.lineplot(
                data=square_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax1,
                legend=False
            )
        ax1.set_ylim(SQUARE_Y_LOWER, SQUARE_Y_UPPER)
        ax1.set_title("Square", fontsize=28, pad=20)
        ax1.set_xlabel("")
        ax1.set_ylabel("Zero Sampling\nSuccess Rate", fontsize=24, labelpad=10)
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax1.tick_params(axis='both', labelsize=22)
        ax1.xaxis.get_offset_text().set_fontsize(18)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Threading (Zero Sampling)
        if threading_df is not None:
            sns.lineplot(
                data=threading_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax2,
                legend=False
            )
        ax2.set_ylim(THREADING_Y_LOWER, THREADING_Y_UPPER)
        ax2.set_title("Threading", fontsize=28, pad=20)
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.tick_params(axis='both', labelsize=22)
        ax2.xaxis.get_offset_text().set_fontsize(18)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        # =========================================================================
        # ROW 2: Random Sampling
        # =========================================================================

        # Plot Square (Random Sampling)
        if square_random_df is not None:
            sns.lineplot(
                data=square_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax3,
                legend=False
            )
        ax3.set_ylim(SQUARE_RANDOM_Y_LOWER, SQUARE_RANDOM_Y_UPPER)
        ax3.set_title("")
        ax3.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax3.set_ylabel("Random Sampling\nSuccess Rate", fontsize=24, labelpad=10)
        ax3.yaxis.set_major_locator(MaxNLocator(4))
        ax3.tick_params(axis='both', labelsize=22)
        ax3.xaxis.get_offset_text().set_fontsize(18)
        ax3.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Threading (Random Sampling)
        if threading_random_df is not None:
            sns.lineplot(
                data=threading_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax4,
                legend=False
            )
        ax4.set_ylim(THREADING_RANDOM_Y_LOWER, THREADING_RANDOM_Y_UPPER)
        ax4.set_title("")
        ax4.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax4.set_ylabel("")
        ax4.yaxis.set_major_locator(MaxNLocator(4))
        ax4.tick_params(axis='both', labelsize=22)
        ax4.xaxis.get_offset_text().set_fontsize(18)
        ax4.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Create single legend at the bottom
        handles = [plt.Line2D([0], [0], color=COLOR_PALETTE[g], linewidth=5) for g in GROUP_ORDER]
        labels = [legends[g] for g in GROUP_ORDER[::-1]]
        fig.legend(
            handles[::-1],
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize=28,
            frameon=True
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output}")

        # Also save as PNG
        png_output = output.rsplit('.', 1)[0] + '.png'
        plt.savefig(png_output, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {png_output}")

        plt.show()
    else:
        print("Failed to fetch data for the plots.")


# =============================================================================
# Mode: main_benchmark (all 5 tasks, 4 methods, zero + random sampling)
# =============================================================================

def run_main_benchmark(api, output):
    # --- Plot Configuration ---
    CAN_Y_LOWER = 0.0
    CAN_Y_UPPER = 1.0
    SQUARE_Y_LOWER = 0.25
    SQUARE_Y_UPPER = 0.65
    BOX_CLEARANCE_Y_LOWER = 0.1
    BOX_CLEARANCE_Y_UPPER = 1.0
    TRAY_LIFTING_Y_LOWER = 0.0
    TRAY_LIFTING_Y_UPPER = 1.0
    THREADING_Y_LOWER = 0.0
    THREADING_Y_UPPER = 1.0

    CAN_RANDOM_Y_LOWER = 0.0
    CAN_RANDOM_Y_UPPER = 1.0
    SQUARE_RANDOM_Y_LOWER = 0.25
    SQUARE_RANDOM_Y_UPPER = 0.6
    BOX_CLEARANCE_RANDOM_Y_LOWER = 0.1
    BOX_CLEARANCE_RANDOM_Y_UPPER = 1.0
    TRAY_LIFTING_RANDOM_Y_LOWER = 0.0
    TRAY_LIFTING_RANDOM_Y_UPPER = 1.0
    THREADING_RANDOM_Y_LOWER = 0.0
    THREADING_RANDOM_Y_UPPER = 1.0

    CAN_DOWNSAMPLE = 1
    SQUARE_DOWNSAMPLE = 1
    BOX_CLEARANCE_DOWNSAMPLE = 1
    TRAY_LIFTING_DOWNSAMPLE = 1
    THREADING_DOWNSAMPLE = 1

    CAN_SMOOTHING = 0.8
    SQUARE_SMOOTHING = 0.95
    BOX_CLEARANCE_SMOOTHING = 0.8
    TRAY_LIFTING_SMOOTHING = 0.85
    THREADING_SMOOTHING = 0.95

    COLOR_PALETTE = {
        "FPO++": "#00BFFF",        # Deep sky blue
        "Vanilla FPO": "#666666",  # Slate gray
        "DPPO - fixed noise": "#32CD32",          # Lime green
        "DPPO - learned noise": "#FF8C00",     # Dark orange
    }

    GROUP_ORDER = ["DPPO - learned noise", "DPPO - fixed noise", "Vanilla FPO", "FPO++"]

    # =============================================================================
    # CAN Configuration (Zero Sampling)
    # =============================================================================
    can_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e0h4wy1r", "i6rmkgrh", "wbxzw7z3"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gj312c6y", "k2kvz5n2", "o84n7r75"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["3kdb60fl", "2ley356r", "482uupak"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e11pm661", "wt64jgp5", "sx1h2e3d"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Square Configuration (Zero Sampling)
    # =============================================================================
    square_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["z2u9ryms", "wzyv707a", "rsmunbo4"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["745i6plq", "pa1r700a", "smdfpqvl"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["un79mjlp", "tanu9fup", "lg0q8txy"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4p2y062m", "cvwx0b0i", "7jhcs2bd"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2837,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Box Clearance Configuration (Zero Sampling)
    # =============================================================================
    box_clearance_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["o6kv0feo", "qlux3x9d", "ujjdjtov"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7398,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["23jsmjy6", "bd97icaa", "u6wpu7t9"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7398,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["93o9w0re", "vcu4ig3r", "sobx9t35"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7398,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["cpyryymn", "g5u3986h", "w7ed81pa"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7398,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Tray Lifting Configuration (Zero Sampling)
    # =============================================================================
    tray_lifting_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["rdigmvk4", "oi516dvb", "dcwh6cja"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6252,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["0r7h65oz", "6ws7b7ww", "sfeuum4m"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6252,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["mftw0dor", "6o61ur1d", "ysc7e8i4"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6252,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["y6m73qf6", "hald8fwf", "pjv4oekf"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6252,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Threading Configuration (Zero Sampling)
    # =============================================================================
    threading_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["bt3sl4ex", "fu8wpmbd", "g2ldgoss"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["2zqt8qnd", "h8fv0ykv", "uvvyaqni"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["siy3wxtk", "suomxjcx", "rgrer2ba"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["xtl4xpds", "3g2cydj6", "du5sms92"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2254,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # RANDOM SAMPLING CONFIGURATIONS
    # =============================================================================

    # =============================================================================
    # CAN Configuration (Random Sampling)
    # =============================================================================
    can_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e0h4wy1r", "i6rmkgrh", "wbxzw7z3"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.10,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gj312c6y", "k2kvz5n2", "o84n7r75"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.10,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["3kdb60fl", "2ley356r", "482uupak"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.10,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e11pm661", "wt64jgp5", "sx1h2e3d"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.10,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Square Configuration (Random Sampling)
    # =============================================================================
    square_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["z2u9ryms", "wzyv707a", "rsmunbo4"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["745i6plq", "pa1r700a", "smdfpqvl"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["un79mjlp", "tanu9fup", "lg0q8txy"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4p2y062m", "cvwx0b0i", "7jhcs2bd"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2831,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Box Clearance Configuration (Random Sampling)
    # =============================================================================
    box_clearance_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["o6kv0feo", "qlux3x9d", "ujjdjtov"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.3637,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["23jsmjy6", "bd97icaa", "u6wpu7t9"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.3637,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["93o9w0re", "vcu4ig3r", "sobx9t35"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.3637,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["cpyryymn", "g5u3986h", "w7ed81pa"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.3637,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Tray Lifting Configuration (Random Sampling)
    # =============================================================================
    tray_lifting_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["rdigmvk4", "oi516dvb", "dcwh6cja"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2915,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["0r7h65oz", "6ws7b7ww", "sfeuum4m"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2915,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["mftw0dor", "6o61ur1d", "ysc7e8i4"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2915,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["y6m73qf6", "hald8fwf", "pjv4oekf"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.2915,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # Threading Configuration (Random Sampling)
    # =============================================================================
    threading_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["bt3sl4ex", "fu8wpmbd", "g2ldgoss"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["2zqt8qnd", "h8fv0ykv", "uvvyaqni"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["siy3wxtk", "suomxjcx", "rgrer2ba"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["xtl4xpds", "3g2cydj6", "du5sms92"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.094,  # TODO: Replace with actual initial rate
            "use_merge_asof": False
        },
    }

    # --- Fetch all zero sampling data ---
    print("=" * 50)
    print("Fetching CAN data...")
    print("=" * 50)
    can_df = fetch_and_process(api, can_configs, "CAN", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Square data...")
    print("=" * 50)
    square_df = fetch_and_process(api, square_configs, "Square", SQUARE_DOWNSAMPLE, SQUARE_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Box Clearance data...")
    print("=" * 50)
    box_clearance_df = fetch_and_process(api, box_clearance_configs, "Box Cleanup", BOX_CLEARANCE_DOWNSAMPLE, BOX_CLEARANCE_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Tray Lifting data...")
    print("=" * 50)
    tray_lifting_df = fetch_and_process(api, tray_lifting_configs, "Tray Lift", TRAY_LIFTING_DOWNSAMPLE, TRAY_LIFTING_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Threading data...")
    print("=" * 50)
    threading_df = fetch_and_process(api, threading_configs, "Threading", THREADING_DOWNSAMPLE, THREADING_SMOOTHING)

    # --- Fetch all random sampling data ---
    print("\n" + "=" * 50)
    print("Fetching CAN Random Sampling data...")
    print("=" * 50)
    can_random_df = fetch_and_process(api, can_random_configs, "CAN Random", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Square Random Sampling data...")
    print("=" * 50)
    square_random_df = fetch_and_process(api, square_random_configs, "Square Random", SQUARE_DOWNSAMPLE, SQUARE_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Box Clearance Random Sampling data...")
    print("=" * 50)
    box_clearance_random_df = fetch_and_process(api, box_clearance_random_configs, "Box Cleanup Random", BOX_CLEARANCE_DOWNSAMPLE, BOX_CLEARANCE_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Tray Lifting Random Sampling data...")
    print("=" * 50)
    tray_lifting_random_df = fetch_and_process(api, tray_lifting_random_configs, "Tray Lift Random", TRAY_LIFTING_DOWNSAMPLE, TRAY_LIFTING_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching Threading Random Sampling data...")
    print("=" * 50)
    threading_random_df = fetch_and_process(api, threading_random_configs, "Threading Random", THREADING_DOWNSAMPLE, THREADING_SMOOTHING)

    # --- Create combined figure ---
    zero_data_available = (can_df is not None and square_df is not None and
                           box_clearance_df is not None and tray_lifting_df is not None and
                           threading_df is not None)

    random_data_available = (can_random_df is not None and square_random_df is not None and
                             box_clearance_random_df is not None and tray_lifting_random_df is not None and
                             threading_random_df is not None)

    if zero_data_available or random_data_available:
        print("\nGenerating combined plot...")

        fig, axes = plt.subplots(2, 5, figsize=(24, 10))

        ax1, ax2, ax3, ax4, ax5 = axes[0]
        ax6, ax7, ax8, ax9, ax10 = axes[1]

        # =========================================================================
        # ROW 1: Zero Sampling
        # =========================================================================

        # Plot CAN (Zero Sampling)
        if can_df is not None:
            sns.lineplot(
                data=can_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax1,
                legend=False
            )
        ax1.set_ylim(CAN_Y_LOWER, CAN_Y_UPPER)
        ax1.set_title("Can", fontsize=28, pad=20)
        ax1.set_xlabel("")
        ax1.set_ylabel("Zero Sampling\nSuccess Rate", fontsize=24, labelpad=10)
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax1.tick_params(axis='both', labelsize=22)
        ax1.xaxis.get_offset_text().set_fontsize(18)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Square (Zero Sampling)
        if square_df is not None:
            sns.lineplot(
                data=square_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax2,
                legend=False
            )
        ax2.set_ylim(SQUARE_Y_LOWER, SQUARE_Y_UPPER)
        ax2.set_title("Square", fontsize=28, pad=20)
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.tick_params(axis='both', labelsize=22)
        ax2.xaxis.get_offset_text().set_fontsize(18)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Box Clearance (Zero Sampling)
        if box_clearance_df is not None:
            sns.lineplot(
                data=box_clearance_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax3,
                legend=False
            )
        ax3.set_ylim(BOX_CLEARANCE_Y_LOWER, BOX_CLEARANCE_Y_UPPER)
        ax3.set_title("Box Cleanup", fontsize=28, pad=20)
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.yaxis.set_major_locator(MaxNLocator(4))
        ax3.tick_params(axis='both', labelsize=22)
        ax3.xaxis.get_offset_text().set_fontsize(18)
        ax3.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Tray Lifting (Zero Sampling)
        if tray_lifting_df is not None:
            sns.lineplot(
                data=tray_lifting_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax4,
                legend=False
            )
        ax4.set_ylim(TRAY_LIFTING_Y_LOWER, TRAY_LIFTING_Y_UPPER)
        ax4.set_title("Tray Lift", fontsize=28, pad=20)
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.yaxis.set_major_locator(MaxNLocator(4))
        ax4.tick_params(axis='both', labelsize=22)
        ax4.xaxis.get_offset_text().set_fontsize(18)
        ax4.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Threading (Zero Sampling)
        if threading_df is not None:
            sns.lineplot(
                data=threading_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax5,
                legend=False
            )
        ax5.set_ylim(THREADING_Y_LOWER, THREADING_Y_UPPER)
        ax5.set_title("Threading", fontsize=28, pad=20)
        ax5.set_xlabel("")
        ax5.set_ylabel("")
        ax5.yaxis.set_major_locator(MaxNLocator(4))
        ax5.tick_params(axis='both', labelsize=22)
        ax5.xaxis.get_offset_text().set_fontsize(18)
        ax5.grid(True, which="both", linestyle="--", linewidth=0.5)

        # =========================================================================
        # ROW 2: Random Sampling
        # =========================================================================

        # Plot CAN (Random Sampling)
        if can_random_df is not None:
            sns.lineplot(
                data=can_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax6,
                legend=False
            )
        ax6.set_ylim(CAN_RANDOM_Y_LOWER, CAN_RANDOM_Y_UPPER)
        ax6.set_title("")
        ax6.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax6.set_ylabel("Random Sampling\nSuccess Rate", fontsize=24, labelpad=10)
        ax6.yaxis.set_major_locator(MaxNLocator(4))
        ax6.tick_params(axis='both', labelsize=22)
        ax6.xaxis.get_offset_text().set_fontsize(18)
        ax6.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Square (Random Sampling)
        if square_random_df is not None:
            sns.lineplot(
                data=square_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax7,
                legend=False
            )
        ax7.set_ylim(SQUARE_RANDOM_Y_LOWER, SQUARE_RANDOM_Y_UPPER)
        ax7.set_title("")
        ax7.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax7.set_ylabel("")
        ax7.yaxis.set_major_locator(MaxNLocator(4))
        ax7.tick_params(axis='both', labelsize=22)
        ax7.xaxis.get_offset_text().set_fontsize(18)
        ax7.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Box Clearance (Random Sampling)
        if box_clearance_random_df is not None:
            sns.lineplot(
                data=box_clearance_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax8,
                legend=False
            )
        ax8.set_ylim(BOX_CLEARANCE_RANDOM_Y_LOWER, BOX_CLEARANCE_RANDOM_Y_UPPER)
        ax8.set_title("")
        ax8.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax8.set_ylabel("")
        ax8.yaxis.set_major_locator(MaxNLocator(4))
        ax8.tick_params(axis='both', labelsize=22)
        ax8.xaxis.get_offset_text().set_fontsize(18)
        ax8.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Tray Lifting (Random Sampling)
        if tray_lifting_random_df is not None:
            sns.lineplot(
                data=tray_lifting_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax9,
                legend=False
            )
        ax9.set_ylim(TRAY_LIFTING_RANDOM_Y_LOWER, TRAY_LIFTING_RANDOM_Y_UPPER)
        ax9.set_title("")
        ax9.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax9.set_ylabel("")
        ax9.yaxis.set_major_locator(MaxNLocator(4))
        ax9.tick_params(axis='both', labelsize=22)
        ax9.xaxis.get_offset_text().set_fontsize(18)
        ax9.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot Threading (Random Sampling)
        if threading_random_df is not None:
            sns.lineplot(
                data=threading_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax10,
                legend=False
            )
        ax10.set_ylim(THREADING_RANDOM_Y_LOWER, THREADING_RANDOM_Y_UPPER)
        ax10.set_title("")
        ax10.set_xlabel("Env Step (M)", fontsize=24, labelpad=10)
        ax10.set_ylabel("")
        ax10.yaxis.set_major_locator(MaxNLocator(4))
        ax10.tick_params(axis='both', labelsize=22)
        ax10.xaxis.get_offset_text().set_fontsize(18)
        ax10.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Create single legend at the bottom
        handles = [plt.Line2D([0], [0], color=COLOR_PALETTE[g], linewidth=5) for g in GROUP_ORDER]
        fig.legend(
            handles[::-1],
            GROUP_ORDER[::-1],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            fontsize=28,
            frameon=True
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10)

        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output}")

        plt.show()
    else:
        print("Failed to fetch data for the plots.")


# =============================================================================
# Mode: base_policy_ablation (Can task, multiple views, zero vs random sampling)
# =============================================================================

def run_base_policy_ablation(api, output):
    # --- Plot Configuration ---
    CAN_Y_LOWER = 0.1
    CAN_Y_UPPER = 1.0
    CAN_ALT_Y_LOWER = 0.92
    CAN_ALT_Y_UPPER = 1.0
    CAN_RANDOM_Y_LOWER = 0.0
    CAN_RANDOM_Y_UPPER = 0.7
    CAN_ALT_RANDOM_Y_LOWER = 0.55
    CAN_ALT_RANDOM_Y_UPPER = 0.95

    CAN_DOWNSAMPLE = 1

    CAN_SMOOTHING = 0.8

    COLOR_PALETTE = {
        "FPO++": "#00BFFF",        # Deep sky blue
        "Vanilla FPO": "#666666",  # Slate gray
        "DPPO - fixed noise": "#32CD32",          # Lime green
        "DPPO - learned noise": "#FF8C00",     # Dark orange
    }

    GROUP_ORDER = ["DPPO - learned noise", "DPPO - fixed noise", "Vanilla FPO", "FPO++"]

    # =============================================================================
    # CAN Configuration
    # =============================================================================
    can_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e0h4wy1r", "i6rmkgrh", "wbxzw7z3"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gj312c6y", "k2kvz5n2", "o84n7r75"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["3kdb60fl", "2ley356r", "482uupak"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e11pm661", "wt64jgp5", "sx1h2e3d"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.7135,
            "use_merge_asof": False
        },
    }

    can_configs_alt = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["txf5crib", "ylxhw9uu", "6zp46k32"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.9611,  # DUMMY_INITIAL_RATE
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gzm22fqh", "opygcs30", "89jg66ll"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.9611,  # DUMMY_INITIAL_RATE
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["7lz0maky", "ss13djab", "nqkx8w0s"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.9611,  # DUMMY_INITIAL_RATE
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4iz08o2h", "0wo72noe", "0yibse8t"],
            "metric_key": "eval/success_rate_zero_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.9611,  # DUMMY_INITIAL_RATE
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # CAN Random Sampling Configuration
    # =============================================================================
    can_random_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e0h4wy1r", "i6rmkgrh", "wbxzw7z3"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.1,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gj312c6y", "k2kvz5n2", "o84n7r75"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.1,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["3kdb60fl", "2ley356r", "482uupak"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.1,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e11pm661", "wt64jgp5", "sx1h2e3d"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.1,
            "use_merge_asof": False
        },
    }

    can_random_configs_alt = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["txf5crib", "ylxhw9uu", "6zp46k32"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6406,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gzm22fqh", "opygcs30", "89jg66ll"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6406,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["7lz0maky", "ss13djab", "nqkx8w0s"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6406,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4iz08o2h", "0wo72noe", "0yibse8t"],
            "metric_key": "eval/success_rate_random_sampling",
            "x_axis_key": "_step",
            "initial_rate": 0.6406,
            "use_merge_asof": False
        },
    }

    # =============================================================================
    # CAN Rewards Configuration
    # =============================================================================
    can_rewards_configs = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e0h4wy1r", "i6rmkgrh", "wbxzw7z3"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gj312c6y", "k2kvz5n2", "o84n7r75"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["3kdb60fl", "2ley356r", "482uupak"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["e11pm661", "wt64jgp5", "sx1h2e3d"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
    }

    can_rewards_configs_alt = {
        "FPO++": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["txf5crib", "ylxhw9uu", "6zp46k32"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
        "Vanilla FPO": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["gzm22fqh", "opygcs30", "89jg66ll"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
        "DPPO - learned noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["7lz0maky", "ss13djab", "nqkx8w0s"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
        "DPPO - fixed noise": {
            "project": "far-wandb/flow-bc-fpo-finetuning",
            "run_ids": ["4iz08o2h", "0wo72noe", "0yibse8t"],
            "metric_key": "charts/rewards",
            "x_axis_key": "_step",
            "initial_rate": None,
            "use_merge_asof": False
        },
    }

    # --- Fetch all data ---
    print("=" * 50)
    print("Fetching CAN Zero Sampling data...")
    print("=" * 50)
    can_df = fetch_and_process(api, can_configs, "CAN", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching CAN Zero Sampling data (alt)...")
    print("=" * 50)
    can_df_alt = fetch_and_process(api, can_configs_alt, "CAN (alt)", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching CAN Random Sampling data...")
    print("=" * 50)
    can_random_df = fetch_and_process(api, can_random_configs, "CAN Random", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching CAN Random Sampling data (alt)...")
    print("=" * 50)
    can_random_df_alt = fetch_and_process(api, can_random_configs_alt, "CAN Random (alt)", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching CAN Rewards data...")
    print("=" * 50)
    can_rewards_df = fetch_and_process(api, can_rewards_configs, "CAN Rewards", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    print("\n" + "=" * 50)
    print("Fetching CAN Rewards data (alt)...")
    print("=" * 50)
    can_rewards_df_alt = fetch_and_process(api, can_rewards_configs_alt, "CAN Rewards (alt)", CAN_DOWNSAMPLE, CAN_SMOOTHING)

    # --- Create combined figure ---
    if can_df is not None:
        print("\nGenerating combined plot...")

        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

        # --- Row 1: CAN ---
        # Plot CAN Zero Sampling
        sns.lineplot(
            data=can_df,
            x="total env step",
            y="Success Rate",
            hue="Group",
            hue_order=GROUP_ORDER,
            palette=COLOR_PALETTE,
            ci=95,
            linewidth=5,
            ax=ax1,
            legend=False
        )
        ax1.set_ylim(CAN_Y_LOWER, CAN_Y_UPPER)
        ax1.set_title("Zero Sampling", fontsize=24)
        ax1.set_xlabel("", fontsize=20)
        ax1.set_ylabel("Success Rate", fontsize=24, labelpad=10)
        ax1.yaxis.set_major_locator(MaxNLocator(4))
        ax1.tick_params(axis='both', labelsize=22)
        ax1.xaxis.get_offset_text().set_fontsize(18)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot CAN Random Sampling
        if can_random_df is not None:
            sns.lineplot(
                data=can_random_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax2,
                legend=False
            )
        ax2.set_ylim(CAN_RANDOM_Y_LOWER, CAN_RANDOM_Y_UPPER)
        ax2.set_title("Random Sampling", fontsize=24)
        ax2.set_xlabel("", fontsize=20)
        ax2.set_ylabel("", fontsize=24)
        ax2.yaxis.set_major_locator(MaxNLocator(4))
        ax2.tick_params(axis='both', labelsize=22)
        ax2.xaxis.get_offset_text().set_fontsize(18)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot CAN Rewards
        if can_rewards_df is not None:
            sns.lineplot(
                data=can_rewards_df,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax3,
                legend=False
            )
        ax3.set_title("", fontsize=24)
        ax3.set_xlabel("", fontsize=20)
        ax3.set_ylabel("Training rewards", fontsize=24, labelpad=20)
        ax3.yaxis.set_major_locator(MaxNLocator(4))
        ax3.tick_params(axis='both', labelsize=22)
        ax3.xaxis.get_offset_text().set_fontsize(18)
        ax3.grid(True, which="both", linestyle="--", linewidth=0.5)

        # --- Row 2: CAN Alt ---
        # Plot CAN Alt Zero Sampling
        if can_df_alt is not None:
            sns.lineplot(
                data=can_df_alt,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax4,
                legend=False
            )
        ax4.set_ylim(CAN_ALT_Y_LOWER, CAN_ALT_Y_UPPER)
        ax4.set_title("", fontsize=24)
        ax4.set_xlabel("Env Step (M)", fontsize=20)
        ax4.set_ylabel("Success Rate", fontsize=24, labelpad=10)
        ax4.yaxis.set_major_locator(MaxNLocator(4))
        ax4.tick_params(axis='both', labelsize=22)
        ax4.xaxis.get_offset_text().set_fontsize(18)
        ax4.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot CAN Alt Random Sampling
        if can_random_df_alt is not None:
            sns.lineplot(
                data=can_random_df_alt,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax5,
                legend=False
            )
        ax5.set_ylim(CAN_ALT_RANDOM_Y_LOWER, CAN_ALT_RANDOM_Y_UPPER)
        ax5.set_title("", fontsize=24)
        ax5.set_xlabel("Env Step (M)", fontsize=20)
        ax5.set_ylabel("", fontsize=24)
        ax5.yaxis.set_major_locator(MaxNLocator(4))
        ax5.tick_params(axis='both', labelsize=22)
        ax5.xaxis.get_offset_text().set_fontsize(18)
        ax5.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot CAN Alt Rewards
        if can_rewards_df_alt is not None:
            sns.lineplot(
                data=can_rewards_df_alt,
                x="total env step",
                y="Success Rate",
                hue="Group",
                hue_order=GROUP_ORDER,
                palette=COLOR_PALETTE,
                ci=95,
                linewidth=5,
                ax=ax6,
                legend=False
            )
        ax6.set_title("", fontsize=24)
        ax6.set_xlabel("Env Step (M)", fontsize=20)
        ax6.set_ylabel("Training rewards", fontsize=24, labelpad=20)
        ax6.yaxis.set_major_locator(MaxNLocator(4))
        ax6.tick_params(axis='both', labelsize=22)
        ax6.xaxis.get_offset_text().set_fontsize(18)
        ax6.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Create single legend at the bottom
        handles = [plt.Line2D([0], [0], color=COLOR_PALETTE[g], linewidth=5) for g in GROUP_ORDER]
        fig.legend(
            handles[::-1],
            GROUP_ORDER[::-1],
            loc='upper center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=4,
            fontsize=22,
            frameon=True
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output}")

        plt.show()
    else:
        print("Failed to fetch data for one or both plots.")


# =============================================================================
# Main entry point
# =============================================================================

DEFAULT_OUTPUTS = {
    "main_benchmark": "main_benchmark_plot.pdf",
    "fpoplusplus_ablation": "fpoplusplus_ablation_plot.pdf",
    "base_policy_ablation": "base_policy_ablation_plot.pdf",
}

MODE_FUNCTIONS = {
    "main_benchmark": run_main_benchmark,
    "fpoplusplus_ablation": run_fpoplusplus_ablation,
    "base_policy_ablation": run_base_policy_ablation,
}


def main():
    parser = argparse.ArgumentParser(
        description="Consolidated plotting script for FPO experiment results."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(MODE_FUNCTIONS.keys()),
        help="Plot mode: main_benchmark, fpoplusplus_ablation, or base_policy_ablation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default depends on mode)"
    )
    args = parser.parse_args()

    output = args.output if args.output else DEFAULT_OUTPUTS[args.mode]

    # Setup W&B API
    wandb.login()
    api = wandb.Api()

    # Run the selected mode
    MODE_FUNCTIONS[args.mode](api, output)


if __name__ == "__main__":
    main()
