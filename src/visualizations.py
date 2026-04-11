import ast
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
COLORS = {
    "GRID": "#DEE2E6",
    "ORACLE": "#E03C31",
    "LABEL": "black",
    "TICKS": "black",
    "TITLE": "black",
    "WHISKER": "black",
    "CAP": "black",
}
GRID_COLOR = "DEE2E6"
ALPHA = {"GRID": 0.7}

APPROACHES_MAP = {
    "naive": "Naive",
    "label_propagation": "Label propagation",
}

METRICS_MAP = {"Balanced_Acc": "Balanced accuracy", "ROC_AUC": "ROC AUC"}


def plot_experiment_results(df: pd.DataFrame, metric: str, dataset: str) -> None:
    oracle_data = df[df["Approach"] == "Oracle"]
    oracle_mean = oracle_data[metric].mean() if not oracle_data.empty else 0

    plot_data = df[df["Approach"] != "Oracle"]
    schemes = plot_data["Scheme"].unique()
    approaches = plot_data["Approach"].unique()

    n_schemes = len(schemes)
    n_approaches = len(approaches)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.yaxis.grid(True, linestyle="--", color=COLORS["GRID"], alpha=ALPHA["GRID"], zorder=0)
    ax.xaxis.grid(False)

    colors = COLOR_PALETTE[:n_approaches]

    width = 0.8 / n_approaches
    x_base = np.arange(n_schemes)

    legend_handles = []

    if not np.isnan(oracle_mean):
        ax.axhline(
            y=oracle_mean,
            color=COLORS["ORACLE"],
            linestyle="--",
            linewidth=2,
            zorder=4,
            alpha=0.7,
        )
        oracle_line = plt.Line2D(
            [0],
            [0],
            color=COLORS["ORACLE"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Oracle mean ({oracle_mean:.3f})",
        )
        legend_handles.append(oracle_line)

    for i, approach in enumerate(approaches):
        approach_data = []
        for scheme in schemes:
            vals = (
                plot_data[(plot_data["Scheme"] == scheme) & (plot_data["Approach"] == approach)][metric]
                .dropna()
                .values
            )
            if len(vals) == 0:
                vals = [np.nan]
            approach_data.append(vals)

        pos = x_base - (n_approaches * width) / 2 + (i + 0.5) * width

        bp = ax.boxplot(
            approach_data,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            zorder=3,
            boxprops=dict(facecolor=colors[i], color=colors[i], alpha=0.75, linewidth=1.5),
            capprops=dict(color=COLORS["CAP"], linewidth=1.5),
            whiskerprops=dict(color=COLORS["WHISKER"], linewidth=1.5, linestyle="-"),
            flierprops=dict(
                marker="o",
                markerfacecolor=colors[i],
                markersize=5,
                markeredgecolor="none",
                alpha=0.6,
            ),
            medianprops=dict(color="black", linewidth=1),
        )

        legend_handles.append(
            mpatches.Patch(
                color=colors[i],
                label=APPROACHES_MAP.get(approach, approach),
                alpha=0.75,
            )
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(schemes, rotation=35, ha="right", fontsize=11, color=COLORS["TICKS"])

    ax.set_ylabel(
        f"{METRICS_MAP.get(metric, metric)} (test data)",
        fontsize=12,
        color=COLORS["LABEL"],
    )
    ax.set_xlabel("Missing Data Mechanism", fontsize=12, color=COLORS["LABEL"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#adb5bd")
    ax.spines["bottom"].set_color("#adb5bd")

    fig.suptitle(
        f"Comparison of approaches - {METRICS_MAP.get(metric, metric)} (dataset: {dataset})",
        fontsize=14,
        color=COLORS["TITLE"],
        y=0.95,
    )

    ax.legend(
        handles=legend_handles,
        title="Approach",
        title_fontsize=12,
        loc="best",
    )

    plt.tight_layout()
    plt.show()


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    expected_metrics = ["Missing_Percent", "Accuracy", "Balanced_Acc", "F1", "ROC_AUC"]
    metrics = [m for m in expected_metrics if m in df.columns]

    grouped = df.groupby(["Scheme", "Approach"])[metrics]

    mean_df = grouped.mean()
    std_df = grouped.std().fillna(0)

    summary_df = pd.DataFrame(index=mean_df.index)

    for col in metrics:
        if col == "Missing_Percent":
            summary_df[col] = (
                mean_df[col].map("{:.1f}".format) + "% ± " + std_df[col].map("{:.1f}".format) + "%"
            )
        else:
            summary_df[col] = mean_df[col].map("{:.3f}".format) + " ± " + std_df[col].map("{:.3f}".format)

    return summary_df.reset_index()


def plot_sigma_results(results_df, dataset_name):
    n_seeds = results_df["Seed"].nunique()

    df_lp = results_df[results_df["Approach"] == "label_propagation"].copy()

    if isinstance(df_lp["Imputation_score"].iloc[0], str):
        df_lp["Imputation_score"] = df_lp["Imputation_score"].apply(ast.literal_eval)

    df_lp["Final_imputation_score"] = df_lp["Imputation_score"].apply(
        lambda x: x[-1] if len(x) > 0 else np.nan
    )

    schemes = [s for s in df_lp["Scheme"].unique() if s != "None"]
    sigmas = sorted(df_lp["sigma"].unique())
    metrics = ["Balanced_Acc", "ROC_AUC", "Final_imputation_score"]
    metric_labels = [
        "Balanced Accuracy (test)",
        "ROC AUC (test)",
        "Imputation score (final)",
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(schemes),
        figsize=(4 * len(schemes), 4 * len(metrics)),
        sharey="row",
    )

    for col_idx, scheme in enumerate(schemes):
        df_scheme = df_lp[df_lp["Scheme"] == scheme]

        for row_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row_idx, col_idx] if len(schemes) > 1 else axes[row_idx]

            means, stds = [], []
            for sigma in sigmas:
                vals = df_scheme[df_scheme["sigma"] == sigma][metric].dropna()
                means.append(vals.mean())
                stds.append(vals.std())

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(sigmas, means, marker="o", linewidth=2, color="#378ADD", zorder=3)
            ax.fill_between(sigmas, means - stds, means + stds, alpha=0.2, color="#378ADD")

            for sigma in sigmas:
                vals = df_scheme[df_scheme["sigma"] == sigma][metric].dropna().values
                ax.scatter([sigma] * len(vals), vals, color="gray", alpha=0.4, s=30, zorder=2)
            
            if row_idx == 0:
                ax.set_title(scheme, fontsize=16, pad=10)
                
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=14)

            ax.set_xlabel(r"$\sigma$", fontsize=15)
            
            ax.tick_params(axis='both', which='major', labelsize=16)

            ax.set_xscale("log")
            ax.grid(alpha=0.3)

    fig.suptitle(
        f"Label propagation results for different $\sigma$ values (Dataset: {dataset_name}) \n(averaged over {n_seeds} seeds $\pm$ 1SD)",
        fontsize=20,
        y=1.02
    )
    
    plt.tight_layout()
    plt.savefig("sigma_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()