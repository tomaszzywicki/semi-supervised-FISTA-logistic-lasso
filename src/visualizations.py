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

    ax.yaxis.grid(
        True, linestyle="--", color=COLORS["GRID"], alpha=ALPHA["GRID"], zorder=0
    )
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
                plot_data[
                    (plot_data["Scheme"] == scheme)
                    & (plot_data["Approach"] == approach)
                ][metric]
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
            boxprops=dict(
                facecolor=colors[i], color=colors[i], alpha=0.75, linewidth=1.5
            ),
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
    ax.set_xticklabels(
        schemes, rotation=35, ha="right", fontsize=11, color=COLORS["TICKS"]
    )

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
                mean_df[col].map("{:.1f}".format)
                + "% ± "
                + std_df[col].map("{:.1f}".format)
                + "%"
            )
        else:
            summary_df[col] = (
                mean_df[col].map("{:.3f}".format)
                + " ± "
                + std_df[col].map("{:.3f}".format)
            )

    return summary_df.reset_index()


