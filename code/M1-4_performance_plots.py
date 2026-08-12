from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def save_grouped_metric_bars(
    models: list[str],
    metrics: dict[str, list[float]],
    output_path: Path,
    *,
    colors: list[str],
    figsize: tuple[float, float] = (8.2, 4.8),
    legend_ncols: int = 1,
) -> None:
    metric_names = list(metrics.keys())
    values = np.array([metrics[m] for m in metric_names], dtype=float)
    n_models = len(models)
    if n_models == 0:
        raise ValueError("No models to plot.")
    if values.shape != (len(metric_names), n_models):
        raise ValueError("metrics values must be shaped as [n_metrics, n_models].")
    if len(colors) < n_models:
        raise ValueError("Need at least one color per model.")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.linewidth": 1.2,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(metric_names))
    width = min(0.16, 0.8 / n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    for i, model in enumerate(models):
        ax.bar(
            x + offsets[i],
            values[:, i],
            width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_xlabel("Evaluation Metrics", fontsize=15)
    ax.set_ylabel("Performance", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1)
    ax.legend(
        loc="upper left",
        frameon=False,
        fancybox=False,
        edgecolor="black",
        fontsize=10,
        ncol=legend_ncols,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def build_metrics_dict(df: pd.DataFrame, models: list[str]) -> dict[str, list[float]]:
    by_model = df.set_index("model")
    missing = [m for m in models if m not in by_model.index]
    if missing:
        raise KeyError(f"Missing models in metrics table: {', '.join(missing)}")
    return {
        "RMSE": [float(by_model.loc[m, "rmse"]) for m in models],
        r"$R^2$": [float(by_model.loc[m, "r2"]) for m in models],
        "Balanced RMSE": [float(by_model.loc[m, "rmse_bal"]) for m in models],
        r"Balanced $R^2$": [float(by_model.loc[m, "r2_bal"]) for m in models],
    }


def main() -> None:
    root = find_root()
    out_dir = root / "results" / "plots"
    metrics_path = root / "results" / "explore" / "test" / "explore_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)

    # -----------------------------
    # Linear M1–M4 (+ M-MTE) bars
    # -----------------------------
    linear_models = ["M1-R", "M-MTE", "M2-R", "M3-R", "M4-R"]
    # Prefer explore_metrics for micro; bal for unweighted models may be empty,
    # so fall back to the dedicated linear comparison table when needed.
    linear_csv = root / "results" / "plots" / "m1_m4_linear_comparison.csv"
    if linear_csv.exists():
        linear_df = pd.read_csv(linear_csv)
    else:
        linear_df = metrics_df
    linear_metrics = build_metrics_dict(linear_df, linear_models)
    save_grouped_metric_bars(
        linear_models,
        linear_metrics,
        out_dir / "m1_m4_metric_bars.pdf",
        colors=["#9e0b2f", "#e8704d", "#f4c9b0", "#c8dceb", "#6fa7cf"],
    )

    # -----------------------------
    # M1–M4 XGB/RF + Residual XGB/RF
    # -----------------------------
    ml_models = [
        "M1-RF",
        "M1-XGB",
        "M2-RF",
        "M2-XGB",
        "M3-RF",
        "M3-XGB",
        "M4-RF",
        "M4-XGB",
        "Residual-RF",
        "Residual-XGB",
    ]
    ml_metrics = build_metrics_dict(metrics_df, ml_models)
    # Paired hues by model family; RF is always the darker shade.
    ml_colors = [
        "#6b081f",  # M1-RF  (dark red)
        "#e07a7a",  # M1-XGB (light red)
        "#a33a12",  # M2-RF  (dark orange)
        "#f0b48a",  # M2-XGB (light peach)
        "#2f5f7a",  # M3-RF  (dark blue)
        "#a8cce0",  # M3-XGB (light blue)
        "#3d4f8c",  # M4-RF  (dark indigo)
        "#b7c0e8",  # M4-XGB (light indigo)
        "#2f6b3a",  # Residual-RF  (dark green)
        "#9ccc9a",  # Residual-XGB (light green)
    ]
    save_grouped_metric_bars(
        ml_models,
        ml_metrics,
        out_dir / "m1_m4_ml_residual_metric_bars.pdf",
        colors=ml_colors,
        figsize=(11.5, 5.2),
        legend_ncols=2,
    )


if __name__ == "__main__":
    main()
