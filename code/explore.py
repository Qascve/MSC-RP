#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
TEMP_COL = "temperature"
CLADE_COL = "class"
K_BOLTZMANN_EV_PER_K = 8.617e-5


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["taxon_name", CLADE_COL, MASS_COL, TEMP_COL, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip().replace("", pd.NA)
    out[CLADE_COL] = out[CLADE_COL].astype("string").str.strip().replace("", pd.NA)
    for col in [MASS_COL, TEMP_COL, TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).copy()
    out = out[(out[MASS_COL] > 0) & (out[TARGET] > 0)].copy()
    if out.empty:
        raise ValueError(f"No valid rows left after cleaning: {path}")
    return out.reset_index(drop=True)


def add_mte_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_K"] = out[TEMP_COL] + 273.15
    out["inv_kT"] = 1.0 / (K_BOLTZMANN_EV_PER_K * out["temp_K"])
    out["log_mass"] = np.log(out[MASS_COL].to_numpy())
    out["log_BMR"] = np.log(out[TARGET].to_numpy())
    return out


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict_ols(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return X @ coef


def build_design_m3(df: pd.DataFrame, clade_levels: list[str]) -> tuple[np.ndarray, list[str]]:
    baseline = clade_levels[0]
    x_log_mass = df["log_mass"].to_numpy()
    x_inv_kT = df["inv_kT"].to_numpy()

    columns = [
        np.ones(len(df), dtype=float),
        x_log_mass,
        x_inv_kT,
    ]
    names = ["Intercept", "log_mass", "inv_kT"]

    for clade in clade_levels[1:]:
        d = (df[CLADE_COL] == clade).to_numpy(dtype=float)
        columns.extend([d, d * x_log_mass, d * x_inv_kT])
        names.extend(
            [
                f"clade[{clade}]",
                f"log_mass:clade[{clade}]",
                f"inv_kT:clade[{clade}]",
            ]
        )

    X = np.column_stack(columns)
    names[0] = f"Intercept (baseline={baseline})"
    return X, names


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    y_train_log = train_df["log_BMR"].to_numpy()
    y_test_log = test_df["log_BMR"].to_numpy()

    # m0: log_BMR ~ offset(0.75 * log_mass)
    alpha_m0 = float(np.mean(y_train_log - 0.75 * train_df["log_mass"].to_numpy()))
    yhat_m0_log = alpha_m0 + 0.75 * test_df["log_mass"].to_numpy()

    # m1: log_BMR ~ log_mass
    X1_train = np.column_stack([np.ones(len(train_df)), train_df["log_mass"].to_numpy()])
    X1_test = np.column_stack([np.ones(len(test_df)), test_df["log_mass"].to_numpy()])
    coef_m1 = fit_ols(X1_train, y_train_log)
    yhat_m1_log = predict_ols(X1_test, coef_m1)

    # m2: log_BMR ~ log_mass + inv_kT
    X2_train = np.column_stack(
        [
            np.ones(len(train_df)),
            train_df["log_mass"].to_numpy(),
            train_df["inv_kT"].to_numpy(),
        ]
    )
    X2_test = np.column_stack(
        [
            np.ones(len(test_df)),
            test_df["log_mass"].to_numpy(),
            test_df["inv_kT"].to_numpy(),
        ]
    )
    coef_m2 = fit_ols(X2_train, y_train_log)
    yhat_m2_log = predict_ols(X2_test, coef_m2)

    # m3: log_BMR ~ (log_mass + inv_kT) * clade
    clade_levels = sorted(train_df[CLADE_COL].dropna().unique().tolist())
    if not clade_levels:
        raise ValueError("No clade levels available in train data.")

    known_mask = test_df[CLADE_COL].isin(clade_levels)
    if not bool(known_mask.all()):
        dropped_n = int((~known_mask).sum())
        print(
            f"Warning: dropped {dropped_n} test rows with unseen clade values for m3."
        )
    test_df_m3 = test_df[known_mask].copy()

    X3_train, names_m3 = build_design_m3(train_df, clade_levels)
    X3_test, _ = build_design_m3(test_df_m3, clade_levels)
    coef_m3 = fit_ols(X3_train, y_train_log)
    yhat_m3_log = predict_ols(X3_test, coef_m3)

    metric_rows = []
    y_true = test_df[TARGET].to_numpy()
    metric_rows.append(
        {
            "model": "m0_fixed_b_3_4",
            **evaluate(y_true, np.exp(yhat_m0_log)),
        }
    )
    metric_rows.append(
        {
            "model": "m1_estimated_b",
            **evaluate(y_true, np.exp(yhat_m1_log)),
        }
    )
    metric_rows.append(
        {
            "model": "m2_baseline_mte",
            **evaluate(y_true, np.exp(yhat_m2_log)),
        }
    )

    if len(test_df_m3) > 0:
        y_true_m3 = test_df_m3[TARGET].to_numpy()
        y_pred_m3 = np.exp(yhat_m3_log)
        metric_rows.append({"model": "m3_clade_specific_mte", **evaluate(y_true_m3, y_pred_m3)})

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)

    return metrics_df


def save_model_performance_plot(metrics_df: pd.DataFrame, out_dir: Path) -> Path:
    plot_df = metrics_df.copy()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=plot_df, x="model", y="rmse", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=plot_df, x="model", y="mae", ax=axes[1], color="#55A868")
    axes[1].set_title("MAE")
    axes[1].tick_params(axis="x", rotation=20)

    sns.barplot(data=plot_df, x="model", y="r2", ax=axes[2], color="#C44E52")
    axes[2].set_title("R2")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.set_xlabel("")

    fig.suptitle("Model Performance Comparison", fontsize=14)
    fig.tight_layout()

    output_path = out_dir / "model_performance_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Fit four MTE-style linear models equivalent to R formulas m0-m3 "
            "using stratified train/test splits."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/stratified/train.csv"),
        help="Train CSV path.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/splits/stratified/test.csv"),
        help="Test CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/explore"),
        help="Output directory for metrics and plots.",
    )
    args = parser.parse_args()

    train_path = _resolve_path(root, args.train)
    test_path = _resolve_path(root, args.test)
    out_dir = _resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = add_mte_features(load_split_data(train_path))
    test_df = add_mte_features(load_split_data(test_path))

    metrics_df = run_models(train_df, test_df)
    plot_path = save_model_performance_plot(metrics_df, out_dir)

    metrics_path = out_dir / "explore_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {plot_path}")
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
