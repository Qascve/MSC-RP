#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
TEMP_COL = "temperature"
CLADE_COL = "class"
K_BOLTZMANN_EV_PER_K = 8.617e-5
LOG_TARGET = "log_BMR"

# m0-m2 keep the same variable choices as explore.py,
# m3 uses clade names directly (categorical class).
MODEL_FEATURES: dict[str, list[str]] = {
    "m0": ["log_mass"],
    "m1": ["log_mass"],
    "m2": ["log_mass", "inv_kT"],
    "m3": ["log_mass", "inv_kT", CLADE_COL],
    "m4": ["log_mass", "inv_kT", "pc1", "pc2", "pc3", "pc4", "pc5"],
}


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
    required = ["taxon_name", CLADE_COL, MASS_COL, TEMP_COL, "pc1", "pc2", "pc3", "pc4", "pc5", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip().replace("", pd.NA)
    out[CLADE_COL] = out[CLADE_COL].astype("string").str.strip().replace("", pd.NA)
    for col in [MASS_COL, TEMP_COL, "pc1", "pc2", "pc3", "pc4", "pc5", TARGET]:
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
    out[LOG_TARGET] = np.log(out[TARGET].to_numpy())
    return out


def build_feature_frames(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = train_df[feature_cols].reset_index(drop=True).copy()
    test_raw = test_df[feature_cols].reset_index(drop=True).copy()
    merged_raw = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)

    cat_cols = [c for c in feature_cols if c == CLADE_COL]
    merged_encoded = pd.get_dummies(merged_raw, columns=cat_cols, prefix=cat_cols, dtype=float)

    split_idx = len(train_raw)
    X_train = merged_encoded.iloc[:split_idx].copy()
    X_test = merged_encoded.iloc[split_idx:].copy()
    return X_train, X_test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    y_train_log = train_df[LOG_TARGET].to_numpy()
    y_true = test_df[TARGET].to_numpy()

    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float | str]] = []

    for spec_name, feature_cols in MODEL_FEATURES.items():
        X_train, X_test = build_feature_frames(train_df, test_df, feature_cols)

        rf = RandomForestRegressor(
            n_estimators=600,
            max_depth=4,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train_log)
        rf_name = f"random_forest_{spec_name}"
        rf_pred = np.exp(rf.predict(X_test))
        predictions[rf_name] = rf_pred
        metric_rows.append({"model": rf_name, **evaluate(y_true, rf_pred)})

        xgb = XGBRegressor(
           objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        )
        xgb.fit(X_train, y_train_log)
        xgb_name = f"xgboost_{spec_name}"
        xgb_pred = np.exp(xgb.predict(X_test))
        predictions[xgb_name] = xgb_pred
        metric_rows.append({"model": xgb_name, **evaluate(y_true, xgb_pred)})

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    return metrics_df, predictions, y_true


def save_model_performance_plot(metrics_df: pd.DataFrame, out_dir: Path) -> Path:
    plot_df = metrics_df.copy()
    sns.set_theme(style="whitegrid")
    fig_width = max(12.0, 0.8 * len(plot_df) + 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))

    sns.barplot(data=plot_df, x="model", y="rmse", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=plot_df, x="model", y="r2", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2")
    axes[1].tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.set_xlabel("")

    fig.suptitle("ML Model Performance Comparison (M0-M4)", fontsize=14)
    fig.tight_layout()

    output_path = out_dir / "explore_ml_model_performance_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_residual_plot(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    for model_name, y_pred in predictions.items():
        residual = y_true - y_pred
        plt.scatter(y_pred, residual, s=14, alpha=0.45, label=model_name)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Predicted BMR (W)")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("ML Residual Plot (M0-M4)")
    plt.legend()
    plt.tight_layout()
    output_path = out_dir / "explore_ml_residual_plot.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def save_prediction_table(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    pred_df = test_df[["taxon_name", CLADE_COL, "log_mass", "inv_kT"]].copy()
    pred_df["y_true"] = y_true
    for model_name, y_pred in predictions.items():
        pred_df[model_name] = y_pred

    output_path = out_dir / "explore_ml_predictions_test.csv"
    pred_df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Compare RandomForest and XGBoost under M0-M4 feature settings "
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
        help="Output directory for ML comparison results.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_path = _resolve_path(root, args.train)
    test_path = _resolve_path(root, args.test)
    out_dir = _resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = add_mte_features(load_split_data(train_path))
    test_df = add_mte_features(load_split_data(test_path))

    metrics_df, predictions, y_true = run_models(
        train_df=train_df,
        test_df=test_df,
        random_state=args.random_state,
    )

    metrics_path = out_dir / "explore_ml_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    pred_path = save_prediction_table(test_df, y_true, predictions, out_dir)
    plot_path = save_model_performance_plot(metrics_df, out_dir)
    residual_plot_path = save_residual_plot(y_true, predictions, out_dir)

    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved residual plot: {residual_plot_path}")
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
