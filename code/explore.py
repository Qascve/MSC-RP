#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

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
ML_MODEL_SUFFIXES = ("m0", "m1", "m2", "m3", "m4")
LINEAR_NAME_MAP = {
    "m0_fixed_b_3_4": "M0-L",
    "m1_estimated_b": "M1-L",
    "m2_baseline_mte": "M2-L",
    "m3_clade_specific_mte": "M3-L",
    "m4_pglmm_phyr_mte": "M4-L",
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
    x_log_mass = df["log_mass"].to_numpy()
    x_inv_kT = df["inv_kT"].to_numpy()
    clade_dummies = pd.get_dummies(df[CLADE_COL], dtype=float).reindex(
        columns=clade_levels[1:],
        fill_value=0.0,
    )

    X = np.column_stack(
        [
            np.ones(len(df), dtype=float),
            x_log_mass,
            x_inv_kT,
            clade_dummies.to_numpy(dtype=float),
        ]
    )
    names = [
        "Intercept",
        "log_mass",
        "inv_kT",
        *[f"{CLADE_COL}[T.{clade}]" for clade in clade_levels[1:]],
    ]
    return X, names


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def to_short_model_name(model_name: str) -> str:
    if model_name in LINEAR_NAME_MAP:
        return LINEAR_NAME_MAP[model_name]
    if model_name.startswith("random_forest_"):
        suffix = model_name.replace("random_forest_", "", 1)
        return f"{suffix.upper()}-RF"
    if model_name.startswith("xgboost_"):
        suffix = model_name.replace("xgboost_", "", 1)
        return f"{suffix.upper()}-XGB"
    return model_name


def load_benchmark_predictions(path: Path, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    pred_df = pd.read_csv(path)
    benchmark_cols = [
        c
        for c in pred_df.columns
        if (c.startswith("random_forest_") or c.startswith("xgboost_"))
        and c.endswith(ML_MODEL_SUFFIXES)
    ]
    benchmark_cols = sorted(dict.fromkeys(benchmark_cols))
    required = ["y_true", *benchmark_cols]
    missing = [c for c in ["y_true"] if c not in pred_df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")
    if len(benchmark_cols) == 0:
        raise KeyError(
            f"{path.name} missing ML model columns like random_forest_m0..m4/xgboost_m0..m4."
        )

    for col in required:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    pred_df = pred_df.dropna(subset=required).reset_index(drop=True)

    y_true = test_df[TARGET].to_numpy()
    if len(pred_df) != len(test_df):
        raise ValueError(
            "Benchmark predictions row count mismatches test split. "
            "Please rerun benchmark on the same train/test files."
        )

    if not np.allclose(pred_df["y_true"].to_numpy(), y_true, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "Benchmark predictions y_true is not aligned with current test split. "
            "Please rerun benchmark on the same test file."
        )

    return {col: pred_df[col].to_numpy(dtype=float) for col in benchmark_cols}


def load_residual_learning_predictions(path: Path, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    pred_df = pd.read_csv(path)
    required = ["y_true", "random_forest", "xgboost"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    for col in required:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    pred_df = pred_df.dropna(subset=required).reset_index(drop=True)

    y_true = test_df[TARGET].to_numpy()
    if len(pred_df) != len(test_df):
        raise ValueError(
            "Residual-learning predictions row count mismatches test split. "
            "Please rerun ml_residual_learning.py on the same train/test files."
        )
    if not np.allclose(pred_df["y_true"].to_numpy(), y_true, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "Residual-learning predictions y_true is not aligned with current test split. "
            "Please rerun ml_residual_learning.py on the same test file."
        )

    return {
        "Residual-RF": pred_df["random_forest"].to_numpy(dtype=float),
        "Residual-XGB": pred_df["xgboost"].to_numpy(dtype=float),
    }


def run_pglmm_with_phyr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    root: Path,
    train_path: Path,
    test_path: Path,
    tree_path: Path,
    r_script_path: Path,
    out_dir: Path,
) -> np.ndarray:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found in PATH. Please install R and make sure Rscript is available."
        )

    resolved_tree_path = _resolve_path(root, tree_path)
    resolved_r_script_path = _resolve_path(root, r_script_path)
    resolved_out_dir = _resolve_path(root, out_dir)
    if not resolved_tree_path.exists():
        raise FileNotFoundError(f"Phylogeny tree file not found: {resolved_tree_path}")
    if not resolved_r_script_path.exists():
        raise FileNotFoundError(f"PGLMM R script not found: {resolved_r_script_path}")

    cmd = [
        rscript,
        str(resolved_r_script_path),
        "--train",
        str(train_path),
        "--test",
        str(test_path),
        "--tree",
        str(resolved_tree_path),
        "--out-dir",
        str(resolved_out_dir),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "PGLMM (phyr) failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    pred_csv = resolved_out_dir / "pglmm_test_predictions.csv"
    if not pred_csv.exists():
        raise RuntimeError("PGLMM output file was not generated.")

    pred_df = pd.read_csv(pred_csv)

    required = ["row_id", "taxon_name", TARGET, "y_pred_BMR"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise KeyError(f"PGLMM output missing required columns: {', '.join(missing)}")

    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="coerce")
    pred_df[TARGET] = pd.to_numeric(pred_df[TARGET], errors="coerce")
    pred_df["y_pred_BMR"] = pd.to_numeric(pred_df["y_pred_BMR"], errors="coerce")
    pred_df = pred_df.dropna(subset=["row_id"]).copy()
    pred_df["row_id"] = pred_df["row_id"].astype(int)
    pred_df = pred_df.sort_values("row_id").reset_index(drop=True)

    if len(pred_df) != len(test_df):
        raise ValueError(
            "PGLMM output row count mismatch with test data. "
            "Please rerun pglmm_phyr.R on the same test split."
        )
    expected_row_ids = np.arange(len(test_df), dtype=int)
    if not np.array_equal(pred_df["row_id"].to_numpy(), expected_row_ids):
        raise ValueError("PGLMM output row_id is not aligned with test data order.")
    if not np.array_equal(
        pred_df["taxon_name"].astype("string").to_numpy(),
        test_df["taxon_name"].astype("string").to_numpy(),
    ):
        raise ValueError("PGLMM output taxon_name is not aligned with test data order.")
    if not np.allclose(
        pred_df[TARGET].to_numpy(dtype=float),
        test_df[TARGET].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("PGLMM output BMR is not aligned with test data order.")

    return pred_df["y_pred_BMR"].to_numpy(dtype=float)


def run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    benchmark_predictions: dict[str, np.ndarray],
    pglmm_predictions: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    y_train_log = train_df["log_BMR"].to_numpy()

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

    # m3: log_BMR ~ log_mass + inv_kT + class
    # class is treatment-coded as a categorical predictor.
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

    y_true = test_df[TARGET].to_numpy()
    predictions: dict[str, np.ndarray] = {
        "m0_fixed_b_3_4": np.exp(yhat_m0_log),
        "m1_estimated_b": np.exp(yhat_m1_log),
        "m2_baseline_mte": np.exp(yhat_m2_log),
        "m4_pglmm_phyr_mte": pglmm_predictions,
        **benchmark_predictions,
    }

    metric_rows = []
    y_true = test_df[TARGET].to_numpy()
    metric_rows.append(
        {
            "model": "m0_fixed_b_3_4",
            **evaluate(y_true, predictions["m0_fixed_b_3_4"]),
        }
    )
    metric_rows.append(
        {
            "model": "m1_estimated_b",
            **evaluate(y_true, predictions["m1_estimated_b"]),
        }
    )
    metric_rows.append(
        {
            "model": "m2_baseline_mte",
            **evaluate(y_true, predictions["m2_baseline_mte"]),
        }
    )

    if len(test_df_m3) > 0:
        y_true_m3 = test_df_m3[TARGET].to_numpy()
        y_pred_m3 = np.exp(yhat_m3_log)
        y_pred_m3_full = np.full(len(test_df), np.nan, dtype=float)
        y_pred_m3_full[known_mask.to_numpy()] = y_pred_m3
        predictions["m3_clade_specific_mte"] = y_pred_m3_full
        metric_rows.append({"model": "m3_clade_specific_mte", **evaluate(y_true_m3, y_pred_m3)})

    pglmm_mask = ~np.isnan(predictions["m4_pglmm_phyr_mte"])
    if bool(pglmm_mask.any()):
        metric_rows.append(
            {
                "model": "m4_pglmm_phyr_mte",
                **evaluate(y_true[pglmm_mask], predictions["m4_pglmm_phyr_mte"][pglmm_mask]),
            }
        )
    for model_name in sorted(benchmark_predictions):
        metric_rows.append({"model": model_name, **evaluate(y_true, predictions[model_name])})

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    metrics_df["model"] = metrics_df["model"].map(to_short_model_name)
    predictions_short = {to_short_model_name(k): v for k, v in predictions.items()}

    return metrics_df, predictions_short, y_true


def save_model_performance_plot(metrics_df: pd.DataFrame, out_dir: Path) -> Path:
    plot_df = metrics_df.copy()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=plot_df, x="model", y="rmse", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=plot_df, x="model", y="r2", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2")
    axes[1].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.set_xlabel("")

    fig.suptitle("Model Performance Comparison", fontsize=14)
    fig.tight_layout()

    output_path = out_dir / "model_performance_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_top5_plus_residual_learning_plot(
    metrics_df: pd.DataFrame,
    y_true: np.ndarray,
    residual_learning_predictions: dict[str, np.ndarray],
    out_dir: Path,
) -> tuple[Path, Path]:
    top5_df = metrics_df.sort_values("rmse").head(5).copy()
    residual_rows = [
        {"model": model_name, **evaluate(y_true, y_pred)}
        for model_name, y_pred in residual_learning_predictions.items()
    ]
    plot_df = pd.concat([top5_df, pd.DataFrame(residual_rows)], ignore_index=True)
    plot_df = plot_df.drop_duplicates(subset=["model"], keep="first").reset_index(drop=True)
    plot_df = plot_df.sort_values("rmse").reset_index(drop=True)
    model_order = plot_df["model"].tolist()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.barplot(
        data=plot_df,
        x="model",
        y="rmse",
        order=model_order,
        ax=axes[0],
        color="#4C72B0",
    )
    axes[0].set_title("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=plot_df,
        x="model",
        y="r2",
        order=model_order,
        ax=axes[1],
        color="#C44E52",
    )
    axes[1].set_title("R2")
    axes[1].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.set_xlabel("")

    fig.suptitle("Top 5 Models + Residual Learning (All)", fontsize=14)
    fig.tight_layout()

    plot_path = out_dir / "top5_plus_residual_learning_performance.png"
    data_path = out_dir / "top5_plus_residual_learning_metrics.csv"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plot_df.to_csv(data_path, index=False, encoding="utf-8")
    return plot_path, data_path


def save_residual_plot(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 7))
    for model_name, y_pred in predictions.items():
        mask = ~np.isnan(y_pred)
        pred_x = y_pred[mask]
        residual = y_true[mask] - y_pred[mask]
        plt.scatter(pred_x, residual, s=14, alpha=0.45, label=model_name)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Predicted BMR (W)")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("Residual Plot")
    plt.legend()
    plt.tight_layout()
    output_path = out_dir / "residual_plot_all_models.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Fit MTE-style models m0-m3 plus a phyr::pglmm model "
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
    parser.add_argument(
        "--benchmark-predictions",
        type=Path,
        default=Path("results/explore/explore_ml_predictions_test.csv"),
        help="Prediction CSV path containing random_forest_m0..m4 and xgboost_m0..m4 outputs.",
    )
    parser.add_argument(
        "--residual-learning-predictions",
        type=Path,
        default=Path("results/benchmark/all/benchmark_predictions_test.csv"),
        help="Residual-learning all-dataset prediction CSV with random_forest and xgboost outputs.",
    )
    parser.add_argument(
        "--phylo-tree",
        type=Path,
        default=Path("data/phylogeny/unique_taxon_names.nwk"),
        help="Newick tree file used by phyr::pglmm.",
    )
    parser.add_argument(
        "--pglmm-r-script",
        type=Path,
        default=Path("code/pglmm_phyr.R"),
        help="R script path for fitting phyr::pglmm and predicting test data.",
    )
    parser.add_argument(
        "--pglmm-output-dir",
        type=Path,
        default=Path("results/pglmm_phyr"),
        help="Output directory used by pglmm_phyr.R.",
    )
    args = parser.parse_args()

    train_path = _resolve_path(root, args.train)
    test_path = _resolve_path(root, args.test)
    out_dir = _resolve_path(root, args.output_dir)
    benchmark_pred_path = _resolve_path(root, args.benchmark_predictions)
    residual_learning_pred_path = _resolve_path(root, args.residual_learning_predictions)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = add_mte_features(load_split_data(train_path))
    test_df = add_mte_features(load_split_data(test_path))
    benchmark_predictions = load_benchmark_predictions(benchmark_pred_path, test_df)
    residual_learning_predictions = load_residual_learning_predictions(
        residual_learning_pred_path,
        test_df,
    )
    pglmm_predictions = run_pglmm_with_phyr(
        train_df=train_df,
        test_df=test_df,
        root=root,
        train_path=train_path,
        test_path=test_path,
        tree_path=args.phylo_tree,
        r_script_path=args.pglmm_r_script,
        out_dir=args.pglmm_output_dir,
    )

    metrics_df, predictions, y_true = run_models(
        train_df,
        test_df,
        benchmark_predictions,
        pglmm_predictions,
    )
    plot_path = save_model_performance_plot(metrics_df, out_dir)
    top5_residual_plot_path, top5_residual_metrics_path = save_top5_plus_residual_learning_plot(
        metrics_df,
        y_true,
        residual_learning_predictions,
        out_dir,
    )
    residual_plot_path = save_residual_plot(y_true, predictions, out_dir)

    metrics_path = out_dir / "explore_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved top-5 + residual-learning plot: {top5_residual_plot_path}")
    print(f"Saved top-5 + residual-learning metrics: {top5_residual_metrics_path}")
    print(f"Saved residual plot: {residual_plot_path}")
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
