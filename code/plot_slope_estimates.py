#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
TEMP_COL = "temperature"
CLADE_COL = "class"
LOG_TARGET = "log_BMR"
K_BOLTZMANN_EV_PER_K = 8.617e-5
MTE_FIXED_B = 0.75
REFERENCE_SLOPE = MTE_FIXED_B
CI_Z = 1.96
# Display names follow Table 1: Mn-R / Mn-RF / Mn-XGB.
# M-MTE is the fixed-b (3/4) version of M2: log_BMR ~ inv_kT + offset(0.75*log_mass).
M_MTE_LABEL = "M-MTE"
M1_L_LABEL = "M1-R"
M2_L_LABEL = "M2-R"
M3_L_LABEL = "M3-R"
M4_L_LABEL = "M4-R"
M3_L_SHORT_LABEL = "M3-R"
M3_XGB_LABEL = "M3-XGB"
M4_XGB_LABEL = "M4-XGB"
M3_RF_LABEL = "M3-RF"
M4_RF_LABEL = "M4-RF"
RESIDUAL_XGB_LABEL = "Residual-XGB"
RESIDUAL_RF_LABEL = "Residual-RF"
PHYLO_PC_COLS = ["pc1", "pc2", "pc3", "pc4", "pc5"]

# Default matplotlib sizes + 2, all bold.
FONT_SIZE = 12
TITLE_SIZE = 14
SUPTITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 12
LEGEND_SIZE = 12


def apply_bold_fonts(ax: Axes, *, title_size: int = TITLE_SIZE) -> None:
    ax.set_title(ax.get_title(), fontsize=title_size, fontweight="bold")
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_SIZE)
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)
            text.set_fontweight("bold")
        title = legend.get_title()
        if title is not None and title.get_text():
            title.set_fontsize(LEGEND_SIZE)
            title.set_fontweight("bold")


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [CLADE_COL, MASS_COL, TARGET, TEMP_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df.copy()
    out[CLADE_COL] = out[CLADE_COL].astype("string").str.strip()
    out[MASS_COL] = pd.to_numeric(out[MASS_COL], errors="coerce")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out[TEMP_COL] = pd.to_numeric(out[TEMP_COL], errors="coerce")
    out = out.dropna(subset=[CLADE_COL, MASS_COL, TARGET, TEMP_COL]).copy()
    out = out[(out[MASS_COL] > 0) & (out[TARGET] > 0)].copy()
    out["temp_K"] = out[TEMP_COL] + 273.15
    out["inv_kT"] = 1.0 / (K_BOLTZMANN_EV_PER_K * out["temp_K"])
    out["log_mass"] = np.log10(out[MASS_COL].to_numpy(dtype=float))
    out[LOG_TARGET] = np.log10(out[TARGET].to_numpy(dtype=float))
    return out.reset_index(drop=True)


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = [CLADE_COL, MASS_COL, TARGET, TEMP_COL, *PHYLO_PC_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df.copy()
    out[CLADE_COL] = out[CLADE_COL].astype("string").str.strip()
    out[MASS_COL] = pd.to_numeric(out[MASS_COL], errors="coerce")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out[TEMP_COL] = pd.to_numeric(out[TEMP_COL], errors="coerce")
    for col in PHYLO_PC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).copy()
    out = out[(out[MASS_COL] > 0) & (out[TARGET] > 0)].copy()
    out["temp_K"] = out[TEMP_COL] + 273.15
    out["inv_kT"] = 1.0 / (K_BOLTZMANN_EV_PER_K * out["temp_K"])
    out["log_mass"] = np.log10(out[MASS_COL].to_numpy(dtype=float))
    out[LOG_TARGET] = np.log10(out[TARGET].to_numpy(dtype=float))
    return out.reset_index(drop=True)


def build_design_m3(df: pd.DataFrame, clade_levels: list[str]) -> np.ndarray:
    clade_dummies = pd.get_dummies(df[CLADE_COL], dtype=float).reindex(
        columns=clade_levels[1:],
        fill_value=0.0,
    )
    return np.column_stack(
        [
            np.ones(len(df), dtype=float),
            df["log_mass"].to_numpy(dtype=float),
            df["inv_kT"].to_numpy(dtype=float),
            clade_dummies.to_numpy(dtype=float),
        ]
    )


def build_design_m4(df: pd.DataFrame, pc_cols: list[str] | None = None) -> np.ndarray:
    pc_cols = pc_cols or PHYLO_PC_COLS
    x_log_mass = df["log_mass"].to_numpy(dtype=float)
    x_inv_kT = df["inv_kT"].to_numpy(dtype=float)

    blocks = [np.ones(len(df), dtype=float), x_log_mass, x_inv_kT]
    for pc in pc_cols:
        blocks.append(df[pc].to_numpy(dtype=float))
    for pc in pc_cols:
        x_pc = df[pc].to_numpy(dtype=float)
        blocks.append(x_log_mass * x_pc)
        blocks.append(x_inv_kT * x_pc)
    return np.column_stack(blocks)


def predict_m3_linear(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)
    clade_levels = sorted(train_df[CLADE_COL].dropna().unique().tolist())
    if not clade_levels:
        raise ValueError("No clade levels available in train data.")

    known_mask = test_df[CLADE_COL].isin(clade_levels).to_numpy()
    pred = np.full(len(test_df), np.nan, dtype=float)
    if bool(known_mask.any()):
        test_known = test_df.loc[known_mask].copy()
        X3_train = build_design_m3(train_df, clade_levels)
        X3_test = build_design_m3(test_known, clade_levels)
        pred[known_mask] = predict_ols(X3_test, fit_ols(X3_train, y_train))
    return pred


def load_prediction_column(
    path: Path,
    column: str,
    y_true: np.ndarray,
    split_label: str = "test",
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    pred_df = pd.read_csv(path)
    if "y_true" not in pred_df.columns:
        raise KeyError(f"{path.name} missing required column: y_true")
    if column not in pred_df.columns:
        raise KeyError(f"{path.name} missing required column: {column}")

    pred_df["y_true"] = pd.to_numeric(pred_df["y_true"], errors="coerce")
    pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")
    pred_df = pred_df.dropna(subset=["y_true", column]).reset_index(drop=True)

    y_true = np.asarray(y_true, dtype=float)
    if len(pred_df) != len(y_true):
        raise ValueError(
            f"{path.name} row count mismatches {split_label} split "
            f"({len(pred_df)} rows vs {len(y_true)})."
        )
    if not np.allclose(pred_df["y_true"].to_numpy(), y_true, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"{path.name} y_true is not aligned with current {split_label} split."
        )
    return pred_df[column].to_numpy(dtype=float)


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict_ols(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return X @ coef


def evaluate_log_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = np.asarray(y_true_log, dtype=float)[mask]
    y_pred_log = np.asarray(y_pred_log, dtype=float)[mask]
    if len(y_true_log) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2": float(r2_score(y_true_log, y_pred_log)),
    }


def evaluate_weighted(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    # Class-balanced weighted RMSE/MAE/R2 on log10(BMR).
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    sw = np.asarray(sample_weight, dtype=float)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log) & np.isfinite(sw) & (sw > 0)
    y_true_log = y_true_log[mask]
    y_pred_log = y_pred_log[mask]
    sw = sw[mask]
    if len(y_true_log) == 0:
        return {"rmse_bal": np.nan, "mae_bal": np.nan, "r2_bal": np.nan}
    resid = y_true_log - y_pred_log
    w_sum = float(np.sum(sw))
    rmse_v = float(np.sqrt(np.sum(sw * resid**2) / w_sum))
    mae = float(np.sum(sw * np.abs(resid)) / w_sum)
    y_bar = float(np.sum(sw * y_true_log) / w_sum)
    ss_tot = float(np.sum(sw * (y_true_log - y_bar) ** 2))
    ss_res = float(np.sum(sw * resid**2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {"rmse_bal": rmse_v, "mae_bal": mae, "r2_bal": r2}


def evaluate_macro_by_class(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    # Unweighted mean of per-class micro metrics (each class counts equally).
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    classes = np.asarray(classes)
    rows = []
    for cls in sorted(pd.unique(classes)):
        mask = classes == cls
        if not bool(np.any(mask)):
            continue
        rows.append(evaluate_log_predictions(y_true_log[mask], y_pred_log[mask]))
    if not rows:
        return {
            "rmse_macro": np.nan,
            "mae_macro": np.nan,
            "r2_macro": np.nan,
            "n_classes_eval": 0,
        }
    return {
        "rmse_macro": float(np.nanmean([r["rmse"] for r in rows])),
        "mae_macro": float(np.nanmean([r["mae"] for r in rows])),
        "r2_macro": float(np.nanmean([r["r2"] for r in rows])),
        "n_classes_eval": int(len(rows)),
    }


def evaluate_reporting_suite(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    # Micro, macro, and class-balanced weighted metrics (CSV only; plots use micro).
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    classes = np.asarray(classes)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = y_true_log[mask]
    y_pred_log = y_pred_log[mask]
    classes = classes[mask]
    micro = evaluate_log_predictions(y_true_log, y_pred_log)
    macro = evaluate_macro_by_class(y_true_log, y_pred_log, classes)
    if len(classes) == 0:
        bal = {"rmse_bal": np.nan, "mae_bal": np.nan, "r2_bal": np.nan}
    else:
        unique = np.unique(classes)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique,
            y=classes,
        )
        weight_map = dict(zip(unique, class_weights))
        sw = np.array([weight_map[c] for c in classes], dtype=float)
        bal = evaluate_weighted(y_true_log, y_pred_log, sw)
    return {**micro, **macro, **bal}


def fit_m1_m2_on_train(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Fit M1-R and M2-R on train; return coefficients for each model.
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)
    X1 = np.column_stack([np.ones(len(train_df)), train_df["log_mass"].to_numpy()])
    X2 = np.column_stack(
        [
            np.ones(len(train_df)),
            train_df["log_mass"].to_numpy(),
            train_df["inv_kT"].to_numpy(),
        ]
    )
    return fit_ols(X1, y_train), fit_ols(X2, y_train)


def predict_m1_m2(
    test_df: pd.DataFrame, coef_m1: np.ndarray, coef_m2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X1 = np.column_stack([np.ones(len(test_df)), test_df["log_mass"].to_numpy()])
    X2 = np.column_stack(
        [
            np.ones(len(test_df)),
            test_df["log_mass"].to_numpy(),
            test_df["inv_kT"].to_numpy(),
        ]
    )
    return predict_ols(X1, coef_m1), predict_ols(X2, coef_m2)


def predict_m_mte(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    # Fixed-b MTE (M2 with mass slope locked at 3/4).
    #
    # Equivalent to: log_BMR ~ inv_kT + offset(0.75 * log_mass)
    #
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)
    log_mass_train = train_df["log_mass"].to_numpy(dtype=float)
    log_mass_test = test_df["log_mass"].to_numpy(dtype=float)
    y_adj = y_train - MTE_FIXED_B * log_mass_train
    X_train = np.column_stack([np.ones(len(train_df)), train_df["inv_kT"].to_numpy(dtype=float)])
    X_test = np.column_stack([np.ones(len(test_df)), test_df["inv_kT"].to_numpy(dtype=float)])
    coef = fit_ols(X_train, y_adj)
    return MTE_FIXED_B * log_mass_test + predict_ols(X_test, coef)


def build_m1_m4_linear_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    # M-MTE + M1–M4 (estimated-slope) models on the held-out test set.
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)
    y_test = test_df[LOG_TARGET].to_numpy(dtype=float)
    classes = test_df[CLADE_COL].to_numpy()
    pred_mte = predict_m_mte(train_df, test_df)
    coef_m1, coef_m2 = fit_m1_m2_on_train(train_df)
    pred_m1, pred_m2 = predict_m1_m2(test_df, coef_m1, coef_m2)
    pred_m3 = predict_m3_linear(train_df, test_df)
    pred_m4 = predict_ols(build_design_m4(test_df), fit_ols(build_design_m4(train_df), y_train))
    rows = [
        {"model": M1_L_LABEL, **evaluate_reporting_suite(y_test, pred_m1, classes)},
        {"model": M_MTE_LABEL, **evaluate_reporting_suite(y_test, pred_mte, classes)},
        {"model": M2_L_LABEL, **evaluate_reporting_suite(y_test, pred_m2, classes)},
        {"model": M3_L_LABEL, **evaluate_reporting_suite(y_test, pred_m3, classes)},
        {"model": M4_L_LABEL, **evaluate_reporting_suite(y_test, pred_m4, classes)},
    ]
    return pd.DataFrame(rows)


def build_ml_residual_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    explore_ml_predictions_path: Path,
    residual_predictions_path: Path,
) -> pd.DataFrame:
    # M-MTE + M3/M4 RF/XGB + Residual-RF/XGB on the held-out test set.
    y_test = test_df[LOG_TARGET].to_numpy(dtype=float)
    classes = test_df[CLADE_COL].to_numpy()
    pred_mte = predict_m_mte(train_df, test_df)
    pred_m3_xgb = load_prediction_column(
        explore_ml_predictions_path, "xgboost_m3", y_test, split_label="test"
    )
    pred_m4_xgb = load_prediction_column(
        explore_ml_predictions_path, "xgboost_m4", y_test, split_label="test"
    )
    pred_m3_rf = load_prediction_column(
        explore_ml_predictions_path, "random_forest_m3", y_test, split_label="test"
    )
    pred_m4_rf = load_prediction_column(
        explore_ml_predictions_path, "random_forest_m4", y_test, split_label="test"
    )
    pred_residual_xgb = load_prediction_column(
        residual_predictions_path, "xgboost", y_test, split_label="test"
    )
    pred_residual_rf = load_prediction_column(
        residual_predictions_path, "random_forest", y_test, split_label="test"
    )

    rows = [
        {"model": M_MTE_LABEL, **evaluate_reporting_suite(y_test, pred_mte, classes)},
        {"model": M3_RF_LABEL, **evaluate_reporting_suite(y_test, pred_m3_rf, classes)},
        {"model": M4_RF_LABEL, **evaluate_reporting_suite(y_test, pred_m4_rf, classes)},
        {"model": M3_XGB_LABEL, **evaluate_reporting_suite(y_test, pred_m3_xgb, classes)},
        {"model": M4_XGB_LABEL, **evaluate_reporting_suite(y_test, pred_m4_xgb, classes)},
        {"model": RESIDUAL_RF_LABEL, **evaluate_reporting_suite(y_test, pred_residual_rf, classes)},
        {"model": RESIDUAL_XGB_LABEL, **evaluate_reporting_suite(y_test, pred_residual_xgb, classes)},
    ]
    return pd.DataFrame(rows)


def save_rmse_r2_comparison_plot(metrics_df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = metrics_df.sort_values("rmse").reset_index(drop=True)
    model_order = plot_df["model"].tolist()
    fig_width = max(8.0, 1.8 * len(plot_df) + 4.0)

    sns.set_theme(style="whitegrid")
    with plt.rc_context(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": SUPTITLE_SIZE,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5))

        sns.barplot(
            data=plot_df,
            x="model",
            y="rmse",
            order=model_order,
            ax=axes[0],
            color="#4C72B0",
        )
        axes[0].set_title("RMSE", fontsize=TITLE_SIZE, fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("rmse", fontsize=LABEL_SIZE, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=20, labelsize=TICK_SIZE)
        axes[0].tick_params(axis="y", labelsize=TICK_SIZE)

        sns.barplot(
            data=plot_df,
            x="model",
            y="r2",
            order=model_order,
            ax=axes[1],
            color="#C44E52",
        )
        axes[1].set_title("R2", fontsize=TITLE_SIZE, fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("r2", fontsize=LABEL_SIZE, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=20, labelsize=TICK_SIZE)
        axes[1].tick_params(axis="y", labelsize=TICK_SIZE)

        fig.suptitle(title, fontsize=SUPTITLE_SIZE, fontweight="bold")
        fig.tight_layout()
        for ax in axes:
            apply_bold_fonts(ax)
            for label in ax.get_xticklabels():
                label.set_rotation(20)
                label.set_horizontalalignment("right")
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def save_m1_m4_linear_comparison_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    save_rmse_r2_comparison_plot(
        metrics_df,
        output_path,
        title="M-MTE and M1-M4 regression models on test set",
    )


def save_ml_residual_comparison_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    save_rmse_r2_comparison_plot(
        metrics_df,
        output_path,
        title="Performance comparison of Machine Learning models and M-MTE model on test set",
    )


def fit_m2_mass_slope(
    log_mass: np.ndarray,
    inv_kT: np.ndarray,
    log_bmr: np.ndarray,
) -> tuple[float, float, float, float, int]:
    x = np.asarray(log_mass, dtype=float)
    t = np.asarray(inv_kT, dtype=float)
    y = np.asarray(log_bmr, dtype=float)
    mask = np.isfinite(x) & np.isfinite(t) & np.isfinite(y)
    x = x[mask]
    t = t[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n

    design_full = np.column_stack([np.ones(n, dtype=float), x, t])
    rank = int(np.linalg.matrix_rank(design_full, tol=1e-10))
    if rank < 3 or float(np.nanstd(t)) <= 0:
        # Constant/collinear temperature: M2 mass slope reduces to mass-only b.
        if n < 3:
            return float("nan"), float("nan"), float("nan"), float("nan"), n
        design = np.column_stack([np.ones(n, dtype=float), x])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        resid = y - fitted
        dof = n - design.shape[1]
        if dof <= 0:
            return float(coef[1]), float("nan"), float(coef[0]), float("nan"), n
        mse = float(np.sum(resid**2) / dof)
        try:
            cov = mse * np.linalg.inv(design.T @ design)
            se = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            se = float("nan")
        return float(coef[1]), se, float(coef[0]), float("nan"), n

    coef, _, _, _ = np.linalg.lstsq(design_full, y, rcond=None)
    fitted = design_full @ coef
    resid = y - fitted
    dof = n - design_full.shape[1]
    if dof <= 0:
        return float(coef[1]), float("nan"), float(coef[0]), float(coef[2]), n

    mse = float(np.sum(resid**2) / dof)
    try:
        cov = mse * np.linalg.inv(design_full.T @ design_full)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(coef[1]), se, float(coef[0]), float(coef[2]), n


def build_slope_summary(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    # Class/global mass slopes from M2 (mass + temperature), not mass-only M1.
    required = ["log_mass", "inv_kT", LOG_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"build_slope_summary missing columns: {', '.join(missing)}")

    global_b, global_se, global_intercept, global_temp, global_n = fit_m2_mass_slope(
        df["log_mass"].to_numpy(),
        df["inv_kT"].to_numpy(),
        df[LOG_TARGET].to_numpy(),
    )

    rows: list[dict[str, object]] = []
    for class_name, group in df.groupby(CLADE_COL, sort=True):
        b, se, intercept, temp_coef, n = fit_m2_mass_slope(
            group["log_mass"].to_numpy(),
            group["inv_kT"].to_numpy(),
            group[LOG_TARGET].to_numpy(),
        )
        if n < min_rows or not np.isfinite(b):
            continue
        ci95 = CI_Z * se if np.isfinite(se) else float("nan")
        rows.append(
            {
                "class": str(class_name),
                "b": b,
                "se": se,
                "ci95_low": b - ci95 if np.isfinite(ci95) else float("nan"),
                "ci95_high": b + ci95 if np.isfinite(ci95) else float("nan"),
                "intercept": intercept,
                "temp_coef": temp_coef,
                "n_rows": n,
                "n_species": int(group["taxon_name"].nunique()) if "taxon_name" in group else n,
            }
        )

    summary = pd.DataFrame(rows).sort_values("b", ascending=True).reset_index(drop=True)
    summary.attrs["global_b"] = global_b
    summary.attrs["global_se"] = global_se
    summary.attrs["global_intercept"] = global_intercept
    summary.attrs["global_temp_coef"] = global_temp
    summary.attrs["global_n"] = global_n
    summary.attrs["reference_b"] = REFERENCE_SLOPE
    summary.attrs["slope_model"] = "M2: log_BMR ~ log_mass + inv_kT"
    return summary


def save_slope_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        raise ValueError("No class-level slope estimates available for plotting.")

    global_b = float(summary.attrs["global_b"])
    classes = summary["class"].tolist()
    y_pos = np.arange(len(classes))

    fig_height = max(4.5, 0.45 * len(classes) + 2.0)
    with plt.rc_context(
        {
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    ):
        fig, ax = plt.subplots(figsize=(8.5, fig_height))

        ax.errorbar(
            summary["b"],
            y_pos,
            xerr=[
                summary["b"] - summary["ci95_low"],
                summary["ci95_high"] - summary["b"],
            ],
            fmt="o",
            color="#4C72B0",
            ecolor="#4C72B0",
            elinewidth=1.2,
            capsize=3,
            markersize=6,
            label="95% CI",
            zorder=3,
        )

        ax.axvline(
            REFERENCE_SLOPE,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            label=f"b = {REFERENCE_SLOPE:.2f}",
            zorder=2,
        )
        ax.axvline(
            global_b,
            color="#55A868",
            linestyle="--",
            linewidth=1.6,
            label=f"Estimated b = {global_b:.3f}",
            zorder=2,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, fontsize=TICK_SIZE, fontweight="bold")
        ax.set_xlabel(
            "Exponent b",
            fontsize=LABEL_SIZE,
            fontweight="bold",
        )
        ax.set_ylabel("Clades", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_title(
             "M2 Exponent b estimates",
            fontsize=TITLE_SIZE,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        ax.legend(loc="lower right", frameon=True, prop={"size": LEGEND_SIZE, "weight": "bold"})
        apply_bold_fonts(ax)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    global_b = float(summary.attrs["global_b"])
    global_se = float(summary.attrs["global_se"])
    global_n = int(summary.attrs["global_n"])
    global_temp = float(summary.attrs.get("global_temp_coef", np.nan))
    slope_model = str(summary.attrs.get("slope_model", "M2: log_BMR ~ log_mass + inv_kT"))

    meta = pd.DataFrame(
        [
            {
                "scope": "global",
                "class": "(all)",
                "b": global_b,
                "se": global_se,
                "n_rows": global_n,
                "temp_coef": global_temp,
                "model": slope_model,
            },
            {
                "scope": "reference",
                "class": "(all)",
                "b": REFERENCE_SLOPE,
                "se": np.nan,
                "n_rows": global_n,
                "temp_coef": np.nan,
                "model": "MTE fixed b = 0.75",
            },
        ]
    )
    class_rows = summary.copy()
    class_rows.insert(0, "scope", "class")
    class_rows["model"] = slope_model
    out = pd.concat([meta, class_rows], ignore_index=True, sort=False)
    out.to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Plot class-level temperature-corrected mass-scaling exponent b "
            "(from M2: log_BMR ~ log_mass + inv_kT) with 95% CI, "
            "plus reference lines at b = 0.75 and the global M2 estimate."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/train.csv"),
        help="Training CSV used to estimate slopes and fit linear models.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/splits/test/test.csv"),
        help="Held-out test CSV for model comparisons (default: data/splits/test/test.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plots/slope_estimates.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/plots/slope_estimates.csv"),
        help="Output CSV with per-class and global slope estimates.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10,
        help="Minimum rows per class required to estimate b (default: 10).",
    )
    parser.add_argument(
        "--m1-m4-linear-output",
        type=Path,
        default=Path("results/plots/m1_m4_linear_comparison.pdf"),
        help="Output PDF for M1–M4 linear RMSE/R2 comparison on test set.",
    )
    parser.add_argument(
        "--m1-m4-linear-metrics-output",
        type=Path,
        default=Path("results/plots/m1_m4_linear_comparison.csv"),
        help="Output CSV for M1–M4 linear test metrics.",
    )
    parser.add_argument(
        "--explore-ml-predictions",
        type=Path,
        default=Path("results/explore/test/explore_ml_predictions_test.csv"),
        help="Test predictions CSV from explore_ml.py (M3-XGB / M4-XGB).",
    )
    parser.add_argument(
        "--residual-predictions",
        type=Path,
        default=Path("results/benchmark/all/test/benchmark_predictions_test.csv"),
        help="Test predictions CSV from ml_residual_learning.py (Residual-RF/XGB).",
    )
    parser.add_argument(
        "--ml-residual-output",
        type=Path,
        default=Path("results/plots/ml_residual_comparison.pdf"),
        help="Output PDF for M-MTE / M3-XGB / M4-XGB / Residual-RF / Residual-XGB.",
    )
    parser.add_argument(
        "--ml-residual-metrics-output",
        type=Path,
        default=Path("results/plots/ml_residual_comparison.csv"),
        help="Output CSV for M-MTE / M3-XGB / M4-XGB / Residual-RF / Residual-XGB.",
    )
    args = parser.parse_args()

    train_path = resolve_path(root, args.train)
    test_path = resolve_path(root, args.test)
    output_path = resolve_path(root, args.output)
    summary_output_path = resolve_path(root, args.summary_output)
    m1_m4_linear_output_path = resolve_path(root, args.m1_m4_linear_output)
    m1_m4_linear_metrics_output_path = resolve_path(root, args.m1_m4_linear_metrics_output)
    explore_ml_predictions_path = resolve_path(root, args.explore_ml_predictions)
    residual_predictions_path = resolve_path(root, args.residual_predictions)
    ml_residual_output_path = resolve_path(root, args.ml_residual_output)
    ml_residual_metrics_output_path = resolve_path(root, args.ml_residual_metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m1_m4_linear_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = prepare_frame(pd.read_csv(train_path))
    test_df = prepare_frame(pd.read_csv(test_path))
    train_model_df = prepare_model_frame(pd.read_csv(train_path))
    test_model_df = prepare_model_frame(pd.read_csv(test_path))
    summary = build_slope_summary(train_df, min_rows=args.min_rows)
    if summary.empty:
        raise ValueError(
            f"No classes met min_rows={args.min_rows}. Lower --min-rows or check training data."
        )

    save_slope_plot(summary, output_path)
    write_summary_table(summary, summary_output_path)

    m1_m4_linear_metrics = build_m1_m4_linear_metrics(train_model_df, test_model_df)
    save_m1_m4_linear_comparison_plot(m1_m4_linear_metrics, m1_m4_linear_output_path)
    m1_m4_linear_metrics.to_csv(m1_m4_linear_metrics_output_path, index=False, encoding="utf-8")

    ml_residual_metrics = build_ml_residual_metrics(
        train_model_df,
        test_model_df,
        explore_ml_predictions_path,
        residual_predictions_path,
    )
    save_ml_residual_comparison_plot(ml_residual_metrics, ml_residual_output_path)
    ml_residual_metrics.to_csv(ml_residual_metrics_output_path, index=False, encoding="utf-8")

    # Remove superseded comparison artifacts.
    for stale in (
        output_path.parent / "m1_m2_linear_comparison.png",
        output_path.parent / "m1_m2_linear_comparison.pdf",
        output_path.parent / "m1_m2_linear_comparison.csv",
        output_path.parent / "m2_m3_m4_linear_comparison.png",
        output_path.parent / "m2_m3_m4_linear_comparison.pdf",
        output_path.parent / "m2_m3_m4_linear_comparison.csv",
        output_path.parent / "m3_ml_residual_comparison.png",
        output_path.parent / "m3_ml_residual_comparison.pdf",
        output_path.parent / "m3_ml_residual_comparison.csv",
        # Superseded PNG versions of current PDF outputs.
        output_path.parent / "slope_estimates.png",
        output_path.parent / "m1_m4_linear_comparison.png",
        output_path.parent / "ml_residual_comparison.png",
    ):
        if stale.exists():
            stale.unlink()

    global_b = float(summary.attrs["global_b"])
    global_temp = float(summary.attrs.get("global_temp_coef", np.nan))
    print(f"Slope model: {summary.attrs.get('slope_model')}")
    print(
        f"Global M2 b = {global_b:.4f} "
        f"(temp_coef = {global_temp:.4f}; reference b = {REFERENCE_SLOPE:.2f})"
    )
    print(f"Plotted classes: {len(summary)}")
    print(f"Saved plot: {output_path}")
    print(f"Saved summary: {summary_output_path}")
    print(f"Saved M1–M4 linear comparison plot: {m1_m4_linear_output_path}")
    print(f"Saved M1–M4 linear comparison metrics: {m1_m4_linear_metrics_output_path}")
    print(m1_m4_linear_metrics.to_string(index=False))
    print(f"Saved ML/residual comparison plot: {ml_residual_output_path}")
    print(f"Saved ML/residual comparison metrics: {ml_residual_metrics_output_path}")
    print(ml_residual_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
