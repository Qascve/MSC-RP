#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
TEMP_COL = "temperature"
CLADE_COL = "class"
K_BOLTZMANN_EV_PER_K = 8.617e-5
PHYLO_PC_COLS = ["pc1", "pc2", "pc3", "pc4", "pc5"]
ML_MODEL_SUFFIXES = ("m1", "m2", "m3", "m4")
MTE_FIXED_B = 0.75
LINEAR_NAME_MAP = {
    "m1_estimated_b": "M1-R",
    # Fixed-b MTE: same predictors as M2, mass slope locked at 3/4.
    "m_mte_fixed_b": "M-MTE",
    "m2_baseline_mte": "M2-R",
    "m3_clade_specific_mte": "M3-R",
    "m4_phylo_linear_mte": "M4-R",
    "m4_pgls_ape_mte": "M4-PGLS",
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


def run_python_dependency(
    script_path: Path,
    root: Path,
    extra_args: list[str],
    label: str,
) -> None:
    # Run a required pipeline script with visible output and fail fast.
    cmd = [sys.executable, str(script_path), *extra_args]
    print(f"\n[{label}] Missing or stale results; running dependency...", flush=True)
    completed = subprocess.run(cmd, cwd=root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} dependency failed with exit code {completed.returncode}: "
            f"{' '.join(cmd)}"
        )


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["taxon_name", CLADE_COL, MASS_COL, TEMP_COL, TARGET, *PHYLO_PC_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip().replace("", pd.NA)
    out[CLADE_COL] = out[CLADE_COL].astype("string").str.strip().replace("", pd.NA)
    for col in [MASS_COL, TEMP_COL, TARGET, *PHYLO_PC_COLS]:
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
    out["log_mass"] = np.log10(out[MASS_COL].to_numpy())
    out["log_BMR"] = np.log10(out[TARGET].to_numpy())
    return out


def make_class_balanced_sample_weight(train_df: pd.DataFrame) -> np.ndarray:
    # Same class-balanced row weights as ml_residual_learning XGB/RF.
    classes = train_df[CLADE_COL].to_numpy()
    unique_classes = np.unique(classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes,
    )
    weight_map = dict(zip(unique_classes, class_weights))
    return np.array([weight_map[c] for c in classes], dtype=float)


def fit_ols(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    # Ordinary or weighted least squares (WLS via row scaling).
    if sample_weight is None:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef
    sw = np.asarray(sample_weight, dtype=float)
    if sw.shape != (len(y),):
        raise ValueError(
            f"sample_weight shape {sw.shape} does not match y length {len(y)}."
        )
    if np.any(sw < 0) or not np.all(np.isfinite(sw)):
        raise ValueError("sample_weight must be finite and non-negative.")
    scale = np.sqrt(sw)
    coef, *_ = np.linalg.lstsq(X * scale[:, None], y * scale, rcond=None)
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


def build_design_m4(df: pd.DataFrame, pc_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    #     M4 linear design: log_BMR ~ log_mass + inv_kT + PC axes + interactions
    # (log_mass * PC, inv_kT * PC), matching bmr_models.py m4 with algo=lm.
    #
    pc_cols = pc_cols or PHYLO_PC_COLS
    x_log_mass = df["log_mass"].to_numpy(dtype=float)
    x_inv_kT = df["inv_kT"].to_numpy(dtype=float)

    blocks = [np.ones(len(df), dtype=float), x_log_mass, x_inv_kT]
    names = ["Intercept", "log_mass", "inv_kT"]

    for pc in pc_cols:
        x_pc = df[pc].to_numpy(dtype=float)
        blocks.append(x_pc)
        names.append(pc)

    for pc in pc_cols:
        x_pc = df[pc].to_numpy(dtype=float)
        blocks.append(x_log_mass * x_pc)
        names.append(f"log_mass_x_{pc}")
        blocks.append(x_inv_kT * x_pc)
        names.append(f"inv_kT_x_{pc}")

    return np.column_stack(blocks), names


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    # Micro-averaged metrics on pooled log10(BMR) observations.
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = np.asarray(y_true_log, dtype=float)[mask]
    y_pred_log = np.asarray(y_pred_log, dtype=float)[mask]
    if len(y_true_log) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
        if len(yt) < 2 or np.isclose(np.var(yt), 0.0):
            return float("nan")
        return float(r2_score(yt, yp))

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2": _r2(y_true_log, y_pred_log),
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
    rmse = float(np.sqrt(np.sum(sw * resid**2) / w_sum))
    mae = float(np.sum(sw * np.abs(resid)) / w_sum)
    y_bar = float(np.sum(sw * y_true_log) / w_sum)
    ss_tot = float(np.sum(sw * (y_true_log - y_bar) ** 2))
    ss_res = float(np.sum(sw * resid**2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {"rmse_bal": rmse, "mae_bal": mae, "r2_bal": r2}


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
        rows.append(evaluate(y_true_log[mask], y_pred_log[mask]))
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
    #     Report micro (pooled), macro (per-class then equal-average), and
    # class-balanced weighted metrics using the same w_c formula as training.
    #
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    classes = np.asarray(classes)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = y_true_log[mask]
    y_pred_log = y_pred_log[mask]
    classes = classes[mask]
    micro = evaluate(y_true_log, y_pred_log)
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


def to_short_model_name(model_name: str) -> str:
    if model_name in LINEAR_NAME_MAP:
        return LINEAR_NAME_MAP[model_name]
    if model_name.startswith("random_forest_"):
        suffix = model_name.replace("random_forest_", "", 1)
        return f"{suffix.upper()}-RF"
    if model_name.startswith("xgboost_"):
        suffix = model_name.replace("xgboost_", "", 1)
        return f"{suffix.upper()}-XGB"
    if model_name in ("Residual-RF", "Residual-XGB"):
        return model_name
    return model_name


LINEAR_MODEL_KEYS = (
    "m1_estimated_b",
    "m_mte_fixed_b",
    "m2_baseline_mte",
    "m3_clade_specific_mte",
    "m4_phylo_linear_mte",
    "m4_pgls_ape_mte",
)
# Models trained with class-balanced sample weights (same formula as residual XGB/RF).
CLASS_WEIGHTED_TRAIN_MODELS = {
    "m3_clade_specific_mte",
    "m4_phylo_linear_mte",
    "Residual-RF",
    "Residual-XGB",
    # explore_ml RF/XGB M1–M4
    *[f"random_forest_m{i}" for i in (1, 2, 3, 4)],
    *[f"xgboost_m{i}" for i in (1, 2, 3, 4)],
}
CLASS_BALANCED_WEIGHT_FORMULA = (
    "w_c = n / (n_classes * n_c), where n is the number of training rows, "
    "n_classes is the number of taxonomic classes present, and n_c is the "
    "number of training rows in class c (sklearn class_weight='balanced')."
)


def load_benchmark_predictions(path: Path, eval_df: pd.DataFrame, split_label: str = "test") -> dict[str, np.ndarray]:
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
            f"{path.name} missing ML model columns like random_forest_m1..m4/xgboost_m1..m4."
        )

    for col in required:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    pred_df = pred_df.dropna(subset=required).reset_index(drop=True)

    y_true = eval_df["log_BMR"].to_numpy()
    if len(pred_df) != len(eval_df):
        raise ValueError(
            f"Benchmark predictions row count mismatches {split_label} split "
            f"({path}: {len(pred_df)} rows vs {split_label} {len(eval_df)}). "
            "Please rerun explore_ml.py on the same train/test files."
        )

    if not np.allclose(pred_df["y_true"].to_numpy(), y_true, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"Benchmark predictions y_true is not aligned with current {split_label} split. "
            "Please rerun explore_ml.py on the same split file."
        )

    return {col: pred_df[col].to_numpy(dtype=float) for col in benchmark_cols}


def _residual_model_columns(pred_df: pd.DataFrame) -> list[str]:
    # Accept fold-best single model or legacy dual RF/XGB columns.
    cols = [c for c in ("random_forest", "xgboost") if c in pred_df.columns]
    if cols:
        return cols
    raise KeyError(
        "Missing residual-learning model columns (expected random_forest and/or xgboost)."
    )


def load_residual_learning_predictions(path: Path, eval_df: pd.DataFrame, split_label: str = "test") -> dict[str, np.ndarray]:
    pred_df = pd.read_csv(path)
    if "y_true" not in pred_df.columns:
        raise KeyError(f"{path.name} missing required column: y_true")
    model_cols = _residual_model_columns(pred_df)

    for col in ["y_true", *model_cols]:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    if "log_mass" in pred_df.columns:
        pred_df["log_mass"] = pd.to_numeric(pred_df["log_mass"], errors="coerce")
    if "inv_kT" in pred_df.columns:
        pred_df["inv_kT"] = pd.to_numeric(pred_df["inv_kT"], errors="coerce")
    pred_df = pred_df.dropna(subset=["y_true", *model_cols]).reset_index(drop=True)

    y_true = eval_df["log_BMR"].to_numpy()
    aligned = False
    if (
        len(pred_df) == len(eval_df)
        and np.allclose(pred_df["y_true"].to_numpy(), y_true, rtol=1e-10, atol=1e-12)
    ):
        aligned = True
    elif {
        "taxon_name",
        "log_mass",
        "inv_kT",
    }.issubset(pred_df.columns) and {
        "taxon_name",
        "log_mass",
        "inv_kT",
    }.issubset(eval_df.columns):
        # CV OOF rows may be ordered by fold; align back onto eval_df row order.
        left = eval_df[["taxon_name", "log_mass", "inv_kT"]].copy().reset_index(drop=True)
        left["_row_id"] = np.arange(len(left))
        left["y_true"] = y_true
        right = pred_df[
            ["taxon_name", "log_mass", "inv_kT", "y_true", *model_cols]
        ].copy()
        merged = left.merge(
            right,
            on=["taxon_name", "log_mass", "inv_kT", "y_true"],
            how="left",
            validate="one_to_one",
        ).sort_values("_row_id")
        if merged[model_cols].isna().any().any():
            raise ValueError(
                f"Residual-learning predictions could not be aligned to {split_label} split "
                f"({path}). Please rerun ml_residual_learning.py on the same split files."
            )
        pred_df = merged.reset_index(drop=True)
        aligned = True

    if not aligned:
        raise ValueError(
            f"Residual-learning predictions are not aligned with current {split_label} split "
            f"({path}: {len(pred_df)} rows vs {split_label} {len(eval_df)}). "
            "Please rerun ml_residual_learning.py on the same train/test files."
        )

    name_map = {"random_forest": "Residual-RF", "xgboost": "Residual-XGB"}
    return {
        name_map[col]: pred_df[col].to_numpy(dtype=float) for col in model_cols
    }


def load_residual_learning_predictions_by_class(path: Path, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Residual-learning benchmark directory not found: {path}")

    combined: dict[str, np.ndarray] = {}
    loaded_classes: list[str] = []
    for class_dir in sorted([p for p in path.iterdir() if p.is_dir()]):
        pred_path = class_dir / "benchmark_predictions_test.csv"
        if not pred_path.exists():
            continue
        class_name = class_dir.name
        class_mask = test_df[CLADE_COL].astype("string").to_numpy() == class_name
        if not bool(class_mask.any()):
            continue

        pred_df = pd.read_csv(pred_path)
        if "taxon_name" not in pred_df.columns or "y_true" not in pred_df.columns:
            raise KeyError(f"{pred_path.name} missing taxon_name/y_true.")
        model_cols = _residual_model_columns(pred_df)

        for col in ["y_true", *model_cols]:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
        pred_df = pred_df.dropna(subset=["taxon_name", "y_true", *model_cols]).reset_index(drop=True)

        class_test = test_df[class_mask].reset_index(drop=True)
        if len(pred_df) != len(class_test):
            raise ValueError(
                f"Residual-learning predictions for {class_name} have {len(pred_df)} rows, "
                f"but test split has {len(class_test)} rows for that class."
            )
        if not np.array_equal(
            pred_df["taxon_name"].astype("string").to_numpy(),
            class_test["taxon_name"].astype("string").to_numpy(),
        ):
            raise ValueError(f"Residual-learning predictions for {class_name} are not aligned by taxon_name.")
        if not np.allclose(pred_df["y_true"].to_numpy(), class_test["log_BMR"].to_numpy(), rtol=1e-10, atol=1e-12):
            raise ValueError(f"Residual-learning y_true for {class_name} is not aligned with test split.")

        name_map = {"random_forest": "Residual-RF", "xgboost": "Residual-XGB"}
        for col in model_cols:
            out_name = name_map[col]
            if out_name not in combined:
                combined[out_name] = np.full(len(test_df), np.nan, dtype=float)
            combined[out_name][class_mask] = pred_df[col].to_numpy(dtype=float)
        loaded_classes.append(class_name)

    if not loaded_classes:
        raise ValueError(f"No class benchmark predictions loaded from: {path}")
    print("Loaded residual-learning predictions for classes: " + ", ".join(loaded_classes))
    return combined


def run_pgls_with_ape(
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
        raise FileNotFoundError(f"PGLS R script not found: {resolved_r_script_path}")

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
            "PGLS (ape/nlme) failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    pred_csv = resolved_out_dir / "pgls_test_predictions.csv"
    if not pred_csv.exists():
        raise RuntimeError("PGLS output file was not generated.")

    pred_df = pd.read_csv(pred_csv)

    required = ["row_id", "taxon_name", "log_BMR", "y_pred_log_BMR"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise KeyError(f"PGLS output missing required columns: {', '.join(missing)}")

    pred_df["row_id"] = pd.to_numeric(pred_df["row_id"], errors="coerce")
    pred_df["log_BMR"] = pd.to_numeric(pred_df["log_BMR"], errors="coerce")
    pred_df["y_pred_log_BMR"] = pd.to_numeric(pred_df["y_pred_log_BMR"], errors="coerce")
    pred_df = pred_df.dropna(subset=["row_id"]).copy()
    pred_df["row_id"] = pred_df["row_id"].astype(int)
    pred_df = pred_df.sort_values("row_id").reset_index(drop=True)

    if len(pred_df) != len(test_df):
        raise ValueError(
            "PGLS output row count mismatch with test data. "
            "Please rerun pgls_ape.R on the same test split."
        )
    expected_row_ids = np.arange(len(test_df), dtype=int)
    if not np.array_equal(pred_df["row_id"].to_numpy(), expected_row_ids):
        raise ValueError("PGLS output row_id is not aligned with test data order.")
    if not np.array_equal(
        pred_df["taxon_name"].astype("string").to_numpy(),
        test_df["taxon_name"].astype("string").to_numpy(),
    ):
        raise ValueError("PGLS output taxon_name is not aligned with test data order.")
    if not np.allclose(
        pred_df["log_BMR"].to_numpy(dtype=float),
        test_df["log_BMR"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("PGLS output log_BMR is not aligned with test data order.")

    return pred_df["y_pred_log_BMR"].to_numpy(dtype=float)


def load_pgls_train_fitted(path: Path, train_df: pd.DataFrame) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"PGLS train fitted file not found: {path}")
    pred_df = pd.read_csv(path)
    required = ["taxon_name", "log_BMR", "log_mass", "y_fitted_log_BMR"]
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise KeyError(f"PGLS train output missing required columns: {', '.join(missing)}")

    pred_df["taxon_name"] = pred_df["taxon_name"].astype("string").str.strip()
    pred_df["log_BMR"] = pd.to_numeric(pred_df["log_BMR"], errors="coerce")
    pred_df["log_mass"] = pd.to_numeric(pred_df["log_mass"], errors="coerce")
    pred_df["y_fitted_log_BMR"] = pd.to_numeric(pred_df["y_fitted_log_BMR"], errors="coerce")
    pred_df = pred_df.dropna(subset=required).copy()

    merge_keys = ["taxon_name", "log_BMR", "log_mass"]
    train_keys = train_df[merge_keys].copy()
    train_keys["taxon_name"] = train_keys["taxon_name"].astype("string").str.strip()
    merged = train_keys.merge(
        pred_df[merge_keys + ["y_fitted_log_BMR"]],
        on=merge_keys,
        how="left",
    )
    if int(merged["y_fitted_log_BMR"].notna().sum()) == 0:
        raise ValueError("PGLS train fitted values could not be aligned with development data.")
    return merged["y_fitted_log_BMR"].to_numpy(dtype=float)


def _build_model_metrics(
    eval_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    comparison_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    y_true = eval_df["log_BMR"].to_numpy()
    classes = eval_df[CLADE_COL].astype(str).to_numpy()
    if comparison_mask is None:
        comparison_mask = np.ones(len(eval_df), dtype=bool)
    metric_rows = []
    empty_extra = {
        "rmse_macro": np.nan,
        "mae_macro": np.nan,
        "r2_macro": np.nan,
        "n_classes_eval": np.nan,
        "rmse_bal": np.nan,
        "mae_bal": np.nan,
        "r2_bal": np.nan,
    }
    for model_name, y_pred in sorted(predictions.items()):
        y_t = y_true[comparison_mask]
        y_p = y_pred[comparison_mask]
        # Macro / class-balanced metrics only for models trained with class weights.
        # Unweighted models (M1/M-MTE/M2 linear, PGLS) keep micro rmse/mae/r2 only.
        if model_name in CLASS_WEIGHTED_TRAIN_MODELS:
            suite = evaluate_reporting_suite(y_t, y_p, classes[comparison_mask])
        else:
            suite = {**evaluate(y_t, y_p), **empty_extra}
        metric_rows.append(
            {
                "model": model_name,
                "train_class_weighted": int(model_name in CLASS_WEIGHTED_TRAIN_MODELS),
                **suite,
            }
        )
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    metrics_df["model_key"] = metrics_df["model"]
    metrics_df["model"] = metrics_df["model"].map(to_short_model_name)
    return metrics_df


def write_evaluation_protocol_note(out_dir: Path) -> Path:
    # Document split terminology and class-balanced weight / metric definitions.
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation_protocol.txt"
    text = "\n".join(
        [
            "Evaluation protocol (explore.py)",
            "",
            "Partitioning:",
            "- The held-out 20% test bucket is a species-blocked holdout, not by itself",
            "  'cross-validation'. Residual-learning hyper-parameters use a separate",
            "  fixed 4-fold species-blocked CV inside the 80% development set; all",
            "  models are compared on the same held-out test partition.",
            "- Stratification: within each taxonomic class, whole species are assigned",
            "  to F1/F2/F3/F4/T (~20% each) with per-class quotas; remainder slots go",
            "  to currently lightest buckets so small classes still cover as many",
            "  buckets as their species count allows.",
            "- Classes with fewer than 5 species cannot occupy every bucket; some",
            "  folds/test may lack that class. Classes kept in the working data with",
            "  enough species are represented across buckets when n_species >= 5.",
            "",
            "Class-balanced sample weights (training for M3-R, M4-R, Residual-RF/XGB,",
            "  and explore_ml RF/XGB M1–M4):",
            f"- {CLASS_BALANCED_WEIGHT_FORMULA}",
            "",
            "Reported metrics:",
            "- rmse/mae/r2: micro-averaged over all pooled evaluation rows (all models)",
            "- rmse_macro/mae_macro/r2_macro and rmse_bal/mae_bal/r2_bal: only for",
            "  models trained with class-balanced weights (M3-R, M4-R, Residual-RF/XGB,",
            "  explore_ml M1–M4 RF/XGB)",
            "- train_class_weighted=1 marks those models",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    benchmark_predictions: dict[str, np.ndarray],
    benchmark_train_predictions: dict[str, np.ndarray],
    pgls_predictions: np.ndarray,
    pgls_train_predictions: np.ndarray,
    residual_learning_predictions: dict[str, np.ndarray],
    residual_train_predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    y_train_log = train_df["log_BMR"].to_numpy()
    # Same class-balanced weights as residual-learning XGB/RF.
    sw_train = make_class_balanced_sample_weight(train_df)

    # m1: log_BMR ~ log_mass
    X1_train = np.column_stack([np.ones(len(train_df)), train_df["log_mass"].to_numpy()])
    X1_test = np.column_stack([np.ones(len(test_df)), test_df["log_mass"].to_numpy()])
    coef_m1 = fit_ols(X1_train, y_train_log)
    yhat_m1_train = predict_ols(X1_train, coef_m1)
    yhat_m1_log = predict_ols(X1_test, coef_m1)

    # m_mte: fixed-b MTE = M2 with mass slope locked at 3/4
    # log_BMR ~ inv_kT + offset(0.75 * log_mass)
    log_mass_train = train_df["log_mass"].to_numpy(dtype=float)
    log_mass_test = test_df["log_mass"].to_numpy(dtype=float)
    y_mte_adj = y_train_log - MTE_FIXED_B * log_mass_train
    X_mte_train = np.column_stack([np.ones(len(train_df)), train_df["inv_kT"].to_numpy()])
    X_mte_test = np.column_stack([np.ones(len(test_df)), test_df["inv_kT"].to_numpy()])
    coef_mte = fit_ols(X_mte_train, y_mte_adj)
    yhat_mte_train = MTE_FIXED_B * log_mass_train + predict_ols(X_mte_train, coef_mte)
    yhat_mte_log = MTE_FIXED_B * log_mass_test + predict_ols(X_mte_test, coef_mte)

    # m2: log_BMR ~ log_mass + inv_kT  (estimated b)
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
    yhat_m2_train = predict_ols(X2_train, coef_m2)
    yhat_m2_log = predict_ols(X2_test, coef_m2)

    # m3: log_BMR ~ log_mass + inv_kT + class  (class-balanced WLS)
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
    coef_m3 = fit_ols(X3_train, y_train_log, sample_weight=sw_train)
    yhat_m3_train = predict_ols(X3_train, coef_m3)
    yhat_m3_log = predict_ols(X3_test, coef_m3)

    # m4: log_BMR ~ (log_mass + inv_kT) * phylogeny  (class-balanced WLS)
    X4_train, _ = build_design_m4(train_df)
    X4_test, _ = build_design_m4(test_df)
    coef_m4 = fit_ols(X4_train, y_train_log, sample_weight=sw_train)
    yhat_m4_train = predict_ols(X4_train, coef_m4)
    yhat_m4_log = predict_ols(X4_test, coef_m4)

    y_true = test_df["log_BMR"].to_numpy()
    predictions: dict[str, np.ndarray] = {
        "m1_estimated_b": yhat_m1_log,
        "m_mte_fixed_b": yhat_mte_log,
        "m2_baseline_mte": yhat_m2_log,
        "m4_phylo_linear_mte": yhat_m4_log,
        "m4_pgls_ape_mte": pgls_predictions,
        **benchmark_predictions,
        **residual_learning_predictions,
    }
    train_predictions: dict[str, np.ndarray] = {
        "m1_estimated_b": yhat_m1_train,
        "m_mte_fixed_b": yhat_mte_train,
        "m2_baseline_mte": yhat_m2_train,
        "m3_clade_specific_mte": yhat_m3_train,
        "m4_phylo_linear_mte": yhat_m4_train,
        "m4_pgls_ape_mte": pgls_train_predictions,
        **benchmark_train_predictions,
        **residual_train_predictions,
    }

    comparison_mask = np.ones(len(test_df), dtype=bool)
    for y_pred in residual_learning_predictions.values():
        comparison_mask &= np.isfinite(y_pred)
    if not bool(comparison_mask.any()):
        comparison_mask = np.ones(len(test_df), dtype=bool)

    train_comparison_mask = np.ones(len(train_df), dtype=bool)
    for y_pred in residual_train_predictions.values():
        train_comparison_mask &= np.isfinite(y_pred)
    if not bool(train_comparison_mask.any()):
        train_comparison_mask = np.ones(len(train_df), dtype=bool)

    if len(test_df_m3) > 0:
        y_pred_m3_full = np.full(len(test_df), np.nan, dtype=float)
        y_pred_m3_full[known_mask.to_numpy()] = yhat_m3_log
        predictions["m3_clade_specific_mte"] = y_pred_m3_full

    metrics_df = _build_model_metrics(test_df, predictions, comparison_mask)
    train_metrics_df = _build_model_metrics(train_df, train_predictions, train_comparison_mask)
    predictions_short = {to_short_model_name(k): v for k, v in predictions.items()}
    train_predictions_short = {to_short_model_name(k): v for k, v in train_predictions.items()}

    return (
        metrics_df,
        predictions_short,
        y_true,
        train_metrics_df,
        train_predictions_short,
        y_train_log,
    )


def split_linear_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    # Rows for linear/phylo M-MTE and M1–M4 only (exclude explore_ml + residual).
    if "model_key" in metrics_df.columns:
        mask = metrics_df["model_key"].isin(LINEAR_MODEL_KEYS)
        out = metrics_df.loc[mask].drop(columns=["model_key"], errors="ignore")
    else:
        linear_short = {LINEAR_NAME_MAP[k] for k in LINEAR_MODEL_KEYS if k in LINEAR_NAME_MAP}
        out = metrics_df[metrics_df["model"].isin(linear_short)].copy()
    return out.reset_index(drop=True)


def write_metrics_by_fold(out_dir: Path, fold_tags: list[str]) -> Path:
    # Wide summary: one row per model, RMSE/MAE/R2 columns per fold.
    frames = []
    for tag in fold_tags:
        path = out_dir / tag / "explore_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["fold"] = tag
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No explore_metrics.csv under {out_dir}")
    long_df = pd.concat(frames, ignore_index=True)
    # Prefer short model names already stored
    wide_parts = []
    for metric in ("rmse", "mae", "r2"):
        if metric not in long_df.columns:
            continue
        pivot = long_df.pivot_table(index="model", columns="fold", values=metric, aggfunc="first")
        pivot = pivot.reindex(columns=[t for t in fold_tags if t in pivot.columns])
        pivot.columns = [f"{metric}_{c}" for c in pivot.columns]
        wide_parts.append(pivot)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    out_path = out_dir / "explore_metrics_by_fold.csv"
    wide.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[metrics by fold] -> {out_path}")
    return out_path


def write_linear_metrics_by_fold(out_dir: Path, fold_tags: list[str]) -> Path:
    frames = []
    for tag in fold_tags:
        path = out_dir / tag / "explore_linear_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["fold"] = tag
        frames.append(df)
    if not frames:
        print("Skip linear metrics-by-fold: no explore_linear_metrics.csv found.")
        return out_dir / "explore_linear_metrics_by_fold.csv"
    long_df = pd.concat(frames, ignore_index=True)
    wide_parts = []
    for metric in ("rmse", "mae", "r2"):
        if metric not in long_df.columns:
            continue
        pivot = long_df.pivot_table(index="model", columns="fold", values=metric, aggfunc="first")
        pivot = pivot.reindex(columns=[t for t in fold_tags if t in pivot.columns])
        pivot.columns = [f"{metric}_{c}" for c in pivot.columns]
        wide_parts.append(pivot)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    out_path = out_dir / "explore_linear_metrics_by_fold.csv"
    wide.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[linear metrics by fold] -> {out_path}")
    return out_path


def _is_linear_model(name: str) -> bool:
    n = str(name)
    return n in set(LINEAR_NAME_MAP.values()) or n in LINEAR_MODEL_KEYS


def select_model_performance_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    # Keep every evaluated model; sort by RMSE ascending. No filtering.
    work = metrics_df.copy()
    if "model" not in work.columns:
        raise KeyError("metrics_df requires a model column")
    out = work.drop(columns=["model_key"], errors="ignore")
    return out.sort_values("rmse").reset_index(drop=True)


def select_best_ml_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    # Backward-compatible alias for model performance plotting.
    return select_model_performance_rows(metrics_df)


def save_model_performance_plot(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    fold_tag: str | None = None,
) -> Path:
    plot_df = select_model_performance_rows(metrics_df)
    model_order = plot_df["model"].tolist()
    sns.set_theme(style="whitegrid")
    fig_width = max(12.0, 0.75 * len(plot_df) + 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))

    sns.barplot(data=plot_df, x="model", y="rmse", order=model_order, ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE (log10(BMR))")
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)

    sns.barplot(data=plot_df, x="model", y="r2", order=model_order, ax=axes[1], color="#C44E52")
    axes[1].set_title("R2 (log10(BMR))")
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)

    for ax in axes:
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    if fold_tag == "test":
        title = "Model Performance Comparison (held-out test set)"
    elif fold_tag:
        title = f"Model Performance Comparison ({fold_tag})"
    else:
        title = "Model Performance Comparison"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    output_path = out_dir / "model_performance_comparison.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    plot_df.to_csv(out_dir / "model_performance_comparison_metrics.csv", index=False, encoding="utf-8")
    return output_path


def save_top5_plus_residual_learning_plot(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    fold_tag: str | None = None,
    top_n: int = 5,
) -> tuple[Path, Path]:
    # Plot the top-N models by RMSE among all evaluated models.
    plot_df = select_model_performance_rows(metrics_df).head(top_n).copy()
    keep_cols = [c for c in ("model", "rmse", "mae", "r2") if c in plot_df.columns]
    plot_df = plot_df[keep_cols].reset_index(drop=True)
    model_order = plot_df["model"].tolist()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    sns.barplot(
        data=plot_df,
        x="model",
        y="rmse",
        order=model_order,
        ax=axes[0],
        color="#4C72B0",
    )
    axes[0].set_title("RMSE (log10(BMR))")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df,
        x="model",
        y="r2",
        order=model_order,
        ax=axes[1],
        color="#C44E52",
    )
    axes[1].set_title("R2 (log10(BMR))")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    if fold_tag == "test":
        title = f"Top-{top_n} models (held-out test set)"
    elif fold_tag:
        title = f"Top-{top_n} models ({fold_tag})"
    else:
        title = f"Top-{top_n} models"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    plot_path = out_dir / "top5_plus_residual_learning_performance.pdf"
    data_path = out_dir / "top5_plus_residual_learning_metrics.csv"
    fig.savefig(plot_path, bbox_inches="tight")
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
    plt.xlabel("Predicted log10(BMR)")
    plt.ylabel("Residual (log Observed - log Predicted)")
    plt.title("Residual Plot (log10(BMR))")
    plt.legend()
    plt.tight_layout()
    output_path = out_dir / "residual_plot_all_models.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def save_explore_predictions(
    eval_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_dir: Path,
    split: str = "test",
    fold_tag: str | None = None,
) -> Path:
    pred_df = eval_df[["taxon_name", CLADE_COL, "log_mass", "inv_kT"]].copy()
    pred_df["y_true"] = y_true
    pred_df["eval_split"] = split
    for model_name, y_pred in predictions.items():
        pred_df[model_name] = y_pred
    if fold_tag is not None:
        pred_df["fold"] = fold_tag
    path = (
        out_dir / "explore_predictions_test.csv"
        if split == "test"
        else out_dir / f"explore_predictions_{split}.csv"
    )
    pred_df.to_csv(path, index=False, encoding="utf-8")
    return path


def log_bmr_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    # Multiplicative accuracy on log10(BMR): 10^(-|pred - true|).
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    out = np.full(len(y_true_log), np.nan, dtype=float)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    out[mask] = 10.0 ** (-np.abs(y_pred_log[mask] - y_true_log[mask]))
    return np.clip(out, 0.0, 1.0)


def build_species_accuracy_table(
    pred_df: pd.DataFrame,
    model_cols: list[str],
) -> pd.DataFrame:
    work = pred_df.copy()
    work["taxon_name"] = work["taxon_name"].astype("string").str.strip()
    y_true = pd.to_numeric(work["y_true"], errors="coerce").to_numpy(dtype=float)
    for model in model_cols:
        y_pred = pd.to_numeric(work[model], errors="coerce").to_numpy(dtype=float)
        work[model] = log_bmr_accuracy(y_true, y_pred)
    out = (
        work.groupby("taxon_name", as_index=False)[model_cols]
        .mean(numeric_only=True)
        .sort_values("taxon_name")
        .reset_index(drop=True)
    )
    return out[["taxon_name", *model_cols]]


def discover_fold_splits(split_dir: Path, folds: list[str]) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    for name in folds:
        train_path = split_dir / name / "train.csv"
        test_path = split_dir / name / "test.csv"
        if train_path.exists() and test_path.exists():
            found.append((name, train_path, test_path))
    if not found:
        raise FileNotFoundError(
            f"No requested splits found under {split_dir}. Expected test/."
        )
    return found


def write_explore_species_accuracy(out_dir: Path, fold_tags: list[str]) -> Path:
    frames = []
    for tag in fold_tags:
        path = out_dir / tag / "explore_predictions_test.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing explore predictions: {path}")
        fold_df = pd.read_csv(path)
        fold_df["eval_split"] = tag
        frames.append(fold_df)
    stitched = pd.concat(frames, ignore_index=True)
    skip = {"taxon_name", CLADE_COL, "log_mass", "inv_kT", "y_true", "fold", "eval_split"}
    model_cols = [c for c in stitched.columns if c not in skip]
    if not model_cols:
        raise KeyError("No model prediction columns found for species accuracy.")
    accuracy_df = build_species_accuracy_table(stitched, model_cols)
    out_path = out_dir / "explore_species_accuracy.csv"
    accuracy_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"[species accuracy explore] species={len(accuracy_df)}, "
        f"splits={fold_tags}, models={len(model_cols)} -> {out_path}"
    )
    return out_path


def main() -> None:
    root = find_root()
    fold_name_map = {"test": "test"}
    parser = argparse.ArgumentParser(
        description=(
            "Fit linear/phylo M-MTE and M1-M4 on the shared species-blocked held-out test "
            "partition (not a single-split 'CV'), merge explore_ml and "
            "residual-learning test predictions, and write final test reports "
            "with micro, macro, and class-balanced weighted metrics."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/explore"))
    parser.add_argument(
        "--benchmark-predictions-dir",
        type=Path,
        default=Path("results/explore"),
        help="Directory with test/explore_ml_predictions_test.csv from explore_ml.py.",
    )
    parser.add_argument(
        "--residual-learning-dir",
        type=Path,
        default=Path("results/benchmark/all"),
        help="Directory with test/benchmark_predictions_test.csv from ml_residual_learning.py.",
    )
    parser.add_argument(
        "--phylo-tree",
        type=Path,
        default=Path("data/phylogeny/unique_taxon_names.nwk"),
    )
    parser.add_argument("--pgls-r-script", type=Path, default=Path("code/pgls_ape.R"))
    parser.add_argument("--pgls-output-dir", type=Path, default=Path("results/pgls_ape"))
    args = parser.parse_args()

    split_dir = _resolve_path(root, args.split_dir)
    out_dir = _resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_splits = discover_fold_splits(split_dir, ["test"])
    fold_tags: list[str] = []

    for fold_name, train_path, test_path in fold_splits:
        fold_tag = fold_name_map.get(fold_name, fold_name)
        fold_tags.append(fold_tag)
        fold_out = out_dir / fold_tag
        fold_out.mkdir(parents=True, exist_ok=True)

        ml_pred_path = (
            _resolve_path(root, args.benchmark_predictions_dir)
            / fold_tag
            / "explore_ml_predictions_test.csv"
        )
        ml_train_pred_path = (
            _resolve_path(root, args.benchmark_predictions_dir)
            / fold_tag
            / "explore_ml_predictions_train.csv"
        )
        residual_pred_path = (
            _resolve_path(root, args.residual_learning_dir)
            / fold_tag
            / "benchmark_predictions_test.csv"
        )
        residual_cv_pred_path = (
            _resolve_path(root, args.residual_learning_dir)
            / "cv"
            / "benchmark_predictions_cv.csv"
        )
        pgls_fold_out = _resolve_path(root, args.pgls_output_dir) / fold_tag

        print(f"\n=== explore {fold_tag} ({fold_name}) ===", flush=True)
        train_df = add_mte_features(load_split_data(train_path))
        test_df = add_mte_features(load_split_data(test_path))

        benchmark_predictions: dict[str, np.ndarray] | None = None
        benchmark_train_predictions: dict[str, np.ndarray] | None = None
        if ml_pred_path.exists():
            try:
                benchmark_predictions = load_benchmark_predictions(ml_pred_path, test_df, "test")
            except (ValueError, KeyError) as exc:
                print(f"[{fold_tag}] Existing explore_ml test predictions are stale: {exc}")
                benchmark_predictions = None
        if ml_train_pred_path.exists():
            try:
                benchmark_train_predictions = load_benchmark_predictions(
                    ml_train_pred_path, train_df, "train"
                )
            except (ValueError, KeyError) as exc:
                print(f"[{fold_tag}] Existing explore_ml train predictions are stale: {exc}")
                benchmark_train_predictions = None
        if benchmark_predictions is None or benchmark_train_predictions is None:
            ml_output_dir = _resolve_path(root, args.benchmark_predictions_dir)
            run_python_dependency(
                script_path=root / "code" / "explore_ml.py",
                root=root,
                extra_args=[
                    "--split-dir",
                    str(split_dir),
                    "--output-dir",
                    str(ml_output_dir),
                ],
                label="explore_ml",
            )
            benchmark_predictions = load_benchmark_predictions(ml_pred_path, test_df, "test")
            benchmark_train_predictions = load_benchmark_predictions(
                ml_train_pred_path, train_df, "train"
            )
        print(
            f"[{fold_tag}] Loaded explore_ml predictions: "
            f"{len(benchmark_predictions)} test models, "
            f"{len(benchmark_train_predictions)} train models"
        )

        residual_learning_predictions: dict[str, np.ndarray] | None = None
        residual_train_predictions: dict[str, np.ndarray] | None = None
        if residual_pred_path.exists():
            try:
                residual_learning_predictions = load_residual_learning_predictions(
                    residual_pred_path,
                    test_df,
                    "test",
                )
            except (ValueError, KeyError) as exc:
                print(f"[{fold_tag}] Existing residual test predictions are stale: {exc}")
        if residual_cv_pred_path.exists():
            try:
                residual_train_predictions = load_residual_learning_predictions(
                    residual_cv_pred_path,
                    train_df,
                    "cv",
                )
            except (ValueError, KeyError) as exc:
                print(f"[{fold_tag}] Existing residual CV predictions are stale: {exc}")
        if residual_learning_predictions is None or residual_train_predictions is None:
            residual_group_dir = _resolve_path(root, args.residual_learning_dir)
            residual_output_dir = (
                residual_group_dir.parent
                if residual_group_dir.name.lower() == "all"
                else residual_group_dir
            )
            run_python_dependency(
                script_path=root / "code" / "ml_residual_learning.py",
                root=root,
                extra_args=[
                    "--split-dir",
                    str(split_dir),
                    "--output-dir",
                    str(residual_output_dir),
                ],
                label="residual_learning",
            )
            residual_learning_predictions = load_residual_learning_predictions(
                residual_pred_path,
                test_df,
                "test",
            )
            residual_train_predictions = load_residual_learning_predictions(
                residual_cv_pred_path,
                train_df,
                "cv",
            )
        print(
            f"[{fold_tag}] Loaded residual predictions: "
            f"test={list(residual_learning_predictions)}, "
            f"cv={list(residual_train_predictions)}"
        )

        pgls_predictions = run_pgls_with_ape(
            train_df=train_df,
            test_df=test_df,
            root=root,
            train_path=train_path,
            test_path=test_path,
            tree_path=args.phylo_tree,
            r_script_path=args.pgls_r_script,
            out_dir=pgls_fold_out,
        )
        pgls_train_predictions = load_pgls_train_fitted(
            pgls_fold_out / "pgls_train_fitted.csv",
            train_df,
        )

        (
            metrics_df,
            predictions,
            y_true,
            train_metrics_df,
            train_predictions,
            y_train,
        ) = run_models(
            train_df,
            test_df,
            benchmark_predictions,
            benchmark_train_predictions,
            pgls_predictions,
            pgls_train_predictions,
            residual_learning_predictions,
            residual_train_predictions,
        )
        save_explore_predictions(test_df, y_true, predictions, fold_out, split="test", fold_tag=fold_tag)
        save_explore_predictions(
            train_df, y_train, train_predictions, fold_out, split="train", fold_tag=fold_tag
        )

        linear_metrics = split_linear_metrics(metrics_df)
        linear_path = fold_out / "explore_linear_metrics.csv"
        linear_metrics.to_csv(linear_path, index=False, encoding="utf-8")
        train_linear_metrics = split_linear_metrics(train_metrics_df)
        train_linear_path = fold_out / "explore_linear_metrics_train.csv"
        train_linear_metrics.to_csv(train_linear_path, index=False, encoding="utf-8")

        metrics_path = fold_out / "explore_metrics.csv"
        metrics_out = metrics_df.drop(columns=["model_key"], errors="ignore")
        metrics_out.to_csv(metrics_path, index=False, encoding="utf-8")
        train_metrics_path = fold_out / "explore_metrics_train.csv"
        train_metrics_out = train_metrics_df.drop(columns=["model_key"], errors="ignore")
        train_metrics_out.to_csv(train_metrics_path, index=False, encoding="utf-8")
        protocol_path = write_evaluation_protocol_note(fold_out)

        plot_path = save_model_performance_plot(metrics_out, fold_out, fold_tag=fold_tag)
        top5_residual_plot_path, top5_residual_metrics_path = save_top5_plus_residual_learning_plot(
            metrics_out,
            fold_out,
            fold_tag=fold_tag,
        )
        residual_plot_path = save_residual_plot(y_true, predictions, fold_out)

        print(f"\n[{fold_tag}] TRAIN FIT (RMSE/MAE on log10(BMR)):")
        print(train_metrics_out.to_string(index=False))
        print(f"\n[{fold_tag}] LINEAR / PHYLO M-MTE and M1-M4 (RMSE/MAE on log10(BMR)):")
        print(linear_metrics.to_string(index=False))
        print(f"\n[{fold_tag}] ALL MODELS (linear + ML + residual; RMSE/MAE on log10(BMR)):")
        print(metrics_out.to_string(index=False))
        print(f"[{fold_tag}] Saved linear metrics: {linear_path}")
        print(f"[{fold_tag}] Saved train linear metrics: {train_linear_path}")
        print(f"[{fold_tag}] Saved all metrics: {metrics_path}")
        print(f"[{fold_tag}] Saved train metrics: {train_metrics_path}")
        print(f"[{fold_tag}] Saved evaluation protocol: {protocol_path}")
        print(f"[{fold_tag}] Class-balanced weight formula: {CLASS_BALANCED_WEIGHT_FORMULA}")
        weighted = metrics_out.loc[metrics_out["train_class_weighted"] == 1]
        if not weighted.empty:
            cols = [
                c
                for c in (
                    "model",
                    "rmse",
                    "rmse_macro",
                    "rmse_bal",
                    "mae",
                    "mae_macro",
                    "mae_bal",
                )
                if c in weighted.columns
            ]
            print(f"\n[{fold_tag}] CLASS-WEIGHTED MODELS (micro / macro / bal):")
            print(weighted[cols].to_string(index=False))
        print(f"[{fold_tag}] Saved plot: {plot_path}")
        print(f"[{fold_tag}] Saved top-5 plot: {top5_residual_plot_path}")
        print(f"[{fold_tag}] Saved top-5 metrics: {top5_residual_metrics_path}")
        print(f"[{fold_tag}] Saved residual plot: {residual_plot_path}")

    if fold_tags:
        write_explore_species_accuracy(out_dir, fold_tags)
        write_metrics_by_fold(out_dir, fold_tags)
        write_linear_metrics_by_fold(out_dir, fold_tags)
    else:
        print("Skip explore species accuracy CSV: no evaluation splits completed.")

    print(f"\nWrote held-out evaluation results under: {out_dir}/test/")
    print(f"Wrote fold summary: {out_dir}/explore_metrics_by_fold.csv")
    print(f"Wrote linear fold summary: {out_dir}/explore_linear_metrics_by_fold.csv")


if __name__ == "__main__":
    main()
