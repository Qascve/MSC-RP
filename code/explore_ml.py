#!/usr/bin/env python3
# Train RF/XGB M1–M4 with class-balanced weights; report micro/macro/bal metrics.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBRegressor

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
TEMP_COL = "temperature"
CLADE_COL = "class"
K_BOLTZMANN_EV_PER_K = 8.617e-5
LOG_TARGET = "log_BMR"
SPEC_NAMES = ("m1", "m2", "m3", "m4")
ALGO_NAMES = ("random_forest", "xgboost")

MODEL_FEATURES: dict[str, list[str]] = {
    "m1": ["log_mass"],
    "m2": ["log_mass", "inv_kT"],
    "m3": ["log_mass", "inv_kT", CLADE_COL],
    "m4": ["log_mass", "inv_kT", "pc1", "pc2", "pc3", "pc4", "pc5"],
}

# Matched to ml_residual_learning.py (same grids, early stopping, 4-fold CV).
N_HP_TRIALS = 9
EARLY_STOPPING_ROUNDS = 100
XGB_MAX_ESTIMATORS = 1000
RF_TREE_BATCH = 50
RF_MAX_ESTIMATORS = 1000
CV_FOLD_NAMES = ("fold1", "fold2", "fold3", "fold4")
# Tune HPs once on the richest feature set, reuse across M1–M4.
HP_TUNE_SPEC = "m4"
RF_PARAM_GRID = {
    "max_depth": [4, 6, 8],
}
RF_FIXED_PARAMS = {
    "min_samples_leaf": 5,
    "max_features": 1.0,
}
# Only these keys are searched; all other XGB hyperparameters use library defaults.
XGB_PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.02, 0.03],
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
    out["log_mass"] = np.log10(out[MASS_COL].to_numpy())
    out[LOG_TARGET] = np.log10(out[TARGET].to_numpy())
    return out


def assert_no_species_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    leaked = set(train_df["taxon_name"].astype(str)).intersection(
        set(test_df["taxon_name"].astype(str))
    )
    if leaked:
        raise RuntimeError(f"Species leakage detected: {sorted(leaked)[:5]}")


def species_block_train_val_split(
    train_df: pd.DataFrame, val_frac: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    species = train_df["taxon_name"].astype(str).unique()
    if len(species) < 3:
        raise RuntimeError("Need at least 3 training species for inner val split.")
    rng = np.random.default_rng(random_state)
    order = rng.permutation(species)
    n_val = max(1, int(round(len(order) * val_frac)))
    n_val = min(n_val, len(order) - 1)
    val_species = set(order[:n_val].tolist())
    is_val = train_df["taxon_name"].astype(str).isin(val_species)
    fit_df = train_df.loc[~is_val].reset_index(drop=True)
    val_df = train_df.loc[is_val].reset_index(drop=True)
    if fit_df.empty or val_df.empty:
        raise RuntimeError("Inner species-block split produced an empty fit/val set.")
    return fit_df, val_df


def encode_features(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    feature_cols: list[str],
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = [c for c in feature_cols if c == CLADE_COL]
    train_raw = train_df[feature_cols].reset_index(drop=True).copy()
    other_raw = other_df[feature_cols].reset_index(drop=True).copy()
    if feature_columns is None:
        merged = pd.concat([train_raw, other_raw], axis=0, ignore_index=True)
        encoded = pd.get_dummies(merged, columns=cat_cols, prefix=cat_cols, dtype=float)
        split = len(train_raw)
        return encoded.iloc[:split].copy(), encoded.iloc[split:].copy()
    other_enc = pd.get_dummies(other_raw, columns=cat_cols, prefix=cat_cols, dtype=float)
    other_enc = other_enc.reindex(columns=feature_columns, fill_value=0.0)
    empty = pd.DataFrame(columns=feature_columns)
    return empty, other_enc


def encode_one(
    df: pd.DataFrame, feature_cols: list[str], feature_columns: list[str] | None = None
) -> pd.DataFrame:
    cat_cols = [c for c in feature_cols if c == CLADE_COL]
    raw = df[feature_cols].reset_index(drop=True).copy()
    enc = pd.get_dummies(raw, columns=cat_cols, prefix=cat_cols, dtype=float)
    if feature_columns is not None:
        enc = enc.reindex(columns=feature_columns, fill_value=0.0)
    return enc


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


CLASS_BALANCED_WEIGHT_FORMULA = (
    "w_c = n / (n_classes * n_c) with sklearn class_weight='balanced' "
    "(n = training rows; n_c = rows in class c)."
)


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


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
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
    # Micro, macro, and class-balanced weighted metrics (same as residual learning).
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
    return {
        **micro,
        **macro,
        **bal,
        "train_class_weighted": 1,
    }


def _params_key(params: dict) -> tuple:
    return tuple(sorted((k, params[k]) for k in params))


def _cartesian_param_dicts(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos: list[dict] = [{}]
    for key in keys:
        combos = [{**base, key: val} for base in combos for val in grid[key]]
    return combos


def draw_unique_param_sets(grid: dict, n_trials: int, random_state: int) -> list[dict]:
    # Randomly draw unique parameter combinations without replacement.
    all_combinations = _cartesian_param_dicts(grid)
    total = len(all_combinations)
    if total == 0:
        raise ValueError("Parameter grid produced no combinations.")
    if len({_params_key(params) for params in all_combinations}) != total:
        raise ValueError("XGB_PARAM_GRID contains duplicate parameter combinations.")
    if n_trials < 1:
        raise ValueError(f"n_hp_trials must be >= 1; got {n_trials}.")
    n_draw = min(int(n_trials), total)
    if n_draw == total:
        return list(all_combinations)
    rng = np.random.default_rng(random_state)
    selected = rng.choice(total, size=n_draw, replace=False)
    return [all_combinations[int(i)] for i in selected]


def make_xgb_regressor(
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    random_state: int,
    early_stopping_rounds: int | None = None,
) -> XGBRegressor:
    # Build XGBRegressor using searched HPs only; remaining args stay at XGBoost defaults.
    kwargs: dict = {
        "objective": "reg:squarederror",
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "n_jobs": -1,
        "eval_metric": "rmse",
    }
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRegressor(**kwargs)


def fit_rf_with_early_stopping(
    X_fit,
    y_fit,
    X_val,
    y_val,
    params,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> tuple[RandomForestRegressor, int, float]:
    best_rmse = np.inf
    best_n = RF_TREE_BATCH
    patience_left = max(1, EARLY_STOPPING_ROUNDS // RF_TREE_BATCH)
    rf = RandomForestRegressor(
        n_estimators=RF_TREE_BATCH,
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(RF_FIXED_PARAMS["min_samples_leaf"]),
        max_features=RF_FIXED_PARAMS["max_features"],
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )
    fit_kwargs: dict = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    n_trees = 0
    while n_trees < RF_MAX_ESTIMATORS:
        n_trees += RF_TREE_BATCH
        rf.set_params(n_estimators=n_trees)
        rf.fit(X_fit, y_fit, **fit_kwargs)
        val_rmse = rmse(y_val, rf.predict(X_val))
        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            best_n = n_trees
            patience_left = max(1, EARLY_STOPPING_ROUNDS // RF_TREE_BATCH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    rf_best = RandomForestRegressor(
        n_estimators=best_n,
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(RF_FIXED_PARAMS["min_samples_leaf"]),
        max_features=RF_FIXED_PARAMS["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )
    rf_best.fit(X_fit, y_fit, **fit_kwargs)
    return rf_best, best_n, float(best_rmse)


def fit_xgb_with_early_stopping(
    X_fit,
    y_fit,
    X_val,
    y_val,
    params,
    sample_weight: np.ndarray | None,
    sample_weight_val: np.ndarray | None,
    random_state: int,
) -> tuple[XGBRegressor, int, float]:
    xgb = make_xgb_regressor(
        n_estimators=XGB_MAX_ESTIMATORS,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=random_state,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    fit_kwargs: dict = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if sample_weight_val is not None:
        fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
    xgb.fit(X_fit, y_fit, **fit_kwargs)
    best_n = int(getattr(xgb, "best_iteration", XGB_MAX_ESTIMATORS - 1)) + 1
    best_score = float(getattr(xgb, "best_score", rmse(y_val, xgb.predict(X_val))))
    return xgb, best_n, best_score


def model_key(algo: str, spec: str) -> str:
    return f"{algo}_{spec}"


def to_display_model_name(model_name: str) -> str:
    # Map internal keys like random_forest_m3 -> M3-RF (Table 1 naming).
    if model_name.startswith("random_forest_"):
        suffix = model_name.replace("random_forest_", "", 1)
        return f"{suffix.upper()}-RF"
    if model_name.startswith("xgboost_"):
        suffix = model_name.replace("xgboost_", "", 1)
        return f"{suffix.upper()}-XGB"
    return model_name


def load_four_fold_cv_splits(split_dir: Path) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    # Load fold1–fold4 species-block CV splits (same protocol as residual learning).
    found: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for name in CV_FOLD_NAMES:
        train_path = split_dir / name / "train.csv"
        val_path = split_dir / name / "test.csv"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Missing CV fold under {split_dir / name}. "
                "Expected fold1..fold4 from split_train_test_bmr.py."
            )
        fit_df = add_mte_features(load_split_data(train_path))
        val_df = add_mte_features(load_split_data(val_path))
        assert_no_species_leakage(fit_df, val_df)
        found.append((name, fit_df, val_df))
    return found


def tune_shared_hyperparams(
    cv_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    random_state: int,
    n_trials: int,
) -> dict:
    # 4-fold species-block HP search on HP_TUNE_SPEC features; shared by all M1–M4.
    feature_cols = MODEL_FEATURES[HP_TUNE_SPEC]
    prepared_splits: list[dict] = []
    for fold_tag, fit_df, val_df in cv_splits:
        X_fit, X_val = encode_features(fit_df, val_df, feature_cols)
        prepared_splits.append(
            {
                "fold_tag": fold_tag,
                "X_fit": X_fit,
                "y_fit": fit_df[LOG_TARGET].to_numpy(dtype=float),
                "X_val": X_val,
                "y_val": val_df[LOG_TARGET].to_numpy(dtype=float),
                "sw_fit": make_class_balanced_sample_weight(fit_df),
                "sw_val": make_class_balanced_sample_weight(val_df),
                "val_species": int(val_df["taxon_name"].nunique()),
                "val_rows": int(len(val_df)),
            }
        )

    rf_param_sets = _cartesian_param_dicts(RF_PARAM_GRID)
    if not rf_param_sets:
        raise ValueError("RF_PARAM_GRID produced no combinations.")
    all_combinations = _cartesian_param_dicts(XGB_PARAM_GRID)
    total_combinations = len(all_combinations)
    xgb_param_sets = draw_unique_param_sets(
        XGB_PARAM_GRID, n_trials=n_trials, random_state=random_state
    )
    n_combinations = len(xgb_param_sets)
    if n_combinations == 0:
        raise ValueError("XGB_PARAM_GRID produced no combinations.")

    best_rf: dict | None = None
    best_xgb: dict | None = None
    rf_trials: list[dict] = []
    xgb_trials: list[dict] = []

    print(
        f"  HP search on {HP_TUNE_SPEC} (matched to residual learning): "
        f"RF={len(rf_param_sets)} configs, XGB={n_combinations}/{total_combinations} "
        f"unique trials x {len(prepared_splits)} folds "
        f"(class-balanced sample weights)",
        flush=True,
    )
    for trial, rf_p in enumerate(rf_param_sets):
        fold_scores: list[float] = []
        fold_estimators: list[int] = []
        rf_row = {**rf_p, **RF_FIXED_PARAMS, "trial": trial}
        for fold_idx, split in enumerate(prepared_splits):
            _, rf_n, rf_score = fit_rf_with_early_stopping(
                split["X_fit"],
                split["y_fit"],
                split["X_val"],
                split["y_val"],
                rf_p,
                split["sw_fit"],
                random_state + fold_idx,
            )
            fold_scores.append(rf_score)
            fold_estimators.append(rf_n)
            rf_row[f"{split['fold_tag']}_val_rmse"] = rf_score
            rf_row[f"{split['fold_tag']}_n_estimators"] = rf_n
        rf_row["cv_mean_rmse"] = float(np.mean(fold_scores))
        rf_row["cv_std_rmse"] = float(np.std(fold_scores, ddof=0))
        rf_row["n_estimators"] = max(1, int(round(np.mean(fold_estimators))))
        rf_trials.append(rf_row)
        if best_rf is None or rf_row["cv_mean_rmse"] < best_rf["cv_mean_rmse"]:
            best_rf = dict(rf_row)
        print(
            f"    RF trial {trial + 1}/{len(rf_param_sets)}: "
            f"max_depth={rf_p['max_depth']} "
            f"cv_mean_rmse={rf_row['cv_mean_rmse']:.4f} "
            f"n_estimators={rf_row['n_estimators']}",
            flush=True,
        )

    for trial, xgb_p in enumerate(xgb_param_sets):
        fold_scores = []
        fold_estimators = []
        xgb_row = {**xgb_p, "trial": trial}
        for fold_idx, split in enumerate(prepared_splits):
            _, xgb_n, xgb_score = fit_xgb_with_early_stopping(
                split["X_fit"],
                split["y_fit"],
                split["X_val"],
                split["y_val"],
                xgb_p,
                split["sw_fit"],
                split["sw_val"],
                random_state + fold_idx,
            )
            fold_scores.append(xgb_score)
            fold_estimators.append(xgb_n)
            xgb_row[f"{split['fold_tag']}_val_rmse"] = xgb_score
            xgb_row[f"{split['fold_tag']}_n_estimators"] = xgb_n
        xgb_row["cv_mean_rmse"] = float(np.mean(fold_scores))
        xgb_row["cv_std_rmse"] = float(np.std(fold_scores, ddof=0))
        xgb_row["n_estimators"] = max(1, int(round(np.mean(fold_estimators))))
        xgb_trials.append(xgb_row)
        if best_xgb is None or xgb_row["cv_mean_rmse"] < best_xgb["cv_mean_rmse"]:
            best_xgb = dict(xgb_row)
        if (trial + 1) % 10 == 0 or trial == 0 or trial + 1 == n_combinations:
            print(
                f"    XGB trial {trial + 1}/{n_combinations}: "
                f"cv_mean_rmse={xgb_row['cv_mean_rmse']:.4f} "
                f"best_so_far={best_xgb['cv_mean_rmse']:.4f}",
                flush=True,
            )

    assert best_rf is not None and best_xgb is not None
    print(
        f"  Best RF  trial={int(best_rf['trial']) + 1} "
        f"cv_mean_rmse={best_rf['cv_mean_rmse']:.4f} "
        f"max_depth={best_rf['max_depth']} n={best_rf['n_estimators']}",
        flush=True,
    )
    print(
        f"  Best XGB trial={int(best_xgb['trial']) + 1} "
        f"cv_mean_rmse={best_xgb['cv_mean_rmse']:.4f} "
        f"params={{max_depth={best_xgb['max_depth']}, lr={best_xgb['learning_rate']}}} "
        f"n={best_xgb['n_estimators']}",
        flush=True,
    )
    return {
        "random_forest": best_rf,
        "xgboost": best_xgb,
        "search_trials": {
            "random_forest": pd.DataFrame(rf_trials),
            "xgboost": pd.DataFrame(xgb_trials),
        },
        "inner_val_species": int(np.mean([s["val_species"] for s in prepared_splits])),
        "inner_val_rows": int(np.mean([s["val_rows"] for s in prepared_splits])),
        "cv_folds": [s["fold_tag"] for s in prepared_splits],
        "hp_tune_spec": HP_TUNE_SPEC,
        "train_class_weighted": 1,
        "class_weight_formula": CLASS_BALANCED_WEIGHT_FORMULA,
    }


def train_all_specs(
    train_df: pd.DataFrame, best_params: dict, random_state: int
) -> dict:
    # Fit RF/XGB for every M1–M4 with shared best HPs on full train.
    models: dict[str, object] = {}
    feature_columns: dict[str, list[str]] = {}
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)
    sw_train = make_class_balanced_sample_weight(train_df)

    rf_bp = best_params["random_forest"]
    xgb_bp = best_params["xgboost"]

    for spec, feature_cols in MODEL_FEATURES.items():
        X = encode_one(train_df, feature_cols)
        feature_columns[spec] = list(X.columns)

        rf = RandomForestRegressor(
            n_estimators=int(rf_bp["n_estimators"]),
            max_depth=int(rf_bp["max_depth"]),
            min_samples_leaf=int(RF_FIXED_PARAMS["min_samples_leaf"]),
            max_features=RF_FIXED_PARAMS["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y_train, sample_weight=sw_train)
        models[model_key("random_forest", spec)] = rf

        xgb = make_xgb_regressor(
            n_estimators=int(xgb_bp["n_estimators"]),
            learning_rate=float(xgb_bp["learning_rate"]),
            max_depth=int(xgb_bp["max_depth"]),
            random_state=random_state,
        )
        xgb.fit(X, y_train, sample_weight=sw_train, verbose=False)
        models[model_key("xgboost", spec)] = xgb

    return {"models": models, "feature_columns": feature_columns}


def save_model_bundle(bundle: dict, tune: dict, model_dir: Path, fold_tag: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    keep_names = set(bundle["models"].keys())
    for name, model in bundle["models"].items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    for stale in model_dir.glob("*.joblib"):
        if stale.stem not in keep_names:
            stale.unlink()
    meta = {
        "fold_tag": fold_tag,
        "model_names": list(bundle["models"].keys()),
        "feature_columns": bundle["feature_columns"],
        "best_params": {
            "random_forest": {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in tune["random_forest"].items()
                if k != "model"
            },
            "xgboost": {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in tune["xgboost"].items()
                if k != "model"
            },
        },
        "inner_val_species": tune["inner_val_species"],
        "inner_val_rows": tune["inner_val_rows"],
        "cv_folds": tune.get("cv_folds", list(CV_FOLD_NAMES)),
        "hp_tune_spec": tune.get("hp_tune_spec", HP_TUNE_SPEC),
        "hp_protocol": "4-fold species-block CV on fold1..fold4; matched to ml_residual_learning.py",
        "model_features": MODEL_FEATURES,
        "train_class_weighted": 1,
        "class_weight_formula": CLASS_BALANCED_WEIGHT_FORMULA,
    }
    # JSON-serialize max_features that may be numpy types
    def _jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    (model_dir / "meta.json").write_text(
        json.dumps(_jsonable(meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tune["search_trials"]["random_forest"].to_csv(
        model_dir / "hp_search_random_forest.csv", index=False, encoding="utf-8"
    )
    tune["search_trials"]["xgboost"].to_csv(
        model_dir / "hp_search_xgboost.csv", index=False, encoding="utf-8"
    )
    print(f"  Saved {len(bundle['models'])} models -> {model_dir}", flush=True)
    return model_dir


def load_model_bundle(model_dir: Path) -> dict:
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    models = {}
    for name in meta["model_names"]:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        models[name] = joblib.load(path)
    return {
        "models": models,
        "feature_columns": meta["feature_columns"],
        "model_features": meta.get("model_features", MODEL_FEATURES),
        "best_params": meta.get("best_params", {}),
        "fold_tag": meta.get("fold_tag"),
    }


def predict_with_bundle(bundle: dict, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    model_features = bundle.get("model_features", MODEL_FEATURES)
    for name, model in bundle["models"].items():
        # name = random_forest_m1 / xgboost_m3
        spec = name.rsplit("_", 1)[-1]
        feature_cols = model_features[spec]
        cols = bundle["feature_columns"][spec]
        X = encode_one(test_df, feature_cols, cols)
        preds[name] = np.asarray(model.predict(X), dtype=float)
    return preds


def write_hp_search_trials_csv(
    out_dir: Path, fold_tag: str, tune: dict, reset: bool = False
) -> Path:
    out_path = out_dir / "explore_ml_hp_search_trials.csv"
    rows = []
    for algo in ALGO_NAMES:
        best_trial = int(tune[algo]["trial"])
        for _, rec in tune["search_trials"][algo].iterrows():
            row = {"fold": fold_tag, "model": algo}
            row.update({k: rec[k] for k in tune["search_trials"][algo].columns})
            row["is_best"] = int(int(rec["trial"]) == best_trial)
            rows.append(row)
    new_df = pd.DataFrame(rows)
    if reset or not out_path.exists():
        new_df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        old = pd.read_csv(out_path)
        keep = old[old["fold"].astype(str) != str(fold_tag)] if "fold" in old.columns else old.iloc[0:0]
        pd.concat([keep, new_df], ignore_index=True).to_csv(
            out_path, index=False, encoding="utf-8"
        )
    new_df.to_csv(out_dir / f"explore_ml_hp_search_trials_{fold_tag}.csv", index=False)
    print(f"  Wrote HP log ({len(new_df)} rows) -> {out_path}", flush=True)
    return out_path


def save_outputs(
    fold_out: Path,
    eval_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    fold_tag: str,
    split: str,
) -> pd.DataFrame:
    fold_out.mkdir(parents=True, exist_ok=True)
    classes = eval_df[CLADE_COL].astype(str).to_numpy()
    metric_rows = [
        {"model": name, **evaluate_reporting_suite(y_true, pred, classes)}
        for name, pred in predictions.items()
    ]
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    if split == "test":
        metrics_path = fold_out / "explore_ml_metrics.csv"
        pred_path = fold_out / "explore_ml_predictions_test.csv"
        protocol_path = fold_out / "explore_ml_evaluation_protocol.txt"
    else:
        metrics_path = fold_out / f"explore_ml_metrics_{split}.csv"
        pred_path = fold_out / f"explore_ml_predictions_{split}.csv"
        protocol_path = None
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    if protocol_path is not None:
        protocol_path.write_text(
            "\n".join(
                [
                    "Evaluation protocol (explore_ml.py)",
                    "",
                    "Hyper-parameter search (matched to ml_residual_learning.py):",
                    "- 4-fold species-block CV on fold1..fold4",
                    f"- Feature set for tuning: {HP_TUNE_SPEC}; selected HPs reused for M1–M4",
                    "- RF search: max_depth in {4,6,8}; fixed min_samples_leaf=5, max_features=1.0",
                    "- XGB search: max_depth in {4,6,8} x learning_rate in {0.01,0.02,0.03}",
                    "  (other XGB args use library defaults)",
                    "- n_estimators chosen by early stopping (patience 100; max 1000)",
                    "",
                    "Training:",
                    f"- Class-balanced sample weights: {CLASS_BALANCED_WEIGHT_FORMULA}",
                    "- Applied to RF/XGB HP search and final M1–M4 retraining on full development set.",
                    "",
                    "Reported metrics:",
                    "- rmse/mae/r2: micro-averaged over pooled evaluation rows",
                    "- rmse_macro/mae_macro/r2_macro: mean of per-class micro metrics",
                    "- rmse_bal/mae_bal/r2_bal: class-balanced weighted metrics",
                    "  (same w_c formula as training)",
                    "- train_class_weighted=1 for all explore_ml models",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    pred_df = eval_df[["taxon_name", CLADE_COL, "log_mass", "inv_kT"]].copy()
    pred_df["y_true"] = y_true
    pred_df["fold"] = fold_tag
    pred_df["eval_split"] = split
    for name, pred in predictions.items():
        pred_df[name] = pred
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    if split != "test":
        print(f"[{fold_tag}] Saved {split} metrics/predictions -> {metrics_path.name}, {pred_path.name}")
        print(metrics_df.to_string(index=False))
        return metrics_df

    plot_df = metrics_df.copy()
    plot_df["model"] = plot_df["model"].map(to_display_model_name)

    sns.set_theme(style="whitegrid")
    fig_width = max(12.0, 0.8 * len(plot_df) + 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))
    sns.barplot(data=plot_df, x="model", y="rmse", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE (log10(BMR))")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=plot_df, x="model", y="r2", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2 (log10(BMR))")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(f"ML Model Performance ({fold_tag})", fontsize=14)
    fig.tight_layout()
    fig.savefig(fold_out / "explore_ml_model_performance_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # Class-balanced companion plot.
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))
    sns.barplot(data=plot_df, x="model", y="rmse_bal", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE_bal (log10(BMR))")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=plot_df, x="model", y="r2_bal", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2_bal (log10(BMR))")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(f"ML Model Performance class-balanced ({fold_tag})", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        fold_out / "explore_ml_model_performance_comparison_bal.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    plt.figure(figsize=(9, 7))
    for name, pred in predictions.items():
        plt.scatter(pred, y_true - pred, s=10, alpha=0.35, label=name)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Predicted log10(BMR)")
    plt.ylabel("Residual")
    plt.title(f"ML Residual Plot ({fold_tag})")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fold_out / "explore_ml_residual_plot.pdf", bbox_inches="tight")
    plt.close()

    print(f"[{fold_tag}] Saved metrics/predictions under {fold_out}")
    print(f"[{fold_tag}] Class-balanced weight formula: {CLASS_BALANCED_WEIGHT_FORMULA}")
    print(metrics_df.to_string(index=False))
    return metrics_df


def run_cv_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_tag: str,
    out_dir: Path,
    random_state: int,
    n_hp_trials: int,
    reset_hp_log: bool,
    cv_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> None:
    model_dir = out_dir / fold_tag / "models"
    print(f"  Tuning + training M1–M4 on {len(train_df)} train rows...", flush=True)
    tune = tune_shared_hyperparams(cv_splits, random_state, n_hp_trials)
    bundle = train_all_specs(train_df, tune, random_state)
    save_model_bundle(bundle, tune, model_dir, fold_tag)
    write_hp_search_trials_csv(out_dir, fold_tag, tune, reset=reset_hp_log)

    print("  Reloading models from disk for evaluation...", flush=True)
    loaded = load_model_bundle(model_dir)
    fold_out = out_dir / fold_tag

    train_preds = predict_with_bundle(loaded, train_df)
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)
    save_outputs(fold_out, train_df, y_train, train_preds, fold_tag, split="train")

    test_preds = predict_with_bundle(loaded, test_df)
    y_test = test_df[LOG_TARGET].to_numpy(dtype=float)
    save_outputs(fold_out, test_df, y_test, test_preds, fold_tag, split="test")


def log_bmr_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    out = np.full(len(y_true_log), np.nan, dtype=float)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    out[mask] = 10.0 ** (-np.abs(y_pred_log[mask] - y_true_log[mask]))
    return np.clip(out, 0.0, 1.0)


def write_explore_ml_species_accuracy(out_dir: Path, fold_tags: list[str]) -> Path:
    frames = []
    for tag in fold_tags:
        path = out_dir / tag / "explore_ml_predictions_test.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        fold_df = pd.read_csv(path)
        fold_df["eval_split"] = tag
        frames.append(fold_df)
    stitched = pd.concat(frames, ignore_index=True)
    model_cols = [
        c
        for c in stitched.columns
        if c.startswith("random_forest_") or c.startswith("xgboost_")
    ]
    if not model_cols:
        raise KeyError("No ML model prediction columns for species accuracy.")
    work = stitched.copy()
    y_true = pd.to_numeric(work["y_true"], errors="coerce").to_numpy(dtype=float)
    for model in model_cols:
        work[model] = log_bmr_accuracy(
            y_true, pd.to_numeric(work[model], errors="coerce").to_numpy(dtype=float)
        )
    accuracy_df = (
        work.groupby("taxon_name", as_index=False)[model_cols]
        .mean(numeric_only=True)
        .sort_values("taxon_name")
        .reset_index(drop=True)
    )
    out_path = out_dir / "explore_ml_species_accuracy.csv"
    accuracy_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"[species accuracy explore_ml] species={len(accuracy_df)}, "
        f"splits={fold_tags}, models={len(model_cols)} -> {out_path}"
    )
    return out_path


def discover_fold_splits(split_dir: Path, folds: list[str]) -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    for name in folds:
        train_path = split_dir / name / "train.csv"
        test_path = split_dir / name / "test.csv"
        if train_path.exists() and test_path.exists():
            found.append((name, train_path, test_path))
    if not found:
        raise FileNotFoundError(
            f"No requested split found under {split_dir}. Expected test/."
        )
    return found


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Tune RF/XGB M1–M4 with the same 4-fold species-block HP protocol as "
            "ml_residual_learning.py (XGB grid: max_depth x learning_rate), "
            "retrain on the complete development set, and evaluate on held-out test."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/explore"))
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-hp-trials", type=int, default=N_HP_TRIALS)
    args = parser.parse_args()

    split_dir = _resolve_path(root, args.split_dir)
    out_dir = _resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_splits = load_four_fold_cv_splits(split_dir)
    fold_splits = discover_fold_splits(split_dir, ["test"])
    _, train_path, test_path = fold_splits[0]
    train_df = add_mte_features(load_split_data(train_path))
    test_df = add_mte_features(load_split_data(test_path))
    assert_no_species_leakage(train_df, test_df)

    print(
        "\n=== explore_ml test (4-fold CV HP tune + development-train/test-eval) ===",
        flush=True,
    )
    run_cv_fold(
        train_df=train_df,
        test_df=test_df,
        fold_tag="test",
        out_dir=out_dir,
        random_state=args.random_state,
        n_hp_trials=args.n_hp_trials,
        reset_hp_log=True,
        cv_splits=cv_splits,
    )
    write_explore_ml_species_accuracy(out_dir, ["test"])
    print(f"\nWrote explore_ml test results under: {out_dir}/test/")
    print(f"HP search log: {out_dir}/explore_ml_hp_search_trials.csv")


if __name__ == "__main__":
    main()
