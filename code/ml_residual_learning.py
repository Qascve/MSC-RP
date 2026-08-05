from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBRegressor

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
LOG_TARGET = "log_BMR"
LOG10_SCALE = "log10_BMR"
MODEL_NAMES = ["random_forest", "xgboost"]
CLADE_COL = "class"
TAXONOMY_MODEL_FEATURES = ["class"]
TAXONOMY_METADATA_COLUMNS = ["class", "order", "family", "Genus", "species"]
BASELINE_MODEL = "m3_linear"
KEEP_CLASSES = {
    "Teleostei",
    "Insecta",
    "Mammalia",
    "Malacostraca",
    "Reptilia",
    "Amphibia",
    "Maxillopoda",
    "Aves",
    "Arachnida",
    "Chondrichthyes",
    "Cephalaspidomorphi",
    "Chondrostei",
    "Branchiopoda",
    "Cephalopoda",
    "Sagittoidea",
    "Hydrozoa",
    "Dipnotetrapodomorpha",
    "Ostracoda",
    "Scyphozoa",
    "Myxini",
    "Chilopoda",
    "Cladistei",
    "Gastropoda",
}
GROUP_CLASS_FILTERS: dict[str, str | None] = {
    "all": None,
    "Teleostei": "Teleostei",
    "Mammalia": "Mammalia",
    "Aves": "Aves",
    "Insecta": "Insecta",
    "Malacostraca": "Malacostraca",
    "Amphibia": "Amphibia",
    "Reptilia": "Reptilia",
    "Cephalopoda": "Cephalopoda",
}
FOLD_DIR_NAMES = {
    "fold1": "f_1",
    "fold2": "f_2",
    "fold3": "f_3",
    "fold4": "f_4",
    "test": "test",
}
TREE_MODEL_FEATURES = [
    *TAXONOMY_MODEL_FEATURES,
    "log_mass",
    "inv_kT",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pc5",
]

EARLY_STOPPING_ROUNDS = 100
XGB_MAX_ESTIMATORS = 1000
RF_TREE_BATCH = 50
RF_MAX_ESTIMATORS = 1000
N_HP_TRIALS = 9
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


def _assert_log10_target_columns(df: pd.DataFrame, path: Path) -> None:
    """Reject stale split CSVs that still store natural-log targets."""
    if LOG_TARGET not in df.columns or TARGET not in df.columns:
        return
    csv_log = pd.to_numeric(df[LOG_TARGET], errors="coerce").to_numpy(dtype=float)
    bmr = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(csv_log) & np.isfinite(bmr) & (bmr > 0)
    if not bool(mask.any()):
        return
    log10_bmr = np.log10(bmr[mask])
    if np.allclose(csv_log[mask], log10_bmr, rtol=1e-8, atol=1e-10):
        return
    ln_bmr = np.log(bmr[mask])
    msg = (
        f"{path.name}: column {LOG_TARGET} does not match log10({TARGET}). "
        "Recompute splits with: python code/split_train_test_bmr.py"
    )
    if np.allclose(csv_log[mask], ln_bmr, rtol=1e-8, atol=1e-10):
        msg += " (values look like natural log / ln)."
    raise ValueError(msg)


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep_cols = list(
        dict.fromkeys(
            [
                "taxon_name",
                *TAXONOMY_METADATA_COLUMNS,
                *TREE_MODEL_FEATURES,
                MASS_COL,
                TARGET,
                LOG_TARGET,
            ]
        )
    )
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    _assert_log10_target_columns(df, path)

    out = df[keep_cols].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    for col in TAXONOMY_METADATA_COLUMNS:
        out[col] = out[col].astype("string").str.strip()
    numeric_features = ["log_mass", "inv_kT", "pc1", "pc2", "pc3", "pc4", "pc5"]
    for col in numeric_features + [MASS_COL, TARGET, LOG_TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["taxon_name"] = out["taxon_name"].replace("", pd.NA)
    for col in TAXONOMY_METADATA_COLUMNS:
        out[col] = out[col].replace("", pd.NA)
    out = out.dropna(subset=keep_cols).copy()
    out = out[(out[MASS_COL] > 0) & (out[TARGET] > 0)].copy()
    out = out[(out["log_mass"].notna()) & (out["inv_kT"].notna())].copy()
    out["log_mass"] = np.log10(out[MASS_COL].to_numpy(dtype=float))
    out[LOG_TARGET] = np.log10(out[TARGET].to_numpy(dtype=float))
    out = out[out["taxon_name"] != ""].copy()
    out = out[out["class"].isin(KEEP_CLASSES)].copy()
    return out.reset_index(drop=True)


def discover_fold_splits(split_dir: Path, folds: list[str] | None = None) -> list[tuple[str, Path, Path]]:
    """
    Discover fixed fold CSVs under split_dir.
    Prefers fold1/, fold2/, test/; falls back to top-level train.csv/test.csv as fold1.
    """
    wanted = folds if folds else ["fold1", "fold2", "fold3", "fold4", "test"]
    found: list[tuple[str, Path, Path]] = []
    for name in wanted:
        train_path = split_dir / name / "train.csv"
        test_path = split_dir / name / "test.csv"
        if train_path.exists() and test_path.exists():
            found.append((name, train_path, test_path))

    if found:
        return found

    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    if train_path.exists() and test_path.exists() and (folds is None or "fold1" in wanted):
        return [("fold1", train_path, test_path)]

    raise FileNotFoundError(
        f"No fixed split CSVs found under {split_dir}. "
        "Expected fold1/..fold4/ and test/ with train.csv & test.csv. "
        "Run: python code/split_train_test_bmr.py"
    )


def assert_no_species_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    leaked = set(train_df["taxon_name"].astype(str)).intersection(
        set(test_df["taxon_name"].astype(str))
    )
    if leaked:
        raise RuntimeError(f"Species leakage detected in fixed split: {sorted(leaked)[:5]}")


def fit_m3_baseline(train_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Fit M3-L: log10(BMR) ~ log_mass + inv_kT + class (additive OLS)."""
    clade_levels = sorted(train_df[CLADE_COL].dropna().astype(str).unique().tolist())
    if not clade_levels:
        raise ValueError("No class levels available to fit the M3-L baseline.")
    X = build_m3_design_matrix(train_df, clade_levels)
    y = train_df[LOG_TARGET].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, clade_levels


def build_m3_design_matrix(df: pd.DataFrame, clade_levels: list[str]) -> np.ndarray:
    clade_dummies = pd.get_dummies(df[CLADE_COL].astype(str), dtype=float).reindex(
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


def predict_m3_baseline(
    df: pd.DataFrame, coef: np.ndarray, clade_levels: list[str]
) -> np.ndarray:
    return build_m3_design_matrix(df, clade_levels) @ coef


def make_class_balanced_sample_weight(train_df: pd.DataFrame) -> np.ndarray:
    classes = train_df["class"].to_numpy()
    unique_classes = np.unique(classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes,
    )
    weight_map = dict(zip(unique_classes, class_weights))
    return np.array([weight_map[c] for c in classes], dtype=float)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _params_key(params: dict) -> tuple:
    return tuple(sorted((k, params[k]) for k in params))


def _cartesian_param_dicts(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos: list[dict] = [{}]
    for key in keys:
        combos = [{**base, key: val} for base in combos for val in grid[key]]
    return combos


def draw_unique_param_sets(grid: dict, n_trials: int, random_state: int) -> list[dict]:
    """Randomly draw unique parameter combinations without replacement."""
    all_combinations = _cartesian_param_dicts(grid)
    total = len(all_combinations)
    if total == 0:
        raise ValueError("Parameter grid produced no combinations.")
    if len({_params_key(params) for params in all_combinations}) != total:
        raise ValueError("XGB_PARAM_GRID contains duplicate parameter combinations.")
    if n_trials < 1:
        raise ValueError(f"n_hp_trials must be >= 1; got {n_trials}.")
    # Small grids (e.g. only max_depth × learning_rate) may have fewer combos than N_HP_TRIALS.
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
    """Build XGBRegressor using searched HPs only; remaining args stay at XGBoost defaults."""
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


def encode_train_frame(
    df: pd.DataFrame,
    m3_coef: np.ndarray,
    m3_clade_levels: list[str],
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Encode residual features for one frame; optionally align to saved columns."""
    cat = [c for c in TREE_MODEL_FEATURES if c in TAXONOMY_MODEL_FEATURES]
    raw = df[TREE_MODEL_FEATURES].reset_index(drop=True).copy()
    encoded = pd.get_dummies(raw, columns=cat, prefix=cat, dtype=float)
    base = predict_m3_baseline(df, m3_coef, m3_clade_levels)
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)
    residual = df[LOG_TARGET].to_numpy(dtype=float) - base
    return encoded, residual, base


def fit_rf_with_early_stopping(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> tuple[RandomForestRegressor, int, float]:
    """Grow RF in batches and stop when validation RMSE stops improving."""
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
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None,
    sample_weight_val: np.ndarray | None,
    random_state: int,
) -> tuple[XGBRegressor, int, float, list[float]]:
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
    evals = xgb.evals_result().get("validation_0", {}).get("rmse", [])
    history = [float(v) for v in evals]
    return xgb, best_n, best_score, history


def tune_models_on_train(
    cv_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    full_train_df: pd.DataFrame,
    random_state: int,
    balance_classes: bool,
    n_hp_trials: int,
) -> dict:
    """
    Tune RF/XGB by fixed four-fold species-block CV with early stopping, then
    retrain both on the complete 80% development set.

    Each CV M3-L baseline is fitted only on that fold's three training
    partitions. Validation targets therefore never contribute to the baseline.
    """
    if len(cv_splits) != 4:
        raise ValueError(f"Expected exactly four CV splits, got {len(cv_splits)}.")

    prepared_splits: list[dict] = []
    for fold_tag, fit_df, val_df in cv_splits:
        assert_no_species_leakage(fit_df, val_df)
        m3_coef_inner, m3_clade_levels_inner = fit_m3_baseline(fit_df)
        X_fit, residual_fit, _ = encode_train_frame(
            fit_df, m3_coef_inner, m3_clade_levels_inner
        )
        feature_columns_inner = list(X_fit.columns)
        X_val, residual_val, _ = encode_train_frame(
            val_df, m3_coef_inner, m3_clade_levels_inner, feature_columns_inner
        )
        prepared_splits.append(
            {
                "fold_tag": fold_tag,
                "X_fit": X_fit,
                "residual_fit": residual_fit,
                "X_val": X_val,
                "residual_val": residual_val,
                "sw_fit": (
                    make_class_balanced_sample_weight(fit_df)
                    if balance_classes
                    else None
                ),
                "sw_val": (
                    make_class_balanced_sample_weight(val_df)
                    if balance_classes
                    else None
                ),
                "val_species": int(val_df["taxon_name"].nunique()),
                "val_rows": int(len(val_df)),
                "m3_n_clade_levels": len(m3_clade_levels_inner),
            }
        )

    rf_param_sets = _cartesian_param_dicts(RF_PARAM_GRID)
    if not rf_param_sets:
        raise ValueError("RF_PARAM_GRID produced no combinations.")

    all_combinations = _cartesian_param_dicts(XGB_PARAM_GRID)
    total_combinations = len(all_combinations)
    xgb_param_sets = draw_unique_param_sets(
        XGB_PARAM_GRID, n_trials=n_hp_trials, random_state=random_state
    )
    n_combinations = len(xgb_param_sets)
    if n_combinations == 0:
        raise ValueError("XGB_PARAM_GRID produced no combinations.")
    xgb_keys = {_params_key(p) for p in xgb_param_sets}
    if len(xgb_keys) != n_combinations:
        raise RuntimeError("Hyperparameter search produced duplicate trials.")

    rf_trials: list[dict] = []
    best_rf: dict | None = None
    print(
        f"  RF four-fold CV: {len(rf_param_sets)} max_depth configs "
        f"x {len(prepared_splits)} folds (early stopping)",
        flush=True,
    )
    for trial, rf_params in enumerate(rf_param_sets):
        fold_scores: list[float] = []
        fold_estimators: list[int] = []
        rf_row = {**rf_params, **RF_FIXED_PARAMS, "trial": trial}
        for fold_idx, split in enumerate(prepared_splits):
            _, rf_n, rf_score = fit_rf_with_early_stopping(
                split["X_fit"],
                split["residual_fit"],
                split["X_val"],
                split["residual_val"],
                rf_params,
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
            f"max_depth={rf_params['max_depth']} "
            f"cv_mean_rmse={rf_row['cv_mean_rmse']:.4f} "
            f"n_estimators={rf_row['n_estimators']}",
            flush=True,
        )

    xgb_trials: list[dict] = []
    best_xgb: dict | None = None
    best_xgb_histories: list[list[float]] = []

    print(
        f"  XGB random four-fold CV: {n_combinations} unique combinations "
        f"drawn from {total_combinations} x {len(prepared_splits)} folds",
        flush=True,
    )
    for trial, xgb_params in enumerate(xgb_param_sets):
        fold_scores = []
        fold_estimators = []
        fold_histories: list[list[float]] = []
        xgb_row = {**xgb_params, "trial": trial}
        for fold_idx, split in enumerate(prepared_splits):
            _, xgb_n, xgb_score, xgb_hist = fit_xgb_with_early_stopping(
                split["X_fit"],
                split["residual_fit"],
                split["X_val"],
                split["residual_val"],
                xgb_params,
                split["sw_fit"],
                split["sw_val"],
                random_state + fold_idx,
            )
            fold_scores.append(xgb_score)
            fold_estimators.append(xgb_n)
            fold_histories.append(xgb_hist)
            xgb_row[f"{split['fold_tag']}_val_rmse"] = xgb_score
            xgb_row[f"{split['fold_tag']}_n_estimators"] = xgb_n
        xgb_row["cv_mean_rmse"] = float(np.mean(fold_scores))
        xgb_row["cv_std_rmse"] = float(np.std(fold_scores, ddof=0))
        xgb_row["n_estimators"] = max(1, int(round(np.mean(fold_estimators))))
        xgb_trials.append(xgb_row)
        if best_xgb is None or xgb_row["cv_mean_rmse"] < best_xgb["cv_mean_rmse"]:
            best_xgb = dict(xgb_row)
            best_xgb_histories = [list(hist) for hist in fold_histories]
        if (trial + 1) % 10 == 0 or trial == 0:
            print(
                f"    combination {trial + 1}/{n_combinations}: "
                f"best_so_far cv_mean_rmse={best_xgb['cv_mean_rmse']:.4f} "
                f"(trial={int(best_xgb['trial']) + 1})",
                flush=True,
            )

    assert best_rf is not None and best_xgb is not None

    rf_n_final = int(best_rf["n_estimators"])
    xgb_n_final = int(best_xgb["n_estimators"])
    max_history = max((len(hist) for hist in best_xgb_histories), default=0)
    history_matrix = np.full((len(best_xgb_histories), max_history), np.nan)
    for row_idx, hist in enumerate(best_xgb_histories):
        history_matrix[row_idx, : len(hist)] = hist
    xgb_hist = (
        np.nanmean(history_matrix, axis=0).astype(float).tolist()
        if max_history
        else []
    )

    # Re-estimate M3-L only after selection, now using all four development folds.
    m3_coef_full, m3_clade_levels_full = fit_m3_baseline(full_train_df)
    X_all, residual_all, _ = encode_train_frame(
        full_train_df, m3_coef_full, m3_clade_levels_full
    )
    feature_columns = list(X_all.columns)
    sw_all = (
        make_class_balanced_sample_weight(full_train_df)
        if balance_classes
        else None
    )

    rf_full = RandomForestRegressor(
        n_estimators=rf_n_final,
        max_depth=int(best_rf["max_depth"]),
        min_samples_leaf=int(RF_FIXED_PARAMS["min_samples_leaf"]),
        max_features=RF_FIXED_PARAMS["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )
    rf_full.fit(X_all, residual_all, sample_weight=sw_all)

    xgb_full = make_xgb_regressor(
        n_estimators=xgb_n_final,
        learning_rate=best_xgb["learning_rate"],
        max_depth=best_xgb["max_depth"],
        random_state=random_state,
    )
    xgb_full.fit(X_all, residual_all, sample_weight=sw_all, verbose=False)

    print(
        f"  Best RF trial={int(best_rf['trial']) + 1} "
        f"cv_mean_rmse={best_rf['cv_mean_rmse']:.4f} "
        f"max_depth={best_rf['max_depth']} n_estimators={rf_n_final}",
        flush=True,
    )
    print(
        f"  Best XGB trial={int(best_xgb['trial']) + 1} "
        f"cv_mean_rmse={best_xgb['cv_mean_rmse']:.4f} "
        f"cv_std_rmse={best_xgb['cv_std_rmse']:.4f} "
        f"n_estimators={xgb_n_final} "
        f"params={{max_depth={best_xgb['max_depth']}, lr={best_xgb['learning_rate']}}}",
        flush=True,
    )

    return {
        "models": {"random_forest": rf_full, "xgboost": xgb_full},
        "m3_coef": m3_coef_full,
        "m3_clade_levels": m3_clade_levels_full,
        "feature_columns": feature_columns,
        "best_params": {
            "random_forest": {
                "trial": int(best_rf["trial"]),
                "max_depth": int(best_rf["max_depth"]),
                "min_samples_leaf": int(RF_FIXED_PARAMS["min_samples_leaf"]),
                "max_features": RF_FIXED_PARAMS["max_features"],
                "n_estimators": rf_n_final,
                "cv_mean_rmse": float(best_rf["cv_mean_rmse"]),
                "cv_std_rmse": float(best_rf["cv_std_rmse"]),
            },
            "xgboost": {
                "trial": int(best_xgb["trial"]),
                "max_depth": int(best_xgb["max_depth"]),
                "learning_rate": float(best_xgb["learning_rate"]),
                "n_estimators": xgb_n_final,
                "cv_mean_rmse": float(best_xgb["cv_mean_rmse"]),
                "cv_std_rmse": float(best_xgb["cv_std_rmse"]),
            },
        },
        "search_trials": {
            "random_forest": pd.DataFrame(rf_trials),
            "xgboost": pd.DataFrame(xgb_trials),
        },
        "loss_curves": {
            "xgboost": xgb_hist,
        },
        "inner_val_species": int(sum(s["val_species"] for s in prepared_splits)),
        "inner_val_rows": int(sum(s["val_rows"] for s in prepared_splits)),
        "cv_folds": [
            {
                "fold": split["fold_tag"],
                "val_species": split["val_species"],
                "val_rows": split["val_rows"],
                "m3_n_clade_levels": split["m3_n_clade_levels"],
            }
            for split in prepared_splits
        ],
        "fold_tag": None,
    }


def save_model_bundle(bundle: dict, model_dir: Path) -> Path:
    """Save RF and XGB separately for this fold."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in ("model.joblib",):
        path = model_dir / obsolete
        if path.exists():
            path.unlink()

    for name in MODEL_NAMES:
        joblib.dump(bundle["models"][name], model_dir / f"{name}.joblib")
    meta = {
        "baseline_model": BASELINE_MODEL,
        "m3_coef": [float(v) for v in bundle["m3_coef"]],
        "m3_clade_levels": list(bundle["m3_clade_levels"]),
        "feature_columns": bundle["feature_columns"],
        "best_params": bundle["best_params"],
        "inner_val_species": bundle["inner_val_species"],
        "inner_val_rows": bundle["inner_val_rows"],
        "cv_folds": bundle.get("cv_folds", []),
        "model_features": TREE_MODEL_FEATURES,
        "fold_tag": bundle.get("fold_tag"),
    }
    (model_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Persist HP-search records; XGB also keeps an early-stopping curve.
    for algo_name, trials_df in bundle["search_trials"].items():
        trials_df.to_csv(
            model_dir / f"hp_search_{algo_name}.csv", index=False, encoding="utf-8"
        )
    hist = bundle["loss_curves"].get("xgboost", [])
    pd.DataFrame(
        {"iteration": np.arange(1, len(hist) + 1, dtype=int), "val_rmse": hist}
    ).to_csv(model_dir / "loss_curve.csv", index=False, encoding="utf-8")
    for leftover in model_dir.glob("loss_curve_*.csv"):
        leftover.unlink()
    print(
        f"  Saved RF and XGB separately -> {model_dir / 'random_forest.joblib'}, "
        f"{model_dir / 'xgboost.joblib'}",
        flush=True,
    )
    return model_dir


def write_hp_search_trials_csv(
    benchmark_dir: Path,
    fold_tag: str,
    bundle: dict,
    model_dir: Path | None = None,
) -> Path:
    """
    Write/append all sampled RF/XGB combinations + four-fold CV RMSE.

    Fixed filenames are overwritten on every run. If a CSV is open and locked
    on Windows, write an ``*_unlocked.csv`` fallback and continue instead of
    losing the completed search.
    """
    def write_with_fallback(df: pd.DataFrame, path: Path) -> Path:
        try:
            df.to_csv(path, index=False, encoding="utf-8")
            return path
        except PermissionError:
            fallback = path.with_name(f"{path.stem}_unlocked{path.suffix}")
            df.to_csv(fallback, index=False, encoding="utf-8")
            print(
                f"  Warning: {path} is locked; wrote {fallback} instead.",
                flush=True,
            )
            return fallback

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    out_path = benchmark_dir / "hp_search_trials.csv"
    rows: list[dict] = []
    for model_name, trials_df in bundle["search_trials"].items():
        best_trial = int(bundle["best_params"][model_name]["trial"])
        for _, rec in trials_df.iterrows():
            row = {"fold": fold_tag, "model": model_name}
            row.update({k: rec[k] for k in trials_df.columns})
            is_family_best = int(int(rec["trial"]) == best_trial)
            row["is_best"] = is_family_best
            rows.append(row)
    new_df = pd.DataFrame(rows)

    actual_out_path = write_with_fallback(new_df, out_path)
    best_df = new_df.loc[new_df["is_best"] == 1].copy()
    best_path = write_with_fallback(
        best_df, benchmark_dir / "xgb_best_params.csv"
    )

    model_dir = model_dir or benchmark_dir / "all" / fold_tag / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_with_fallback(best_df, model_dir / "xgb_best_params.csv")

    print(
        f"  Wrote HP random search ({len(new_df)} rows) -> {actual_out_path}",
        flush=True,
    )
    print(
        f"  Wrote best parameters for {fold_tag} -> {best_path}",
        flush=True,
    )
    return actual_out_path


def load_model_bundle(model_dir: Path) -> dict:
    """Load both separately persisted model families from one fold."""
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    models = {}
    for name in MODEL_NAMES:
        model_path = model_dir / f"{name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {name} model: {model_path}. Rerun this fold."
            )
        models[name] = joblib.load(model_path)
    if "m3_coef" not in meta:
        raise ValueError(
            f"Stale model bundle at {model_dir}: expected M3-L baseline metadata. "
            "Rerun ml_residual_learning.py."
        )
    return {
        "models": models,
        "m3_coef": np.asarray(meta["m3_coef"], dtype=float),
        "m3_clade_levels": list(meta["m3_clade_levels"]),
        "feature_columns": list(meta["feature_columns"]),
        "best_params": meta.get("best_params", {}),
        "model_features": list(meta.get("model_features", TREE_MODEL_FEATURES)),
    }


def predict_log_bmr(
    bundle: dict, df: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Predict log10(BMR) with a loaded bundle; also return residual feature frames for SHAP."""
    m3_coef = bundle["m3_coef"]
    m3_clade_levels = bundle["m3_clade_levels"]
    feature_columns = bundle["feature_columns"]
    model_features = bundle.get("model_features", TREE_MODEL_FEATURES)
    cat = [c for c in model_features if c in TAXONOMY_MODEL_FEATURES]
    raw = df[model_features].reset_index(drop=True).copy()
    encoded = pd.get_dummies(raw, columns=cat, prefix=cat, dtype=float)
    base = predict_m3_baseline(df, m3_coef, m3_clade_levels)
    X = encoded.reindex(columns=feature_columns, fill_value=0.0)

    preds: dict[str, np.ndarray] = {}
    shap_inputs: dict[str, pd.DataFrame] = {}
    for name, model in bundle["models"].items():
        residual_hat = np.asarray(model.predict(X), dtype=float)
        preds[name] = base + residual_hat
        shap_inputs[name] = X
    return preds, shap_inputs


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Micro-averaged metrics on pooled log10(BMR) observations."""
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = np.asarray(y_true_log, dtype=float)[mask]
    y_pred_log = np.asarray(y_pred_log, dtype=float)[mask]
    if len(y_true_log) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    r2 = float("nan")
    if len(y_true_log) >= 2 and not np.isclose(np.var(y_true_log), 0.0):
        r2 = float(r2_score(y_true_log, y_pred_log))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2": r2,
    }


CLASS_BALANCED_WEIGHT_FORMULA = (
    "w_c = n / (n_classes * n_c) with sklearn class_weight='balanced' "
    "(n = training rows; n_c = rows in class c)."
)


def evaluate_weighted(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
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
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Micro (pooled), macro (equal class average), and class-balanced weighted metrics.
    Residual RF/XGB are trained with the same balanced weights.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    micro = evaluate(y_true_log, y_pred_log)
    if classes is None:
        return {
            **micro,
            "rmse_macro": micro["rmse"],
            "mae_macro": micro["mae"],
            "r2_macro": micro["r2"],
            "n_classes_eval": 1,
            "rmse_bal": micro["rmse"],
            "mae_bal": micro["mae"],
            "r2_bal": micro["r2"],
            "train_class_weighted": 1,
        }
    classes = np.asarray(classes)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = y_true_log[mask]
    y_pred_log = y_pred_log[mask]
    classes = classes[mask]
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


def save_loss_curve_from_bundle(model_dir: Path, out_dir: Path) -> None:
    """Copy/plot the XGB early-stopping validation loss curve."""
    src = model_dir / "loss_curve.csv"
    if not src.exists():
        # Backward compatibility
        for alt in ("loss_curve_xgboost.csv", "loss_curve_random_forest.csv"):
            if (model_dir / alt).exists():
                src = model_dir / alt
                break
        else:
            return
    lc_df = pd.read_csv(src)
    lc_df.to_csv(out_dir / "loss_curve_data.csv", index=False, encoding="utf-8")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    ycol = "val_rmse" if "val_rmse" in lc_df.columns else lc_df.columns[-1]
    plt.plot(lc_df["iteration"], lc_df[ycol], label="val_rmse", linewidth=2)
    plt.xlabel("XGB boosting iteration")
    plt.ylabel("RMSE (log10(BMR) residual)")
    plt.title("XGBoost Early-Stopping Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.pdf", bbox_inches="tight")
    plt.close()


def prediction_model_cols(pred_df: pd.DataFrame) -> list[str]:
    """Model prediction columns present in a prediction frame."""
    cols = [c for c in MODEL_NAMES if c in pred_df.columns]
    if cols:
        return cols
    if "y_pred" in pred_df.columns:
        return ["y_pred"]
    raise KeyError("No model prediction columns found in prediction frame.")


def save_pred_and_residual_plots(
    out_dir: Path, pred_df: pd.DataFrame, model_names: list[str] | None = None
) -> None:
    sns.set_theme(style="whitegrid")
    model_names = model_names or prediction_model_cols(pred_df)

    for model in model_names:
        plt.figure(figsize=(8, 7))
        plt.scatter(
            pred_df["y_true"],
            pred_df[model],
            s=14,
            alpha=0.55,
            color="#1f77b4",
            label=f"{model} prediction",
        )
        min_v = float(min(pred_df["y_true"].min(), pred_df[model].min()))
        max_v = float(max(pred_df["y_true"].max(), pred_df[model].max()))
        plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
        plt.xlabel("Observed log10(BMR)")
        plt.ylabel("Predicted log10(BMR)")
        plt.title(f"Observed vs Predicted log10(BMR) ({model})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"observed_vs_predicted_scatter_{model}.pdf", bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 7))
    for model in model_names:
        residual = pred_df["y_true"] - pred_df[model]
        plt.scatter(pred_df[model], residual, s=14, alpha=0.45, label=model)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Predicted log10(BMR)")
    plt.ylabel("Residual (log Observed - log Predicted)")
    plt.title("Residual Plot (log10(BMR))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residual_plot.pdf", bbox_inches="tight")
    plt.close()


def save_performance_boxplot(
    out_dir: Path,
    y_true: np.ndarray,
    pred_df: pd.DataFrame,
    random_state: int,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_boot = 200
    rows: list[dict[str, float | str]] = []
    y_true = np.asarray(y_true, dtype=float)
    n = len(y_true)
    model_names = model_names or prediction_model_cols(pred_df)
    for model in model_names:
        y_pred = pd.to_numeric(pred_df[model], errors="coerce").to_numpy(dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            yp = y_pred[idx]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if int(mask.sum()) < 2:
                continue
            rmse_b = float(np.sqrt(mean_squared_error(yt[mask], yp[mask])))
            rows.append({"model": model, "bootstrap_id": b, "rmse": rmse_b})
    perf_df = pd.DataFrame(rows)
    perf_df.to_csv(out_dir / "performance_boxplot_data.csv", index=False, encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=perf_df, x="model", y="rmse")
    plt.xlabel("Model")
    plt.ylabel("Bootstrap RMSE (log10(BMR))")
    plt.title("Model Performance Boxplot (log10(BMR))")
    plt.tight_layout()
    plt.savefig(out_dir / "model_performance_boxplot.pdf", bbox_inches="tight")
    plt.close()
    return perf_df


def _shap_feature_group(feature_name: str) -> str:
    """Map raw design columns into fully pooled reporting groups (bar.pdf)."""
    name = str(feature_name)
    lower = name.lower()
    if lower.startswith("class_") or name == "class":
        return "taxonomy_class"
    if lower.startswith("pc") and lower[2:].isdigit():
        return "phylogeny"
    if name == "log_mass":
        return "mass"
    if name == "inv_kT":
        return "temperature"
    return name


def _is_pc_feature(feature_name: str) -> bool:
    lower = str(feature_name).lower()
    return lower.startswith("pc") and lower[2:].isdigit()


def _display_feature_label(feature_name: str) -> str:
    """Labels for beeswarm / raw-feature bar (PC pooled; mass/temperature renamed)."""
    name = str(feature_name)
    if name == "log_mass":
        return "mass"
    if name == "inv_kT":
        return "temperature"
    if _is_pc_feature(name):
        return "phylogeny"
    return name


def _mean_abs_shap_by_group(
    shap_values: np.ndarray,
    feature_names: list[str],
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    For each observation, sum |SHAP| within a feature group, then average over
    observations. Positive/negative direction is ignored by the abs step.
    """
    values = np.asarray(shap_values, dtype=float)
    if row_mask is not None:
        values = values[row_mask]
    if values.size == 0:
        return pd.DataFrame(columns=["feature_group", "mean_abs_shap", "n_features", "n_rows"])

    abs_vals = np.abs(values)
    groups = [_shap_feature_group(name) for name in feature_names]
    rows: list[dict[str, object]] = []
    for group_name in dict.fromkeys(groups):
        idxs = [i for i, g in enumerate(groups) if g == group_name]
        per_row = abs_vals[:, idxs].sum(axis=1)
        rows.append(
            {
                "feature_group": group_name,
                "mean_abs_shap": float(np.mean(per_row)),
                "n_features": int(len(idxs)),
                "n_rows": int(len(per_row)),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    return out.reset_index(drop=True)


def _build_pc_merged_shap_frame(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Merge PC1–PC5 into one phylogeny column for beeswarm / raw-feature displays.

    Beeswarm uses signed sum of PC SHAP values (direction preserved).
    Feature value for phylogeny coloring is the sum of PC feature values.
    Class dummies stay separate; log_mass→mass, inv_kT→temperature.
    """
    feature_names = list(X.columns)
    shap_values = np.asarray(shap_values, dtype=float)
    pc_idxs = [i for i, name in enumerate(feature_names) if _is_pc_feature(name)]
    keep_idxs = [i for i, name in enumerate(feature_names) if not _is_pc_feature(name)]

    shap_cols: list[np.ndarray] = []
    value_cols: list[np.ndarray] = []
    labels: list[str] = []

    for i in keep_idxs:
        name = feature_names[i]
        labels.append(_display_feature_label(name))
        shap_cols.append(shap_values[:, i])
        value_cols.append(X.iloc[:, i].to_numpy(dtype=float))

    if pc_idxs:
        labels.append("phylogeny")
        shap_cols.append(shap_values[:, pc_idxs].sum(axis=1))
        value_cols.append(X.iloc[:, pc_idxs].to_numpy(dtype=float).sum(axis=1))

    shap_merged = np.column_stack(shap_cols)
    X_merged = pd.DataFrame(
        np.column_stack(value_cols),
        columns=labels,
    )
    return shap_merged, X_merged, labels


def _raw_feature_mean_abs_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    mean(|SHAP|) with PC1–PC5 pooled as phylogeny, classes kept separate,
    log_mass→mass, inv_kT→temperature.
    """
    shap_values = np.asarray(shap_values, dtype=float)
    abs_vals = np.abs(shap_values)
    rows: list[dict[str, object]] = []
    pc_idxs = [i for i, name in enumerate(feature_names) if _is_pc_feature(name)]

    for i, name in enumerate(feature_names):
        if _is_pc_feature(name):
            continue
        rows.append(
            {
                "feature": _display_feature_label(name),
                "mean_abs_shap": float(abs_vals[:, i].mean()),
            }
        )
    if pc_idxs:
        # Same abs-then-sum logic as grouped phylogeny.
        per_row = abs_vals[:, pc_idxs].sum(axis=1)
        rows.append(
            {
                "feature": "phylogeny",
                "mean_abs_shap": float(np.mean(per_row)),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    return out.reset_index(drop=True)


def save_shap_outputs(
    out_dir: Path,
    models: dict[str, object],
    shap_inputs: dict[str, pd.DataFrame],
    class_labels: np.ndarray | pd.Series | None = None,
) -> None:
    """
    Write SHAP CSVs/plots for residual RF and XGB on the evaluation rows.

    Three plot types × two models (random_forest / xgboost):
    - shap_summary_bar_{model}.pdf: fully grouped mean(|SHAP|)
      (taxonomy_class / phylogeny / mass / temperature)
    - shap_summary_bar_raw_features_{model}.pdf: mean(|SHAP|) with 8 class_*
      kept separate + phylogeny (PC1–5) + mass + temperature
    - shap_summary_beeswarm_{model}.pdf: signed SHAP, PC1–5 summed into phylogeny
    """
    def save_current_figure(path: Path) -> None:
        fig = plt.gcf()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    model_names = [name for name in MODEL_NAMES if name in models and name in shap_inputs]
    if not model_names:
        print(f"  Skip SHAP for {out_dir.name}: no models with feature frames.")
        return

    all_feature_frames: list[pd.DataFrame] = []
    all_grouped_frames: list[pd.DataFrame] = []
    all_raw_frames: list[pd.DataFrame] = []
    all_by_class_frames: list[pd.DataFrame] = []

    for model_name in model_names:
        model = models[model_name]
        X_test_res = shap_inputs[model_name]
        if X_test_res.empty:
            print(f"  Skip SHAP for {out_dir.name}/{model_name}: empty feature frame.")
            continue

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_res)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values, dtype=float)
        feature_names = list(X_test_res.columns)

        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_df = pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_names,
                "feature_group": [_shap_feature_group(name) for name in feature_names],
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        all_feature_frames.append(feature_df)

        grouped_df = _mean_abs_shap_by_group(shap_values, feature_names).copy()
        grouped_df.insert(0, "model", model_name)
        all_grouped_frames.append(grouped_df)

        raw_imp = _raw_feature_mean_abs_importance(shap_values, feature_names).copy()
        raw_imp.insert(0, "model", model_name)
        all_raw_frames.append(raw_imp)

        if class_labels is not None:
            labels = pd.Series(class_labels).astype("string").reset_index(drop=True)
            if len(labels) != len(X_test_res):
                raise ValueError(
                    f"SHAP class_labels length {len(labels)} != feature rows {len(X_test_res)}."
                )
            for class_name, idx in labels.groupby(labels, sort=True).groups.items():
                if pd.isna(class_name):
                    continue
                row_mask = np.zeros(len(labels), dtype=bool)
                row_mask[np.asarray(idx, dtype=int)] = True
                class_grouped = _mean_abs_shap_by_group(
                    shap_values, feature_names, row_mask=row_mask
                )
                if class_grouped.empty:
                    continue
                class_grouped = class_grouped.copy()
                class_grouped.insert(0, "class", str(class_name))
                class_grouped.insert(0, "model", model_name)
                all_by_class_frames.append(class_grouped)

        # 1) Grouped bar
        plt.figure(figsize=(9, 6))
        plot_df = grouped_df.sort_values("mean_abs_shap", ascending=True)
        plt.barh(plot_df["feature_group"], plot_df["mean_abs_shap"], color="#4C72B0")
        plt.xlabel("mean(|SHAP|) over test observations")
        plt.title(f"Grouped SHAP importance ({model_name})")
        plt.tight_layout()
        save_current_figure(out_dir / f"shap_summary_bar_{model_name}.pdf")

        # 2) Raw bar: 8 classes + phylogeny + mass + temperature
        plt.figure(figsize=(9, max(6.0, 0.35 * len(raw_imp) + 2.0)))
        plot_raw = raw_imp.sort_values("mean_abs_shap", ascending=True)
        plt.barh(plot_raw["feature"], plot_raw["mean_abs_shap"], color="#4C72B0")
        plt.xlabel("mean(|SHAP|) over test observations")
        plt.title(f"SHAP feature importance ({model_name})")
        plt.tight_layout()
        save_current_figure(out_dir / f"shap_summary_bar_raw_features_{model_name}.pdf")

        # 3) Beeswarm: signed SHAP; PC1–PC5 summed into phylogeny
        shap_merged, X_merged, _ = _build_pc_merged_shap_frame(shap_values, X_test_res)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_merged, X_merged, show=False)
        plt.title(f"SHAP beeswarm ({model_name})")
        plt.tight_layout()
        save_current_figure(out_dir / f"shap_summary_beeswarm_{model_name}.pdf")

        print(f"  SHAP ({model_name}): wrote plots under {out_dir}", flush=True)

    if all_feature_frames:
        pd.concat(all_feature_frames, ignore_index=True).to_csv(
            out_dir / "shap_feature_importance.csv", index=False, encoding="utf-8"
        )
    if all_grouped_frames:
        pd.concat(all_grouped_frames, ignore_index=True).to_csv(
            out_dir / "shap_grouped_importance.csv", index=False, encoding="utf-8"
        )
    if all_raw_frames:
        pd.concat(all_raw_frames, ignore_index=True).to_csv(
            out_dir / "shap_raw_feature_importance.csv", index=False, encoding="utf-8"
        )
    if all_by_class_frames:
        pd.concat(all_by_class_frames, ignore_index=True).to_csv(
            out_dir / "shap_grouped_importance_by_class.csv",
            index=False,
            encoding="utf-8",
        )

    (out_dir / "shap_protocol.txt").write_text(
        "\n".join(
            [
                "SHAP protocol (residual RF and XGB)",
                "",
                "Per test row, each feature gets a signed SHAP contribution to",
                "predicted log10(BMR): positive pushes up, negative pushes down.",
                "",
                "Six plots (3 types × random_forest / xgboost):",
                "- shap_summary_bar_{model}.pdf: fully grouped mean(|SHAP|)",
                "  (taxonomy_class, phylogeny, mass, temperature)",
                "- shap_summary_bar_raw_features_{model}.pdf: mean(|SHAP|) with",
                "  8 class_* features + phylogeny (PC1–PC5) + mass + temperature",
                "- shap_summary_beeswarm_{model}.pdf: signed SHAP beeswarm;",
                "  PC1–PC5 SHAP values are summed into phylogeny (not abs first)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def log_bmr_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    """Multiplicative accuracy on log10(BMR): 10^(-|pred - true|)."""
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
    """One row per taxon_name; columns are per-model mean accuracy across rows."""
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


def write_group_species_accuracy(group_dir: Path, fold_tags: list[str]) -> Path:
    """
    Stitch fold/test predictions and write RF/XGB species-level accuracy.
    """
    frames: list[pd.DataFrame] = []
    for tag in fold_tags:
        pred_path = group_dir / tag / "benchmark_predictions_test.csv"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing prediction file for species accuracy: {pred_path}"
            )
        fold_df = pd.read_csv(pred_path)
        missing = [name for name in MODEL_NAMES if name not in fold_df.columns]
        if missing:
            raise KeyError(f"{pred_path} missing model columns: {missing}")
        fold_df["eval_split"] = tag
        frames.append(fold_df)

    stitched = pd.concat(frames, ignore_index=True)
    if stitched.empty:
        raise ValueError(f"No rows available to stitch under {group_dir}")

    accuracy_df = build_species_accuracy_table(stitched, MODEL_NAMES)
    group_dir.mkdir(parents=True, exist_ok=True)
    out_path = group_dir / "species_accuracy.csv"
    accuracy_df.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"[species accuracy] {group_dir.name}: species={len(accuracy_df)}, "
        f"splits={fold_tags} -> {out_path}"
    )
    return out_path


def write_group_eval_from_predictions(
    group_name: str,
    group_test_df: pd.DataFrame,
    pred_df_all: pd.DataFrame,
    models: dict[str, object],
    shap_inputs_all: dict[str, pd.DataFrame],
    out_dir: Path,
    model_dir: Path,
    random_state: int,
    write_models_copy: bool,
) -> pd.DataFrame:
    """Write RF and XGB metrics/plots for a class subset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = group_test_df.index.to_numpy()
    pred_df = pred_df_all.loc[mask].reset_index(drop=True)
    y_test = pred_df["y_true"].to_numpy(dtype=float)
    model_names = prediction_model_cols(pred_df)

    metrics_rows = []
    classes = (
        pred_df["class"].astype(str).to_numpy()
        if "class" in pred_df.columns
        else None
    )
    for model in model_names:
        metrics_rows.append(
            {
                "model": model,
                **evaluate_reporting_suite(
                    y_test, pred_df[model].to_numpy(), classes
                ),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    metrics_df.to_csv(out_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")
    pred_df.to_csv(out_dir / "benchmark_predictions_test.csv", index=False, encoding="utf-8")
    (out_dir / "evaluation_protocol.txt").write_text(
        "\n".join(
            [
                "Residual-learning evaluation protocol",
                "",
                "Training uses class-balanced sample weights:",
                f"  {CLASS_BALANCED_WEIGHT_FORMULA}",
                "Hyper-parameters: fixed 4-fold species-blocked CV on the 80%",
                "development set; final models retrained on all development data;",
                "held-out 20% test is a species-blocked holdout (not a CV fold).",
                "",
                "Metrics: rmse/mae/r2 = micro (pooled); *_macro = equal class average;",
                "*_bal = class-balanced weighted metrics on the evaluation set.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if write_models_copy:
        (out_dir / "model_bundle_path.txt").write_text(str(model_dir), encoding="utf-8")
        save_loss_curve_from_bundle(model_dir, out_dir)

    save_pred_and_residual_plots(out_dir=out_dir, pred_df=pred_df, model_names=model_names)
    save_performance_boxplot(
        out_dir=out_dir,
        y_true=y_test,
        pred_df=pred_df,
        random_state=random_state,
        model_names=model_names,
    )
    shap_inputs = {
        name: frame.loc[mask].reset_index(drop=True) for name, frame in shap_inputs_all.items()
    }
    save_shap_outputs(
        out_dir=out_dir,
        models=models,
        shap_inputs=shap_inputs,
        class_labels=pred_df["class"] if "class" in pred_df.columns else None,
    )

    n_obs = len(pred_df)
    n_species = (
        int(pred_df["taxon_name"].nunique()) if "taxon_name" in pred_df.columns else None
    )
    print(f"\n[{group_name}] Eval rows: {n_obs} (RF + XGB)")
    if n_species is None:
        print(f"[{group_name}] Test data for this group: {n_obs} observations")
    else:
        print(
            f"[{group_name}] Test data for this group: {n_obs} observations, "
            f"{n_species} species"
        )
    print(f"[{group_name}] Saved outputs in: {out_dir}")
    print(metrics_df.to_string(index=False))
    return metrics_df


def fit_fixed_rf(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> RandomForestRegressor:
    """Fit RF with a fixed hyperparameter set (no early stopping)."""
    rf = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(RF_FIXED_PARAMS["min_samples_leaf"]),
        max_features=RF_FIXED_PARAMS["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )
    fit_kwargs: dict = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    rf.fit(X_fit, y_fit, **fit_kwargs)
    return rf


def fit_fixed_xgb(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> XGBRegressor:
    """Fit XGB with a fixed hyperparameter set (no early stopping)."""
    xgb = make_xgb_regressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        random_state=random_state,
    )
    fit_kwargs: dict = {"verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    xgb.fit(X_fit, y_fit, **fit_kwargs)
    return xgb


def run_oof_cv_predictions(
    cv_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    best_params: dict,
    random_state: int,
    balance_classes: bool,
) -> pd.DataFrame:
    """
    Re-run four-fold CV with the selected hyperparameter set and collect
    out-of-fold log10(BMR) predictions for RF and XGB.
    """
    xgb_params = best_params["xgboost"]
    rf_params = best_params["random_forest"]
    prediction_columns = list(
        dict.fromkeys(["taxon_name", *TAXONOMY_METADATA_COLUMNS, *TREE_MODEL_FEATURES])
    )
    frames: list[pd.DataFrame] = []
    print(
        f"  Re-running four-fold CV with best HPs for OOF predictions "
        f"({len(cv_splits)} folds)...",
        flush=True,
    )
    for fold_idx, (fold_tag, fit_df, val_df) in enumerate(cv_splits):
        assert_no_species_leakage(fit_df, val_df)
        m3_coef, m3_clade_levels = fit_m3_baseline(fit_df)
        X_fit, residual_fit, _ = encode_train_frame(fit_df, m3_coef, m3_clade_levels)
        feature_columns = list(X_fit.columns)
        X_val, _, base_val = encode_train_frame(
            val_df, m3_coef, m3_clade_levels, feature_columns
        )
        sw_fit = (
            make_class_balanced_sample_weight(fit_df) if balance_classes else None
        )

        rf = fit_fixed_rf(
            X_fit,
            residual_fit,
            rf_params,
            sw_fit,
            random_state + fold_idx,
        )
        xgb = fit_fixed_xgb(
            X_fit,
            residual_fit,
            xgb_params,
            sw_fit,
            random_state + fold_idx,
        )

        fold_pred = val_df[prediction_columns].copy().reset_index(drop=True)
        fold_pred["fold"] = fold_tag
        fold_pred["y_true"] = val_df[LOG_TARGET].to_numpy(dtype=float)
        fold_pred["random_forest"] = base_val + np.asarray(
            rf.predict(X_val), dtype=float
        )
        fold_pred["xgboost"] = base_val + np.asarray(xgb.predict(X_val), dtype=float)
        frames.append(fold_pred)
        print(
            f"    {fold_tag}: OOF rows={len(fold_pred)} "
            f"(fit={len(fit_df)}, val={len(val_df)})",
            flush=True,
        )

    oof_df = pd.concat(frames, ignore_index=True)
    return oof_df


def save_cv_oof_outputs(out_dir: Path, oof_df: pd.DataFrame) -> list[str]:
    """
    Write OOF prediction/metric CSVs under <group>/cv/ for all and class subsets.
    """
    groups_done: list[str] = []
    oof_reset = oof_df.reset_index(drop=True)
    for group_name, class_name in GROUP_CLASS_FILTERS.items():
        if class_name is None:
            group_df = oof_reset
        else:
            group_df = oof_reset.loc[oof_reset["class"] == class_name].reset_index(
                drop=True
            )
        if group_df.empty:
            print(f"Skip {group_name}/cv: no OOF rows for class={class_name}.")
            continue

        group_out = out_dir / group_name / "cv"
        group_out.mkdir(parents=True, exist_ok=True)
        metrics_rows = []
        classes = (
            group_df["class"].astype(str).to_numpy()
            if "class" in group_df.columns
            else None
        )
        for model in MODEL_NAMES:
            metrics_rows.append(
                {
                    "model": model,
                    **evaluate_reporting_suite(
                        group_df["y_true"].to_numpy(dtype=float),
                        group_df[model].to_numpy(dtype=float),
                        classes,
                    ),
                }
            )
        metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
        group_df.to_csv(
            group_out / "benchmark_predictions_cv.csv", index=False, encoding="utf-8"
        )
        metrics_df.to_csv(
            group_out / "benchmark_metrics_cv.csv", index=False, encoding="utf-8"
        )
        (group_out / "evaluation_protocol.txt").write_text(
            "\n".join(
                [
                    "Residual-learning OOF evaluation protocol",
                    f"Weight formula: {CLASS_BALANCED_WEIGHT_FORMULA}",
                    "rmse/mae/r2=micro; *_macro=equal-class mean; *_bal=balanced weights.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"\n[{group_name}/cv] Saved OOF predictions: {len(group_df)} rows")
        print(metrics_df.to_string(index=False))
        groups_done.append(group_name)
    return groups_done


def evaluate_fold_predictions(
    fold_tag: str,
    test_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    shap_inputs: dict[str, pd.DataFrame],
    models: dict[str, object],
    out_dir: Path,
    model_dir: Path,
    random_state: int,
) -> list[str]:
    """Write all / class-group eval outputs for RF and XGB."""
    missing = [name for name in MODEL_NAMES if name not in preds]
    if missing:
        raise KeyError(f"Missing predictions for: {missing}")

    groups_done: list[str] = []
    prediction_columns = list(
        dict.fromkeys(["taxon_name", *TAXONOMY_METADATA_COLUMNS, *TREE_MODEL_FEATURES])
    )
    pred_df_all = test_df[prediction_columns].copy().reset_index(drop=True)
    pred_df_all["y_true"] = test_df[LOG_TARGET].to_numpy(dtype=float)
    for name in MODEL_NAMES:
        pred_df_all[name] = preds[name]
    shap_inputs = {name: shap_inputs[name].reset_index(drop=True) for name in MODEL_NAMES}
    models = {name: models[name] for name in MODEL_NAMES}
    test_df_reset = test_df.reset_index(drop=True)

    for group_name, class_name in GROUP_CLASS_FILTERS.items():
        if class_name is None:
            group_idx = test_df_reset.index
        else:
            group_idx = test_df_reset.index[test_df_reset["class"] == class_name]
        if len(group_idx) == 0:
            print(f"Skip {group_name}/{fold_tag}: no test rows for class={class_name}.")
            continue
        group_out = out_dir / group_name / fold_tag
        write_group_eval_from_predictions(
            group_name=f"{group_name}/{fold_tag}",
            group_test_df=test_df_reset.loc[group_idx],
            pred_df_all=pred_df_all,
            models=models,
            shap_inputs_all=shap_inputs,
            out_dir=group_out,
            model_dir=model_dir,
            random_state=random_state,
            write_models_copy=(group_name == "all"),
        )
        groups_done.append(group_name)
    return groups_done


def run_four_fold_cv_global(
    cv_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    random_state: int,
    balance_classes: bool,
    n_hp_trials: int,
) -> list[str]:
    """
    Select XGB parameters by four-fold CV, retrain both models on all four
    development folds, save/reload them, evaluate once on held-out test, and
    write OOF CV predictions under <group>/cv/ using the selected HPs.
    """
    fold_tag = "test"
    model_dir = out_dir / "all" / fold_tag / "models"

    print(
        f"  Four-fold CV on {len(full_train_df)} development rows; "
        f"held-out test rows={len(test_df)}...",
        flush=True,
    )
    bundle = tune_models_on_train(
        cv_splits=cv_splits,
        full_train_df=full_train_df,
        random_state=random_state,
        balance_classes=balance_classes,
        n_hp_trials=n_hp_trials,
    )
    bundle["fold_tag"] = fold_tag
    save_model_bundle(bundle, model_dir)
    write_hp_search_trials_csv(
        benchmark_dir=out_dir,
        fold_tag="cv4",
        bundle=bundle,
        model_dir=model_dir,
    )

    oof_df = run_oof_cv_predictions(
        cv_splits=cv_splits,
        best_params=bundle["best_params"],
        random_state=random_state,
        balance_classes=balance_classes,
    )
    save_cv_oof_outputs(out_dir=out_dir, oof_df=oof_df)

    # Remove obsolete in-sample development-train outputs if present.
    for obsolete in (
        out_dir / "all" / fold_tag / "benchmark_predictions_train.csv",
        out_dir / "all" / fold_tag / "benchmark_metrics_train.csv",
    ):
        if obsolete.exists():
            obsolete.unlink()

    print("  Reloading RF and XGB from disk for held-out test evaluation...", flush=True)
    loaded = load_model_bundle(model_dir)
    preds, shap_inputs = predict_log_bmr(loaded, test_df)
    return evaluate_fold_predictions(
        fold_tag=fold_tag,
        test_df=test_df,
        preds=preds,
        shap_inputs=shap_inputs,
        models=loaded["models"],
        out_dir=out_dir,
        model_dir=model_dir,
        random_state=random_state,
    )


def main() -> None:
    print("Running ml_residual_learning.py")
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Tune RF/XGB with matched hyperparameter grids and early stopping "
            "under fixed four-fold species-block CV, retrain on the complete 80% "
            "development set, write OOF CV predictions under <group>/cv/, then "
            "save/reload and evaluate once on the held-out 20% test set."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing fixed fold1..fold4/test train.csv and test.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark"),
        help="Output directory. Final results are stored under <group>/test.",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    split_dir = args.split_dir if args.split_dir.is_absolute() else root / args.split_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_summary_path = split_dir / "class_species_block_split_summary.csv"
    if split_summary_path.exists():
        pd.read_csv(split_summary_path).to_csv(
            out_dir / "class_species_block_split_summary.csv",
            index=False,
            encoding="utf-8",
        )

    required_names = ["fold1", "fold2", "fold3", "fold4", "test"]
    discovered = {
        name: (train_path, eval_path)
        for name, train_path, eval_path in discover_fold_splits(
            split_dir, folds=required_names
        )
    }
    missing = [name for name in required_names if name not in discovered]
    if missing:
        raise FileNotFoundError(
            f"Missing fixed splits: {missing}. Run python code/split_train_test_bmr.py first."
        )

    full_train_path, heldout_path = discovered["test"]
    full_train_df = load_split_data(full_train_path)
    heldout_df = load_split_data(heldout_path)
    assert_no_species_leakage(full_train_df, heldout_df)

    cv_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    validation_species: list[set[str]] = []
    for fold_name in required_names[:4]:
        train_path, val_path = discovered[fold_name]
        train_df = load_split_data(train_path)
        val_df = load_split_data(val_path)
        assert_no_species_leakage(train_df, val_df)
        assert_no_species_leakage(train_df, heldout_df)
        fold_tag = FOLD_DIR_NAMES[fold_name]
        cv_splits.append((fold_tag, train_df, val_df))
        validation_species.append(set(val_df["taxon_name"].astype(str)))
        print(
            f"  Loaded {fold_tag}: train={len(train_df)} rows, "
            f"validation={len(val_df)} rows",
            flush=True,
        )

    for i, species_i in enumerate(validation_species):
        for species_j in validation_species[i + 1 :]:
            if species_i.intersection(species_j):
                raise RuntimeError("Species leakage detected between CV validation folds.")
    cv_species_union = set().union(*validation_species)
    full_train_species = set(full_train_df["taxon_name"].astype(str))
    if cv_species_union != full_train_species:
        raise RuntimeError(
            "The union of fold1..fold4 validation species does not equal test/train species."
        )

    groups_done = run_four_fold_cv_global(
        cv_splits=cv_splits,
        full_train_df=full_train_df,
        test_df=heldout_df,
        out_dir=out_dir,
        random_state=args.random_state,
        balance_classes=True,
        n_hp_trials=N_HP_TRIALS,
    )
    for group_name in groups_done:
        write_group_species_accuracy(
            group_dir=out_dir / group_name,
            fold_tags=["test"],
        )

    print(f"\nRead fixed splits from: {split_dir}")
    print(f"Wrote OOF CV results under: {out_dir}/<group>/cv/")
    print(f"Wrote held-out test results under: {out_dir}/<group>/test/")
    print(f"Final 80%-trained models under: {out_dir}/all/test/models/")
    print(f"Random XGB search log: {out_dir}/hp_search_trials.csv")
    print(f"Best XGB params: {out_dir}/xgb_best_params.csv")
    print(f"Wrote species accuracy under: {out_dir}/<group>/species_accuracy.csv")


if __name__ == "__main__":
    main()
