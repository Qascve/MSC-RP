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
LOG_TARGET = "log_BMR"
MODEL_NAMES = ["random_forest", "xgboost"]
POWER_LAW_FEATURES = ["log_mass"]
TAXONOMY_MODEL_FEATURES = ["class", "order", "family"]
TAXONOMY_METADATA_COLUMNS = ["class", "order", "family", "Genus", "species"]
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
    "Insecta": "Insecta",
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

EARLY_STOPPING_ROUNDS = 30
XGB_MAX_ESTIMATORS = 1000
N_HP_TRIALS = 50
RF_FIXED_PARAMS = {
    "n_estimators": 600,
    "max_depth": 4,
    "min_samples_leaf": 5,
    "max_features": 1.0,
}
XGB_PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.08],
    "subsample": [0.6, 0.7],
    "colsample_bytree": [0.6, 0.7],
    "reg_lambda": [0.9, 2.0, 5.0],
    "min_child_weight": [3, 5],
}


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep_cols = list(
        dict.fromkeys(
            [
                "taxon_name",
                *TAXONOMY_METADATA_COLUMNS,
                *TREE_MODEL_FEATURES,
                TARGET,
                LOG_TARGET,
            ]
        )
    )
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[keep_cols].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    for col in TAXONOMY_METADATA_COLUMNS:
        out[col] = out[col].astype("string").str.strip()
    numeric_features = ["log_mass", "inv_kT", "pc1", "pc2", "pc3", "pc4", "pc5"]
    for col in numeric_features + [TARGET, LOG_TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["taxon_name"] = out["taxon_name"].replace("", pd.NA)
    for col in TAXONOMY_METADATA_COLUMNS:
        out[col] = out[col].replace("", pd.NA)
    out = out.dropna(subset=keep_cols).copy()
    out = out[(out["log_mass"].notna()) & (out["inv_kT"].notna()) & (out[TARGET] > 0)].copy()
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


def fit_alpha_three_quarter(log_mass: np.ndarray, log_bmr: np.ndarray) -> float:
    return float(np.mean(log_bmr - 0.75 * log_mass))


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
    if len({_params_key(params) for params in all_combinations}) != total:
        raise ValueError("XGB_PARAM_GRID contains duplicate parameter combinations.")
    if not 1 <= n_trials <= total:
        raise ValueError(
            f"n_hp_trials must be between 1 and {total}; got {n_trials}."
        )
    rng = np.random.default_rng(random_state)
    selected = rng.choice(total, size=n_trials, replace=False)
    return [all_combinations[int(i)] for i in selected]


def encode_train_frame(
    df: pd.DataFrame, alpha: float, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Encode residual features for one frame; optionally align to saved columns."""
    cat = [c for c in TREE_MODEL_FEATURES if c in TAXONOMY_MODEL_FEATURES]
    raw = df[TREE_MODEL_FEATURES].reset_index(drop=True).copy()
    encoded = pd.get_dummies(raw, columns=cat, prefix=cat, dtype=float)
    base = alpha + 0.75 * df[POWER_LAW_FEATURES[0]].to_numpy(dtype=float)
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)
    residual = df[LOG_TARGET].to_numpy(dtype=float) - base
    return encoded, residual, base


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
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGB_MAX_ESTIMATORS,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse",
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
    Tune XGB by fixed four-fold species-block CV, then retrain RF and XGB on
    the complete 80% development set.

    Each CV baseline alpha is fitted only on that fold's three training
    partitions. Validation targets therefore never contribute to alpha.
    """
    if len(cv_splits) != 4:
        raise ValueError(f"Expected exactly four CV splits, got {len(cv_splits)}.")

    prepared_splits: list[dict] = []
    for fold_tag, fit_df, val_df in cv_splits:
        assert_no_species_leakage(fit_df, val_df)
        alpha_inner = fit_alpha_three_quarter(
            fit_df[POWER_LAW_FEATURES[0]].to_numpy(dtype=float),
            fit_df[LOG_TARGET].to_numpy(dtype=float),
        )
        X_fit, residual_fit, _ = encode_train_frame(fit_df, alpha_inner)
        feature_columns_inner = list(X_fit.columns)
        X_val, residual_val, _ = encode_train_frame(
            val_df, alpha_inner, feature_columns_inner
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
                "alpha_inner": float(alpha_inner),
            }
        )

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

    xgb_trials: list[dict] = []
    best_xgb: dict | None = None
    best_xgb_histories: list[list[float]] = []

    print(
        f"  XGB random four-fold CV: {n_combinations} unique combinations "
        f"drawn from {total_combinations} x {len(prepared_splits)} folds",
        flush=True,
    )
    for trial, xgb_params in enumerate(xgb_param_sets):
        fold_scores: list[float] = []
        fold_estimators: list[int] = []
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

    assert best_xgb is not None

    # XGB uses the winning trial; RF uses one fixed configuration without tuning.
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

    # Re-estimate alpha only after selection, now using all four development folds.
    alpha_full = fit_alpha_three_quarter(
        full_train_df[POWER_LAW_FEATURES[0]].to_numpy(dtype=float),
        full_train_df[LOG_TARGET].to_numpy(dtype=float),
    )
    X_all, residual_all, _ = encode_train_frame(full_train_df, alpha_full)
    feature_columns = list(X_all.columns)
    sw_all = (
        make_class_balanced_sample_weight(full_train_df)
        if balance_classes
        else None
    )

    rf_full = RandomForestRegressor(
        **RF_FIXED_PARAMS,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_full.fit(X_all, residual_all, sample_weight=sw_all)

    xgb_full = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=xgb_n_final,
        learning_rate=best_xgb["learning_rate"],
        max_depth=best_xgb["max_depth"],
        subsample=best_xgb["subsample"],
        colsample_bytree=best_xgb["colsample_bytree"],
        reg_lambda=best_xgb["reg_lambda"],
        min_child_weight=best_xgb["min_child_weight"],
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse",
    )
    xgb_full.fit(X_all, residual_all, sample_weight=sw_all, verbose=False)

    print(
        "  RF fixed params (no HP search): "
        f"{RF_FIXED_PARAMS}",
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

    xgb_score = float(best_xgb["cv_mean_rmse"])

    return {
        "models": {"random_forest": rf_full, "xgboost": xgb_full},
        "alpha": float(alpha_full),
        "feature_columns": feature_columns,
        "best_params": {
            "random_forest": {
                **RF_FIXED_PARAMS,
                "tuned": False,
            },
            "xgboost": {
                "trial": int(best_xgb["trial"]),
                "max_depth": int(best_xgb["max_depth"]),
                "learning_rate": float(best_xgb["learning_rate"]),
                "subsample": float(best_xgb["subsample"]),
                "colsample_bytree": float(best_xgb["colsample_bytree"]),
                "reg_lambda": float(best_xgb["reg_lambda"]),
                "min_child_weight": int(best_xgb["min_child_weight"]),
                "n_estimators": xgb_n_final,
                "cv_mean_rmse": xgb_score,
                "cv_std_rmse": float(best_xgb["cv_std_rmse"]),
            },
        },
        "search_trials": {
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
                "alpha_inner": split["alpha_inner"],
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
        "alpha": bundle["alpha"],
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
    # Only XGB has HP-search records and an early-stopping curve.
    old_rf_log = model_dir / "hp_search_random_forest.csv"
    if old_rf_log.exists():
        old_rf_log.unlink()
    bundle["search_trials"]["xgboost"].to_csv(
        model_dir / "hp_search_xgboost.csv", index=False, encoding="utf-8"
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
    Write/append all sampled unique XGB combinations + four-fold CV RMSE.
    RF is omitted because it uses fixed parameters and is not tuned.

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
        f"  Wrote XGB random search ({len(new_df)} rows) -> {actual_out_path}",
        flush=True,
    )
    print(
        f"  Wrote best XGB parameters for {fold_tag} -> {best_path}",
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
    return {
        "models": models,
        "alpha": float(meta["alpha"]),
        "feature_columns": list(meta["feature_columns"]),
        "best_params": meta.get("best_params", {}),
        "model_features": list(meta.get("model_features", TREE_MODEL_FEATURES)),
    }


def predict_log_bmr(
    bundle: dict, df: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Predict log_BMR with a loaded bundle; also return residual feature frames for SHAP."""
    alpha = bundle["alpha"]
    feature_columns = bundle["feature_columns"]
    model_features = bundle.get("model_features", TREE_MODEL_FEATURES)
    cat = [c for c in model_features if c in TAXONOMY_MODEL_FEATURES]
    raw = df[model_features].reset_index(drop=True).copy()
    encoded = pd.get_dummies(raw, columns=cat, prefix=cat, dtype=float)
    base = alpha + 0.75 * df[POWER_LAW_FEATURES[0]].to_numpy(dtype=float)
    X = encoded.reindex(columns=feature_columns, fill_value=0.0)

    preds: dict[str, np.ndarray] = {}
    shap_inputs: dict[str, pd.DataFrame] = {}
    for name, model in bundle["models"].items():
        residual_hat = np.asarray(model.predict(X), dtype=float)
        preds[name] = base + residual_hat
        shap_inputs[name] = X
    return preds, shap_inputs


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Evaluate on log_BMR only."""
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
    plt.ylabel("RMSE (log_BMR residual)")
    plt.title("XGBoost Early-Stopping Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
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
        plt.xlabel("Observed log_BMR")
        plt.ylabel("Predicted log_BMR")
        plt.title(f"Observed vs Predicted log_BMR ({model})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"observed_vs_predicted_scatter_{model}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 7))
    for model in model_names:
        residual = pred_df["y_true"] - pred_df[model]
        plt.scatter(pred_df[model], residual, s=14, alpha=0.45, label=model)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Predicted log_BMR")
    plt.ylabel("Residual (log Observed - log Predicted)")
    plt.title("Residual Plot (log_BMR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residual_plot.png", dpi=160)
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
    plt.ylabel("Bootstrap RMSE (log_BMR)")
    plt.title("Model Performance Boxplot (log_BMR)")
    plt.tight_layout()
    plt.savefig(out_dir / "model_performance_boxplot.png", dpi=160)
    plt.close()
    return perf_df


def save_shap_outputs(
    out_dir: Path,
    metrics_df: pd.DataFrame,
    models: dict[str, object],
    shap_inputs: dict[str, pd.DataFrame],
) -> None:
    def save_current_figure(path: Path) -> None:
        fig = plt.gcf()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    shap_candidates = list(models.keys()) or ["random_forest", "xgboost"]
    subset = metrics_df[metrics_df["model"].isin(shap_candidates)]
    if subset.empty:
        subset = metrics_df
    best = subset.sort_values("rmse").iloc[0]["model"]
    if best not in models:
        best = next(iter(models))
    model = models[best]
    X_test_res = shap_inputs[best]
    if X_test_res.empty:
        print(f"  Skip SHAP for {out_dir.name}: empty feature frame.")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_res)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": X_test_res.columns, "mean_abs_shap": mean_abs})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(out_dir / "shap_feature_importance.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_res, show=False)
    plt.tight_layout()
    save_current_figure(out_dir / "shap_summary_beeswarm.png")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_res, plot_type="bar", show=False)
    plt.tight_layout()
    save_current_figure(out_dir / "shap_summary_bar.png")
    plt.close()


def log_bmr_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    """Symmetric accuracy on log_BMR: exp(-|pred - true|)."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    out = np.full(len(y_true_log), np.nan, dtype=float)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    out[mask] = np.exp(-np.abs(y_pred_log[mask] - y_true_log[mask]))
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
    for model in model_names:
        metrics_rows.append({"model": model, **evaluate(y_test, pred_df[model].to_numpy())})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    metrics_df.to_csv(out_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")
    pred_df.to_csv(out_dir / "benchmark_predictions_test.csv", index=False, encoding="utf-8")

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
        metrics_df=metrics_df,
        models=models,
        shap_inputs=shap_inputs,
    )

    print(f"\n[{group_name}] Eval rows: {len(pred_df)} (RF + XGB)")
    print(f"[{group_name}] Saved outputs in: {out_dir}")
    print(metrics_df.to_string(index=False))
    return metrics_df


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
    development folds, save/reload them, and evaluate once on held-out test.
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

    print("  Reloading RF and XGB from disk for evaluation...", flush=True)
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
            "Tune XGB with unique random grid combinations and fixed four-fold "
            "species-block CV, retrain RF "
            "and XGB on the complete 80% development set, then save/reload and "
            "evaluate once on the held-out 20% test set."
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
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
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
    print(f"Wrote held-out test results under: {out_dir}/<group>/test/")
    print(f"Final 80%-trained models under: {out_dir}/all/test/models/")
    print(f"Random XGB search log: {out_dir}/hp_search_trials.csv")
    print(f"Best XGB params: {out_dir}/xgb_best_params.csv")
    print(f"Wrote species accuracy under: {out_dir}/<group>/species_accuracy.csv")


if __name__ == "__main__":
    main()
