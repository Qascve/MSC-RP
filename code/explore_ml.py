#!/usr/bin/env python3
"""
explore_ml.py — RF/XGB under M0–M4 feature settings, aligned with ml_residual_learning.py:

- fold1 / fold2: species-block HP search + early stopping inside train,
  save models, reload, evaluate on that fold's held-out set
- test: no HP search; for each model slot load the better of f_1/f_2
- write per-split metrics/predictions under results/explore/{f_1,f_2,test}/
"""
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
from xgboost import XGBRegressor

TARGET = "BMR"
MASS_COL = "wet_Mass_kg"
TEMP_COL = "temperature"
CLADE_COL = "class"
K_BOLTZMANN_EV_PER_K = 8.617e-5
LOG_TARGET = "log_BMR"
FOLD_DIR_NAMES = {"fold1": "f_1", "fold2": "f_2", "test": "test"}
SPEC_NAMES = ("m0", "m1", "m2", "m3", "m4")
ALGO_NAMES = ("random_forest", "xgboost")

MODEL_FEATURES: dict[str, list[str]] = {
    "m0": ["log_mass"],
    "m1": ["log_mass"],
    "m2": ["log_mass", "inv_kT"],
    "m3": ["log_mass", "inv_kT", CLADE_COL],
    "m4": ["log_mass", "inv_kT", "pc1", "pc2", "pc3", "pc4", "pc5"],
}

N_HP_TRIALS = 100
EARLY_STOPPING_ROUNDS = 80
XGB_MAX_ESTIMATORS = 4000
RF_TREE_BATCH = 50
RF_MAX_ESTIMATORS = 1200
INNER_VAL_FRAC = 0.2
# Tune HPs once on the richest feature set, reuse across M0–M4.
HP_TUNE_SPEC = "m4"
RF_PARAM_GRID = {
    "max_depth": [3, 4, 6, 8, 10, 12, 16],
    "min_samples_leaf": [1, 2, 3, 5, 8, 10],
    "max_features": ["sqrt", 0.3, 0.5, 0.7, 0.9, 1.0],
}
XGB_PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "min_child_weight": [1, 3, 5, 10],
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


def _cartesian_param_dicts(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos: list[dict] = [{}]
    for key in keys:
        combos = [{**base, key: val} for base in combos for val in grid[key]]
    return combos


def draw_unique_param_sets(
    grid: dict, n_trials: int, rng: np.random.Generator, model_name: str
) -> list[dict]:
    pool = _cartesian_param_dicts(grid)
    if n_trials > len(pool):
        raise ValueError(
            f"{model_name}: requested {n_trials} unique trials but grid only has {len(pool)}."
        )
    order = rng.permutation(len(pool))
    return [dict(pool[int(i)]) for i in order[:n_trials]]


def fit_rf_with_early_stopping(
    X_fit, y_fit, X_val, y_val, params, random_state: int
) -> tuple[RandomForestRegressor, int, float]:
    best_rmse = np.inf
    best_n = RF_TREE_BATCH
    patience_left = max(1, EARLY_STOPPING_ROUNDS // RF_TREE_BATCH)
    rf = RandomForestRegressor(
        n_estimators=RF_TREE_BATCH,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )
    n_trees = 0
    while n_trees < RF_MAX_ESTIMATORS:
        n_trees += RF_TREE_BATCH
        rf.set_params(n_estimators=n_trees)
        rf.fit(X_fit, y_fit)
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
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )
    rf_best.fit(X_fit, y_fit)
    return rf_best, best_n, float(best_rmse)


def fit_xgb_with_early_stopping(
    X_fit, y_fit, X_val, y_val, params, random_state: int
) -> tuple[XGBRegressor, int, float]:
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
    xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n = int(getattr(xgb, "best_iteration", XGB_MAX_ESTIMATORS - 1)) + 1
    best_score = float(getattr(xgb, "best_score", rmse(y_val, xgb.predict(X_val))))
    return xgb, best_n, best_score


def model_key(algo: str, spec: str) -> str:
    return f"{algo}_{spec}"


def tune_shared_hyperparams(
    train_df: pd.DataFrame, random_state: int, n_trials: int
) -> dict:
    """Species-block HP search on HP_TUNE_SPEC features; shared by all M0–M4."""
    fit_df, val_df = species_block_train_val_split(
        train_df, val_frac=INNER_VAL_FRAC, random_state=random_state
    )
    feature_cols = MODEL_FEATURES[HP_TUNE_SPEC]
    X_fit, X_val = encode_features(fit_df, val_df, feature_cols)
    y_fit = fit_df[LOG_TARGET].to_numpy(dtype=float)
    y_val = val_df[LOG_TARGET].to_numpy(dtype=float)

    rng = np.random.default_rng(random_state)
    rf_sets = draw_unique_param_sets(RF_PARAM_GRID, n_trials, rng, "random_forest")
    xgb_sets = draw_unique_param_sets(XGB_PARAM_GRID, n_trials, rng, "xgboost")

    best_rf: dict | None = None
    best_xgb: dict | None = None
    rf_trials: list[dict] = []
    xgb_trials: list[dict] = []

    print(
        f"  HP search on {HP_TUNE_SPEC}: {n_trials} unique trials "
        f"(inner val species={val_df['taxon_name'].nunique()}, rows={len(val_df)})",
        flush=True,
    )
    for trial, (rf_p, xgb_p) in enumerate(zip(rf_sets, xgb_sets)):
        _, rf_n, rf_score = fit_rf_with_early_stopping(
            X_fit, y_fit, X_val, y_val, rf_p, random_state + trial
        )
        rf_row = {**rf_p, "n_estimators": rf_n, "val_rmse": rf_score, "trial": trial}
        rf_trials.append(rf_row)
        if best_rf is None or rf_score < best_rf["val_rmse"]:
            best_rf = dict(rf_row)

        _, xgb_n, xgb_score = fit_xgb_with_early_stopping(
            X_fit, y_fit, X_val, y_val, xgb_p, random_state + trial
        )
        xgb_row = {**xgb_p, "n_estimators": xgb_n, "val_rmse": xgb_score, "trial": trial}
        xgb_trials.append(xgb_row)
        if best_xgb is None or xgb_score < best_xgb["val_rmse"]:
            best_xgb = dict(xgb_row)

        if (trial + 1) % 10 == 0 or trial == 0:
            print(
                f"    trial {trial + 1}/{n_trials}: RF={rf_score:.4f}, XGB={xgb_score:.4f}",
                flush=True,
            )

    assert best_rf is not None and best_xgb is not None
    print(
        f"  Best RF  trial={best_rf['trial']} val_rmse={best_rf['val_rmse']:.4f} "
        f"n={best_rf['n_estimators']}",
        flush=True,
    )
    print(
        f"  Best XGB trial={best_xgb['trial']} val_rmse={best_xgb['val_rmse']:.4f} "
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
        "inner_val_species": int(val_df["taxon_name"].nunique()),
        "inner_val_rows": int(len(val_df)),
    }


def train_all_specs(
    train_df: pd.DataFrame, best_params: dict, random_state: int
) -> dict:
    """Fit RF/XGB for every M0–M4 with shared best HPs on full train."""
    models: dict[str, object] = {}
    feature_columns: dict[str, list[str]] = {}
    y_train = train_df[LOG_TARGET].to_numpy(dtype=float)

    rf_bp = best_params["random_forest"]
    xgb_bp = best_params["xgboost"]

    for spec, feature_cols in MODEL_FEATURES.items():
        X = encode_one(train_df, feature_cols)
        feature_columns[spec] = list(X.columns)

        rf = RandomForestRegressor(
            n_estimators=int(rf_bp["n_estimators"]),
            max_depth=rf_bp["max_depth"],
            min_samples_leaf=rf_bp["min_samples_leaf"],
            max_features=rf_bp["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y_train)
        models[model_key("random_forest", spec)] = rf

        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(xgb_bp["n_estimators"]),
            learning_rate=xgb_bp["learning_rate"],
            max_depth=xgb_bp["max_depth"],
            subsample=xgb_bp["subsample"],
            colsample_bytree=xgb_bp["colsample_bytree"],
            reg_lambda=xgb_bp["reg_lambda"],
            min_child_weight=xgb_bp["min_child_weight"],
            random_state=random_state,
            n_jobs=-1,
            eval_metric="rmse",
        )
        xgb.fit(X, y_train, verbose=False)
        models[model_key("xgboost", spec)] = xgb

    return {"models": models, "feature_columns": feature_columns}


def save_model_bundle(bundle: dict, tune: dict, model_dir: Path, fold_tag: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in bundle["models"].items():
        joblib.dump(model, model_dir / f"{name}.joblib")
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
        "model_features": MODEL_FEATURES,
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
        # name = random_forest_m0 / xgboost_m3
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
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    fold_tag: str,
) -> pd.DataFrame:
    fold_out.mkdir(parents=True, exist_ok=True)
    metric_rows = [
        {"model": name, **evaluate(y_true, pred)} for name, pred in predictions.items()
    ]
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True)
    metrics_df.to_csv(fold_out / "explore_ml_metrics.csv", index=False, encoding="utf-8")

    pred_df = test_df[["taxon_name", CLADE_COL, "log_mass", "inv_kT"]].copy()
    pred_df["y_true"] = y_true
    pred_df["fold"] = fold_tag
    for name, pred in predictions.items():
        pred_df[name] = pred
    pred_df.to_csv(fold_out / "explore_ml_predictions_test.csv", index=False, encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig_width = max(12.0, 0.8 * len(metrics_df) + 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))
    sns.barplot(data=metrics_df, x="model", y="rmse", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE (log_BMR)")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=metrics_df, x="model", y="r2", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2 (log_BMR)")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(f"ML Model Performance ({fold_tag})", fontsize=14)
    fig.tight_layout()
    fig.savefig(fold_out / "explore_ml_model_performance_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(9, 7))
    for name, pred in predictions.items():
        plt.scatter(pred, y_true - pred, s=10, alpha=0.35, label=name)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Predicted log_BMR")
    plt.ylabel("Residual")
    plt.title(f"ML Residual Plot ({fold_tag})")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fold_out / "explore_ml_residual_plot.png", dpi=180)
    plt.close()

    print(f"[{fold_tag}] Saved metrics/predictions under {fold_out}")
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
) -> None:
    model_dir = out_dir / fold_tag / "models"
    print(f"  Tuning + training M0–M4 on {len(train_df)} train rows...", flush=True)
    tune = tune_shared_hyperparams(train_df, random_state, n_hp_trials)
    bundle = train_all_specs(train_df, tune, random_state)
    save_model_bundle(bundle, tune, model_dir, fold_tag)
    write_hp_search_trials_csv(out_dir, fold_tag, tune, reset=reset_hp_log)

    print("  Reloading models from disk for evaluation...", flush=True)
    loaded = load_model_bundle(model_dir)
    preds = predict_with_bundle(loaded, test_df)
    y_true = test_df[LOG_TARGET].to_numpy(dtype=float)
    save_outputs(out_dir / fold_tag, test_df, y_true, preds, fold_tag)


def select_best_models_from_cv(out_dir: Path, source_folds: list[str] | None = None) -> dict[str, dict]:
    source_folds = source_folds or ["f_1", "f_2"]
    # Collect all model names from first available fold
    model_names: list[str] = []
    for tag in source_folds:
        meta_path = out_dir / tag / "models" / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_names = list(meta["model_names"])
            break
    if not model_names:
        raise FileNotFoundError(
            f"No explore_ml models under {out_dir}/f_1|f_2/models. Run fold1/fold2 first."
        )

    selected: dict[str, dict] = {}
    for name in model_names:
        best: dict | None = None
        for tag in source_folds:
            metrics_path = out_dir / tag / "explore_ml_metrics.csv"
            model_dir = out_dir / tag / "models"
            if not metrics_path.exists() or not (model_dir / "meta.json").exists():
                raise FileNotFoundError(f"Missing explore_ml artifacts for {tag}")
            metrics = pd.read_csv(metrics_path)
            hit = metrics.loc[metrics["model"] == name, "rmse"]
            if hit.empty:
                continue
            score = float(hit.iloc[0])
            if best is None or score < best["score"]:
                best = {"fold": tag, "model_name": name, "score": score, "model_dir": model_dir}
        if best is None:
            raise RuntimeError(f"No fold metrics found for model {name}")
        selected[name] = best
        print(
            f"  test uses {name} from {best['fold']} (fold RMSE={best['score']:.4f})",
            flush=True,
        )
    return selected


def run_test_from_cv(test_df: pd.DataFrame, out_dir: Path) -> None:
    fold_tag = "test"
    print("  Selecting best saved models from f_1/f_2 (no HP search on test)...", flush=True)
    selection = select_best_models_from_cv(out_dir)
    model_dir = out_dir / fold_tag / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "selection.json").write_text(
        json.dumps(
            {
                name: {
                    "source_fold": info["fold"],
                    "selection_rmse": info["score"],
                    "source_model_dir": str(info["model_dir"]),
                }
                for name, info in selection.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Load each model from its winning fold (may differ by model).
    preds: dict[str, np.ndarray] = {}
    for name, info in selection.items():
        bundle = load_model_bundle(info["model_dir"])
        single = {
            "models": {name: bundle["models"][name]},
            "feature_columns": bundle["feature_columns"],
            "model_features": bundle.get("model_features", MODEL_FEATURES),
        }
        preds[name] = predict_with_bundle(single, test_df)[name]

    y_true = test_df[LOG_TARGET].to_numpy(dtype=float)
    save_outputs(out_dir / fold_tag, test_df, y_true, preds, fold_tag)


def log_bmr_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> np.ndarray:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    out = np.full(len(y_true_log), np.nan, dtype=float)
    mask = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    out[mask] = np.exp(-np.abs(y_pred_log[mask] - y_true_log[mask]))
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
            f"No fold splits found under {split_dir}. Expected fold1/, fold2/, and test/."
        )
    return found


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "RF/XGB under M0–M4 with HP search + early stopping on fold1/fold2; "
            "test uses best saved f_1/f_2 models (no HP search)."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--folds", nargs="+", default=["fold1", "fold2", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/explore"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-hp-trials", type=int, default=N_HP_TRIALS)
    args = parser.parse_args()

    split_dir = _resolve_path(root, args.split_dir)
    out_dir = _resolve_path(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_splits = discover_fold_splits(split_dir, list(args.folds))
    fold_tags: list[str] = []
    hp_reset_done = False

    for fold_name, train_path, test_path in fold_splits:
        fold_tag = FOLD_DIR_NAMES.get(fold_name, fold_name)
        fold_tags.append(fold_tag)
        print(f"\n=== explore_ml {fold_tag} ({fold_name}) ===", flush=True)
        test_df = add_mte_features(load_split_data(test_path))

        if fold_tag == "test" or fold_name == "test":
            if train_path.exists():
                train_df = add_mte_features(load_split_data(train_path))
                assert_no_species_leakage(train_df, test_df)
            run_test_from_cv(test_df, out_dir)
        else:
            train_df = add_mte_features(load_split_data(train_path))
            assert_no_species_leakage(train_df, test_df)
            run_cv_fold(
                train_df=train_df,
                test_df=test_df,
                fold_tag=fold_tag,
                out_dir=out_dir,
                random_state=args.random_state,
                n_hp_trials=args.n_hp_trials,
                reset_hp_log=not hp_reset_done,
            )
            hp_reset_done = True

    if fold_tags:
        write_explore_ml_species_accuracy(out_dir, fold_tags)
    print(f"\nWrote explore_ml results under: {out_dir}/<f_1|f_2|test>/")
    print(f"HP search log: {out_dir}/explore_ml_hp_search_trials.csv")


if __name__ == "__main__":
    main()
