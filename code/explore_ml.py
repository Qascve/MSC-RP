#!/usr/bin/env python3
"""Train RF/XGB M1–M4 with class-balanced weights; report micro/macro/bal metrics."""
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

# Matched to ml_residual_learning.py
N_HP_TRIALS = 50
EARLY_STOPPING_ROUNDS = 30
XGB_MAX_ESTIMATORS = 1000
RF_TREE_BATCH = 50
RF_MAX_ESTIMATORS = 1000
INNER_VAL_FRAC = 0.2
# Tune HPs once on the richest feature set, reuse across M1–M4.
HP_TUNE_SPEC = "m4"
# Shared with XGB grid only; XGB-only params are not copied onto RF.
RF_PARAM_GRID = {
    "max_depth": [4, 6, 8],
}
RF_FIXED_PARAMS = {
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
    """Same class-balanced row weights as ml_residual_learning XGB/RF."""
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
    """Class-balanced weighted RMSE/MAE/R2 on log10(BMR)."""
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
    """Unweighted mean of per-class micro metrics (each class counts equally)."""
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
    """Micro, macro, and class-balanced weighted metrics (same as residual learning)."""
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
    return xgb, best_n, best_score


def model_key(algo: str, spec: str) -> str:
    return f"{algo}_{spec}"


def tune_shared_hyperparams(
    train_df: pd.DataFrame, random_state: int, n_trials: int
) -> dict:
    # Species-block HP search on HP_TUNE_SPEC features; shared by all M1–M4.
    fit_df, val_df = species_block_train_val_split(
        train_df, val_frac=INNER_VAL_FRAC, random_state=random_state
    )
    feature_cols = MODEL_FEATURES[HP_TUNE_SPEC]
    X_fit, X_val = encode_features(fit_df, val_df, feature_cols)
    y_fit = fit_df[LOG_TARGET].to_numpy(dtype=float)
    y_val = val_df[LOG_TARGET].to_numpy(dtype=float)
    sw_fit = make_class_balanced_sample_weight(fit_df)
    sw_val = make_class_balanced_sample_weight(val_df)

    rng = np.random.default_rng(random_state)
    # RF grid is small (shared max_depth only); evaluate the full Cartesian set.
    rf_sets = _cartesian_param_dicts(RF_PARAM_GRID)
    xgb_sets = draw_unique_param_sets(XGB_PARAM_GRID, n_trials, rng, "xgboost")

    best_rf: dict | None = None
    best_xgb: dict | None = None
    rf_trials: list[dict] = []
    xgb_trials: list[dict] = []

    print(
        f"  HP search on {HP_TUNE_SPEC}: RF={len(rf_sets)} configs, "
        f"XGB={len(xgb_sets)} unique trials "
        f"(inner val species={val_df['taxon_name'].nunique()}, rows={len(val_df)}; "
        f"class-balanced sample weights)",
        flush=True,
    )
    for trial, rf_p in enumerate(rf_sets):
        _, rf_n, rf_score = fit_rf_with_early_stopping(
            X_fit, y_fit, X_val, y_val, rf_p, sw_fit, random_state + trial
        )
        rf_row = {
            **rf_p,
            **RF_FIXED_PARAMS,
            "n_estimators": rf_n,
            "val_rmse": rf_score,
            "trial": trial,
        }
        rf_trials.append(rf_row)
        if best_rf is None or rf_score < best_rf["val_rmse"]:
            best_rf = dict(rf_row)
        print(
            f"    RF trial {trial + 1}/{len(rf_sets)}: val_rmse={rf_score:.4f} "
            f"max_depth={rf_p['max_depth']} n={rf_n}",
            flush=True,
        )

    for trial, xgb_p in enumerate(xgb_sets):
        _, xgb_n, xgb_score = fit_xgb_with_early_stopping(
            X_fit, y_fit, X_val, y_val, xgb_p, sw_fit, sw_val, random_state + trial
        )
        xgb_row = {**xgb_p, "n_estimators": xgb_n, "val_rmse": xgb_score, "trial": trial}
        xgb_trials.append(xgb_row)
        if best_xgb is None or xgb_score < best_xgb["val_rmse"]:
            best_xgb = dict(xgb_row)

        if (trial + 1) % 10 == 0 or trial == 0:
            print(
                f"    XGB trial {trial + 1}/{len(xgb_sets)}: val_rmse={xgb_score:.4f}",
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
        "train_class_weighted": 1,
        "class_weight_formula": CLASS_BALANCED_WEIGHT_FORMULA,
    }


def train_all_specs(
    train_df: pd.DataFrame, best_params: dict, random_state: int
) -> dict:
    """Fit RF/XGB for every M1–M4 with shared best HPs on full train."""
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
                    "Training:",
                    f"- Class-balanced sample weights: {CLASS_BALANCED_WEIGHT_FORMULA}",
                    "- Applied to RF/XGB HP search (fit fold) and final M1–M4 retraining.",
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

    sns.set_theme(style="whitegrid")
    fig_width = max(12.0, 0.8 * len(metrics_df) + 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))
    sns.barplot(data=metrics_df, x="model", y="rmse", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE (log10(BMR))")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=metrics_df, x="model", y="r2", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2 (log10(BMR))")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(f"ML Model Performance ({fold_tag})", fontsize=14)
    fig.tight_layout()
    fig.savefig(fold_out / "explore_ml_model_performance_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Class-balanced companion plot.
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))
    sns.barplot(data=metrics_df, x="model", y="rmse_bal", ax=axes[0], color="#4C72B0")
    axes[0].set_title("RMSE_bal (log10(BMR))")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=metrics_df, x="model", y="r2_bal", ax=axes[1], color="#C44E52")
    axes[1].set_title("R2_bal (log10(BMR))")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(f"ML Model Performance class-balanced ({fold_tag})", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        fold_out / "explore_ml_model_performance_comparison_bal.png",
        dpi=180,
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
    plt.savefig(fold_out / "explore_ml_residual_plot.png", dpi=180)
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
) -> None:
    model_dir = out_dir / fold_tag / "models"
    print(f"  Tuning + training M1–M4 on {len(train_df)} train rows...", flush=True)
    tune = tune_shared_hyperparams(train_df, random_state, n_hp_trials)
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
            "Tune RF/XGB M1–M4 with class-balanced sample weights inside the 80% "
            "development set, retrain on the complete development set, and evaluate "
            "on held-out test with micro/macro/class-balanced metrics."
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

    fold_splits = discover_fold_splits(split_dir, ["test"])
    _, train_path, test_path = fold_splits[0]
    train_df = add_mte_features(load_split_data(train_path))
    test_df = add_mte_features(load_split_data(test_path))
    assert_no_species_leakage(train_df, test_df)

    print("\n=== explore_ml test (direct development-train/test-eval) ===", flush=True)
    run_cv_fold(
        train_df=train_df,
        test_df=test_df,
        fold_tag="test",
        out_dir=out_dir,
        random_state=args.random_state,
        n_hp_trials=args.n_hp_trials,
        reset_hp_log=True,
    )
    write_explore_ml_species_accuracy(out_dir, ["test"])
    print(f"\nWrote explore_ml test results under: {out_dir}/test/")
    print(f"HP search log: {out_dir}/explore_ml_hp_search_trials.csv")


if __name__ == "__main__":
    main()
