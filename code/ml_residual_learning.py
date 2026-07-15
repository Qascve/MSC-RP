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
TAXONOMY_MODEL_FEATURES = ["Genus", "species"]
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

EARLY_STOPPING_ROUNDS = 10
XGB_MAX_ESTIMATORS = 1000
INNER_VAL_FRAC = 0.2
RF_FIXED_PARAMS = {
    "n_estimators": 600,
    "max_depth": 4,
    "min_samples_leaf": 5,
    "max_features": 1.0,
}
XGB_PARAM_GRID = {
    "max_depth": [4, 6, 8, 10],
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
    """Load a fixed train/test CSV produced by split_train_test_bmr.py."""
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
    wanted = folds if folds else ["fold1", "fold2", "test"]
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
        "Expected fold1/, fold2/, and test/ with train.csv & test.csv. "
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


def species_block_train_val_split(
    train_df: pd.DataFrame, val_frac: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of training species for inner validation (no row leakage)."""
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
    train_df: pd.DataFrame,
    random_state: int,
    balance_classes: bool,
) -> dict:
    """Exhaustively tune XGB; train RF once with fixed parameters."""
    fit_df, val_df = species_block_train_val_split(
        train_df, val_frac=INNER_VAL_FRAC, random_state=random_state
    )
    alpha = fit_alpha_three_quarter(
        train_df[POWER_LAW_FEATURES[0]].to_numpy(),
        train_df[LOG_TARGET].to_numpy(dtype=float),
    )

    # Feature columns from the fit split (val may have unseen categories → zeros).
    X_fit, residual_fit, _ = encode_train_frame(fit_df, alpha)
    feature_columns = list(X_fit.columns)
    X_val, residual_val, _ = encode_train_frame(val_df, alpha, feature_columns)
    sw_fit = make_class_balanced_sample_weight(fit_df) if balance_classes else None
    sw_val = make_class_balanced_sample_weight(val_df) if balance_classes else None

    xgb_param_sets = _cartesian_param_dicts(XGB_PARAM_GRID)
    n_combinations = len(xgb_param_sets)
    if n_combinations == 0:
        raise ValueError("XGB_PARAM_GRID produced no combinations.")
    xgb_keys = {_params_key(p) for p in xgb_param_sets}
    if len(xgb_keys) != n_combinations:
        raise RuntimeError("Hyperparameter search produced duplicate trials.")

    xgb_trials: list[dict] = []
    best_xgb: dict | None = None
    best_xgb_hist: list[float] = []

    print(
        f"  XGB exhaustive grid search: {n_combinations} combinations, "
        f"inner species val={val_df['taxon_name'].nunique()} species / {len(val_df)} rows",
        flush=True,
    )
    for trial, xgb_params in enumerate(xgb_param_sets):
        _, xgb_n, xgb_score, xgb_hist = fit_xgb_with_early_stopping(
            X_fit,
            residual_fit,
            X_val,
            residual_val,
            xgb_params,
            sw_fit,
            sw_val,
            random_state,
        )
        xgb_row = {**xgb_params, "n_estimators": xgb_n, "val_rmse": xgb_score, "trial": trial}
        xgb_trials.append(xgb_row)
        if best_xgb is None or xgb_score < best_xgb["val_rmse"]:
            best_xgb = dict(xgb_row)
            best_xgb_hist = list(xgb_hist)
        if (trial + 1) % 10 == 0 or trial == 0:
            print(
                f"    combination {trial + 1}/{n_combinations}: "
                f"best_so_far val_rmse={best_xgb['val_rmse']:.4f} "
                f"(trial={int(best_xgb['trial']) + 1})",
                flush=True,
            )

    assert best_xgb is not None

    # XGB uses the winning trial; RF uses one fixed configuration without tuning.
    xgb_n_final = int(best_xgb["n_estimators"])
    xgb_hist = list(best_xgb_hist)

    # Full-train feature space (may include categories only in former val species).
    X_all, residual_all, _ = encode_train_frame(train_df, alpha)
    feature_columns = list(X_all.columns)
    sw_all = make_class_balanced_sample_weight(train_df) if balance_classes else None

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
        f"  Best XGB trial={best_xgb['trial']} val_rmse={best_xgb['val_rmse']:.4f} "
        f"n_estimators={xgb_n_final} "
        f"params={{max_depth={best_xgb['max_depth']}, lr={best_xgb['learning_rate']}}}",
        flush=True,
    )

    xgb_score = float(best_xgb["val_rmse"])

    return {
        "models": {"random_forest": rf_full, "xgboost": xgb_full},
        "alpha": float(alpha),
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
                "search_val_rmse": xgb_score,
            },
        },
        "search_trials": {
            "xgboost": pd.DataFrame(xgb_trials),
        },
        "loss_curves": {
            "xgboost": xgb_hist,
        },
        "inner_val_species": int(val_df["taxon_name"].nunique()),
        "inner_val_rows": int(len(val_df)),
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
    reset: bool = False,
) -> Path:
    """
    Write/append all exhaustive XGB combinations + inner-val RMSE.
    RF is omitted because it uses fixed parameters and is not tuned.

    Per-fold files are authoritative. If a CSV is open and locked on Windows,
    write an ``*_unlocked.csv`` fallback and continue instead of losing the
    completed grid search.
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

    # Save authoritative per-fold outputs before touching the combined CSV.
    fold_path = write_with_fallback(
        new_df, benchmark_dir / f"hp_search_trials_{fold_tag}.csv"
    )
    best_df = new_df.loc[new_df["is_best"] == 1].copy()
    best_path = write_with_fallback(
        best_df, benchmark_dir / f"xgb_best_params_{fold_tag}.csv"
    )

    model_dir = benchmark_dir / "all" / fold_tag / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_with_fallback(best_df, model_dir / "xgb_best_params.csv")

    # The combined CSV is convenient but non-authoritative.
    if reset or not out_path.exists():
        combined_df = new_df
    else:
        try:
            old_df = pd.read_csv(out_path)
            keep = (
                old_df[old_df["fold"].astype(str) != str(fold_tag)]
                if "fold" in old_df.columns
                else old_df.iloc[0:0]
            )
            combined_df = pd.concat([keep, new_df], ignore_index=True)
        except PermissionError:
            combined_df = new_df
    actual_out_path = write_with_fallback(combined_df, out_path)

    print(
        f"  Wrote XGB exhaustive search ({len(new_df)} rows) -> {fold_path}",
        flush=True,
    )
    print(
        f"  Wrote best XGB parameters for {fold_tag} -> {best_path}",
        flush=True,
    )
    return actual_out_path


def fold_model_eval_rmse(
    benchmark_dir: Path, fold_tag: str, model_name: str
) -> float:
    """Return this model family's outer-fold evaluation RMSE."""
    metrics_path = benchmark_dir / "all" / fold_tag / "benchmark_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        hit = metrics.loc[metrics["model"] == model_name, "rmse"]
        if not hit.empty and np.isfinite(hit.iloc[0]):
            return float(hit.iloc[0])
    raise FileNotFoundError(
        f"Missing {model_name} fold metric in {metrics_path}; rerun fold1/fold2."
    )


def select_best_models_from_cv_folds(
    benchmark_dir: Path, source_folds: list[str] | None = None
) -> dict[str, dict]:
    """
    Independently pick the better saved RF and XGB from f_1/f_2.
    The held-out test set is never used for selection.
    """
    source_folds = source_folds or ["f_1", "f_2"]
    selected: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        best: dict | None = None
        for fold_tag in source_folds:
            model_dir = benchmark_dir / "all" / fold_tag / "models"
            model_path = model_dir / f"{model_name}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing saved {model_name} for {fold_tag}: {model_path}. "
                    "Rerun fold1/fold2 with the updated script."
                )
            score = fold_model_eval_rmse(benchmark_dir, fold_tag, model_name)
            if best is None or score < best["score"]:
                best = {
                    "fold": fold_tag,
                    "model_name": model_name,
                    "score": score,
                    "model_dir": model_dir,
                }
        assert best is not None
        selected[model_name] = best
        print(
            f"  test uses {model_name} from {best['fold']} "
            f"(fold eval RMSE={best['score']:.4f})",
            flush=True,
        )
    return selected


def save_test_model_selection(
    selection: dict[str, dict], model_dir: Path
) -> Path:
    """Record the independently selected RF and XGB source folds."""
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {
            "source_fold": info["fold"],
            "source_model_dir": str(info["model_dir"]),
            "selection_rmse": info["score"],
        }
        for name, info in selection.items()
    }
    out = model_dir / "selection.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


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


def load_selected_models(
    selection: dict[str, dict],
) -> dict[str, dict]:
    """Load RF/XGB from their independently selected source folds."""
    return {
        name: load_model_bundle(info["model_dir"])
        for name, info in selection.items()
    }


def predict_selected_models(
    selected_bundles: dict[str, dict], df: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, object]]:
    preds: dict[str, np.ndarray] = {}
    shap_inputs: dict[str, pd.DataFrame] = {}
    models: dict[str, object] = {}
    for name in MODEL_NAMES:
        bundle = selected_bundles[name]
        single_bundle = {
            **bundle,
            "models": {name: bundle["models"][name]},
        }
        p, x = predict_log_bmr(single_bundle, df)
        preds[name] = p[name]
        shap_inputs[name] = x[name]
        models[name] = bundle["models"][name]
    return preds, shap_inputs, models


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


def run_one_fold_global(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_tag: str,
    out_dir: Path,
    random_state: int,
    balance_classes: bool,
    reset_hp_log: bool = False,
) -> list[str]:
    """
    Train fixed-parameter RF plus tuned XGB, save/reload both, then evaluate.
    """
    model_dir = out_dir / "all" / fold_tag / "models"

    print(f"  Tuning + training global models on {len(train_df)} train rows...", flush=True)
    bundle = tune_models_on_train(
        train_df=train_df,
        random_state=random_state,
        balance_classes=balance_classes,
    )
    bundle["fold_tag"] = fold_tag
    save_model_bundle(bundle, model_dir)
    write_hp_search_trials_csv(
        benchmark_dir=out_dir,
        fold_tag=fold_tag,
        bundle=bundle,
        reset=reset_hp_log,
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


def run_test_with_best_cv_models(
    test_df: pd.DataFrame,
    out_dir: Path,
    random_state: int,
    source_folds: list[str] | None = None,
) -> list[str]:
    """
    Held-out test: no HP search. Independently select the better f_1/f_2
    RF and XGB models, then evaluate both on the 20% test set.
    """
    fold_tag = "test"
    model_dir = out_dir / "all" / fold_tag / "models"
    print("  Selecting best saved RF and XGB folds (no HP search on test)...", flush=True)
    selection = select_best_models_from_cv_folds(out_dir, source_folds=source_folds)
    save_test_model_selection(selection, model_dir)
    selected_bundles = load_selected_models(selection)
    preds, shap_inputs, models = predict_selected_models(selected_bundles, test_df)
    return evaluate_fold_predictions(
        fold_tag=fold_tag,
        test_df=test_df,
        preds=preds,
        shap_inputs=shap_inputs,
        models=models,
        out_dir=out_dir,
        model_dir=model_dir,
        random_state=random_state,
    )


def main() -> None:
    print("Running ml_residual_learning.py")
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed-parameter RF and exhaustively tuned XGB residual models on "
            "fold1/fold2, "
            "save/reload both separately, then independently select the best RF and "
            "XGB fold models for held-out test evaluation."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing fixed fold1/fold2/test train.csv and test.csv files.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["fold1", "fold2", "test"],
        help="Which fixed splits to run (default: fold1 fold2 test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark"),
        help="Output directory. Results are stored under <group>/f_1, f_2, and test.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    split_dir = args.split_dir if args.split_dir.is_absolute() else root / args.split_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_splits = discover_fold_splits(split_dir, folds=list(args.folds))
    fold_tags_done: list[str] = []
    groups_done: list[str] = []
    hp_log_reset_done = False

    for fold_name, train_path, test_path in fold_splits:
        fold_tag = FOLD_DIR_NAMES.get(fold_name, fold_name)
        fold_tags_done.append(fold_tag)
        print(f"\n=== {fold_tag} ({fold_name}): {train_path} | {test_path} ===", flush=True)
        test_df_all = load_split_data(test_path)

        if fold_tag == "test" or fold_name == "test":
            # Optional leakage check against the 80% train if present.
            if train_path.exists():
                train_df = load_split_data(train_path)
                assert_no_species_leakage(train_df, test_df_all)
            done = run_test_with_best_cv_models(
                test_df=test_df_all,
                out_dir=out_dir,
                random_state=args.random_state,
                source_folds=["f_1", "f_2"],
            )
        else:
            train_df = load_split_data(train_path)
            assert_no_species_leakage(train_df, test_df_all)
            reset_hp_log = not hp_log_reset_done
            done = run_one_fold_global(
                train_df=train_df,
                test_df=test_df_all,
                fold_tag=fold_tag,
                out_dir=out_dir,
                random_state=args.random_state,
                balance_classes=True,
                reset_hp_log=reset_hp_log,
            )
            hp_log_reset_done = True

        for g in done:
            if g not in groups_done:
                groups_done.append(g)

    if fold_tags_done:
        for group_name in groups_done:
            write_group_species_accuracy(
                group_dir=out_dir / group_name,
                fold_tags=fold_tags_done,
            )
    else:
        print("Skip species accuracy CSV: no evaluation splits completed.")

    print(f"\nRead fixed splits from: {split_dir}")
    print(f"Wrote fold results under: {out_dir}/<group>/f_1|f_2|test/")
    print(f"Global models under: {out_dir}/all/<fold>/models/")
    print(f"Exhaustive XGB search log: {out_dir}/hp_search_trials.csv")
    print(f"Best XGB params: {out_dir}/xgb_best_params_f_1.csv and xgb_best_params_f_2.csv")
    print(f"Wrote species accuracy under: {out_dir}/<group>/species_accuracy.csv")


if __name__ == "__main__":
    main()
