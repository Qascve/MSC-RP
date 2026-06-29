
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET = "BMR"
LOG_TARGET = "log_BMR"
MODEL_NAMES = ["random_forest", "xgboost"]
POWER_LAW_FEATURES = ["log_mass"]
GROUP_CLASS_FILTERS: dict[str, str | None] = {
    "Teleostei": "Teleostei",
    "Mammalia": "Mammalia",
    "Insecta": "Insecta",
}
BASE_COLUMNS = [
    "class",
    "order",
    "family",
    "Genus",
    "species",
    "wet_Mass_g",
    "wet_Mass_kg",
    TARGET,
    "BMR_unit",
    "temperature",
    "temperature_unit",
    "Reference",
    "taxon_name",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pc5",
]
TREE_MODEL_FEATURES = [
    "class",
    "order",
    "family",
    "Genus",
    "species",
    "log_mass",
    "inv_kT",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pc5",
]


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["taxon_name", *TREE_MODEL_FEATURES, TARGET, LOG_TARGET]
    required = list(dict.fromkeys(required))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    categorical_features = ["class", "order", "family", "Genus", "species"]
    numeric_features = ["log_mass", "inv_kT", "pc1", "pc2", "pc3", "pc4", "pc5"]
    for col in categorical_features:
        out[col] = out[col].astype("string").str.strip()
    for col in numeric_features + [TARGET, LOG_TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["taxon_name"] = out["taxon_name"].replace("", pd.NA)
    for col in categorical_features:
        out[col] = out[col].replace("", pd.NA)
    out = out.dropna(subset=required).copy()
    out = out[(out["log_mass"].notna()) & (out["inv_kT"].notna()) & (out[TARGET] > 0)].copy()
    out = out[out["taxon_name"] != ""].copy()
    return out.reset_index(drop=True)


def load_full_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[BASE_COLUMNS].copy()
    categorical_features = ["class", "order", "family", "Genus", "species", "taxon_name"]
    numeric_features = ["wet_Mass_kg", TARGET, "temperature", "pc1", "pc2", "pc3", "pc4", "pc5"]
    for col in categorical_features:
        out[col] = out[col].astype("string").str.strip().replace("", pd.NA)
    for col in numeric_features:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=BASE_COLUMNS).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out[TARGET] > 0)].copy()
    out = out[(out["temperature"] + 273.15) > 0].copy()
    out["log_mass"] = np.log(out["wet_Mass_kg"].to_numpy())
    out[LOG_TARGET] = np.log(out[TARGET].to_numpy())
    temp_k = out["temperature"] + 273.15
    out["inv_kT"] = 1.0 / (8.617e-5 * temp_k.to_numpy())
    required = ["taxon_name", *TREE_MODEL_FEATURES, TARGET, LOG_TARGET]
    required = list(dict.fromkeys(required))
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return out.reset_index(drop=True)


def make_class_species_block_split(
    df: pd.DataFrame,
    test_species_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    out = df.copy().reset_index(drop=True)
    out["split_row_id"] = np.arange(len(out), dtype=int)

    test_species: set[str] = set()
    summary_rows: list[dict[str, object]] = []
    for class_name, class_df in out.groupby("class", sort=True):
        species_counts = class_df.groupby("taxon_name").size().sort_values(ascending=False)
        species_names = species_counts.index.astype(str).to_numpy()
        n_species = len(species_names)

        if n_species < 2:
            n_test_species = 0
            picked_species: list[str] = []
        else:
            n_test_species = int(round(n_species * test_species_ratio))
            n_test_species = max(1, min(n_species - 1, n_test_species))
            shuffled = species_names.copy()
            rng.shuffle(shuffled)
            picked_species = sorted(shuffled[:n_test_species].tolist())
            test_species.update(picked_species)

        test_rows = int(class_df["taxon_name"].astype(str).isin(picked_species).sum())
        summary_rows.append(
            {
                "class": str(class_name),
                "species_total": n_species,
                "species_train": n_species - n_test_species,
                "species_test": n_test_species,
                "rows_total": len(class_df),
                "rows_train": len(class_df) - test_rows,
                "rows_test": test_rows,
            }
        )

    is_test = out["taxon_name"].astype(str).isin(test_species)
    train_df = out[~is_test].drop(columns=["split_row_id"]).reset_index(drop=True)
    test_df = out[is_test].drop(columns=["split_row_id"]).reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Species-block split failed: empty train or test set.")
    leaked_species = set(train_df["taxon_name"].astype(str)).intersection(set(test_df["taxon_name"].astype(str)))
    if leaked_species:
        raise RuntimeError(f"Species leakage detected in block split: {sorted(leaked_species)[:5]}")
    return train_df, test_df, pd.DataFrame(summary_rows).sort_values("rows_total", ascending=False)


def fit_alpha_three_quarter(log_mass: np.ndarray, log_bmr: np.ndarray) -> float:
    return float(np.mean(log_bmr - 0.75 * log_mass))


def build_residual_feature_frames(
    train_df: pd.DataFrame, test_df: pd.DataFrame, alpha: float, model_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    mass_col = POWER_LAW_FEATURES[0]
    train_mass = train_df[mass_col].to_numpy()
    test_mass = test_df[mass_col].to_numpy()
    train_log_base = alpha + 0.75 * train_mass
    test_log_base = alpha + 0.75 * test_mass

    categorical_features = ["class", "order", "family", "Genus", "species"]
    tree_categorical_features = [col for col in model_features if col in categorical_features]
    train_raw = train_df[model_features].reset_index(drop=True).copy()
    test_raw = test_df[model_features].reset_index(drop=True).copy()
    merged_raw = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
    merged_encoded = pd.get_dummies(
        merged_raw,
        columns=tree_categorical_features,
        prefix=tree_categorical_features,
        dtype=float,
    )

    split_idx = len(train_raw)
    X_train_res = merged_encoded.iloc[:split_idx].copy()
    X_test_res = merged_encoded.iloc[split_idx:].copy()
    X_train_res["base_log_pred"] = train_log_base
    X_test_res["base_log_pred"] = test_log_base
    return X_train_res, X_test_res, train_log_base, test_log_base


def make_class_balanced_sample_weight(train_df: pd.DataFrame) -> np.ndarray:
    class_counts = train_df["class"].value_counts(dropna=False)
    weights = train_df["class"].map(lambda x: 1.0 / float(class_counts.loc[x])).to_numpy(dtype=float)
    return weights / weights.mean()


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    balance_classes: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, pd.DataFrame]]:
    y_train = train_df[TARGET].to_numpy()
    log_y_train = train_df[LOG_TARGET].to_numpy()
    alpha = fit_alpha_three_quarter(train_df[POWER_LAW_FEATURES[0]].to_numpy(), log_y_train)
    residual_train = log_y_train - (alpha + 0.75 * train_df[POWER_LAW_FEATURES[0]].to_numpy())
    sample_weight = make_class_balanced_sample_weight(train_df) if balance_classes else None

    X_train_rf, X_test_rf, _rf_train_base, rf_test_base = build_residual_feature_frames(
        train_df, test_df, alpha, TREE_MODEL_FEATURES
    )
    X_train_xgb, X_test_xgb, _xgb_train_base, xgb_test_base = build_residual_feature_frames(
        train_df, test_df, alpha, TREE_MODEL_FEATURES
    )

    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=4,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_rf, residual_train, sample_weight=sample_weight)
    yhat_rf = np.exp(rf_test_base + rf.predict(X_test_rf))

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=6000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train_xgb, residual_train, sample_weight=sample_weight)
    yhat_xgb = np.exp(xgb_test_base + xgb.predict(X_test_xgb))

    preds = {
        "random_forest": yhat_rf,
        "xgboost": yhat_xgb,
    }
    models = {"random_forest": rf, "xgboost": xgb}
    shap_inputs = {"random_forest": X_test_rf, "xgboost": X_test_xgb}
    return preds, models, shap_inputs


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_loss_curve(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    random_state: int,
    balance_classes: bool,
) -> pd.DataFrame:
    log_y_train = train_df[LOG_TARGET].to_numpy()
    alpha = fit_alpha_three_quarter(train_df[POWER_LAW_FEATURES[0]].to_numpy(), log_y_train)
    X_train_res, _X_test_res, train_log_base, _test_log_base = build_residual_feature_frames(
        train_df, test_df, alpha, TREE_MODEL_FEATURES
    )
    residual_train = log_y_train - train_log_base
    sample_weight = make_class_balanced_sample_weight(train_df) if balance_classes else None

    xgb_curve = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse",
    )
    xgb_curve.fit(
        X_train_res,
        residual_train,
        sample_weight=sample_weight,
        eval_set=[(X_train_res, residual_train)],
        sample_weight_eval_set=[sample_weight] if sample_weight is not None else None,
        verbose=False,
    )

    evals = xgb_curve.evals_result()
    train_rmse = np.asarray(evals["validation_0"]["rmse"], dtype=float)
    lc_df = pd.DataFrame(
        {
            "iteration": np.arange(1, len(train_rmse) + 1, dtype=int),
            "train_rmse": train_rmse,
        }
    )
    lc_df.to_csv(out_dir / "loss_curve_data.csv", index=False, encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    plt.plot(lc_df["iteration"], lc_df["train_rmse"], label="xgboost_train_rmse", linewidth=2)
    plt.xlabel("Boosting Iteration")
    plt.ylabel("RMSE (Residual Space)")
    plt.title("XGBoost Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()
    return lc_df


def save_pred_and_residual_plots(out_dir: Path, pred_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    for model in MODEL_NAMES:
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
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Observed BMR (W)")
        plt.ylabel("Predicted BMR (W)")
        plt.title(f"Observed vs Predicted ({model})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"observed_vs_predicted_scatter_{model}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 7))
    for model in MODEL_NAMES:
        residual = pred_df["y_true"] - pred_df[model]
        plt.scatter(pred_df[model], residual, s=14, alpha=0.45, label=model)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Predicted BMR (W)")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("Residual Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "residual_plot.png", dpi=160)
    plt.close()


def save_performance_boxplot(
    out_dir: Path, y_true: np.ndarray, pred_df: pd.DataFrame, random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_boot = 200
    rows: list[dict[str, float | str]] = []
    n = len(y_true)
    for model in MODEL_NAMES:
        y_pred = pred_df[model].to_numpy()
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            rmse_b = float(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx])))
            rows.append({"model": model, "bootstrap_id": b, "rmse": rmse_b})
    perf_df = pd.DataFrame(rows)
    perf_df.to_csv(out_dir / "performance_boxplot_data.csv", index=False, encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=perf_df, x="model", y="rmse")
    plt.xlabel("Model")
    plt.ylabel("Bootstrap RMSE")
    plt.title("Model Performance Boxplot")
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
    shap_candidates = ["random_forest", "xgboost"]
    best = (
        metrics_df[metrics_df["model"].isin(shap_candidates)]
        .sort_values("rmse", ascending=True)
        .iloc[0]["model"]
    )
    model = models[best]
    X_test_res = shap_inputs[best]

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
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_res, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=160)
    plt.close()


def run_single_group(
    group_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    random_state: int,
    balance_classes: bool,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, models, shap_inputs = train_and_predict(
        train_df=train_df,
        test_df=test_df,
        random_state=random_state,
        balance_classes=balance_classes,
    )
    y_test = test_df[TARGET].to_numpy()

    metrics_rows = []
    for model in MODEL_NAMES:
        metrics_rows.append({"model": model, **evaluate(y_test, preds[model])})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    metrics_df.to_csv(out_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")

    pred_df = test_df[["taxon_name", *TREE_MODEL_FEATURES]].copy()
    pred_df["y_true"] = y_test
    for model in MODEL_NAMES:
        pred_df[model] = preds[model]
    pred_df.to_csv(out_dir / "benchmark_predictions_test.csv", index=False, encoding="utf-8")

    save_loss_curve(
        train_df=train_df,
        test_df=test_df,
        out_dir=out_dir,
        random_state=random_state,
        balance_classes=balance_classes,
    )
    save_pred_and_residual_plots(out_dir=out_dir, pred_df=pred_df)
    save_performance_boxplot(
        out_dir=out_dir,
        y_true=y_test,
        pred_df=pred_df,
        random_state=random_state,
    )
    save_shap_outputs(
        out_dir=out_dir,
        metrics_df=metrics_df,
        models=models,
        shap_inputs=shap_inputs,
    )

    print(f"\n[{group_name}] Train rows used: {len(train_df)}")
    print(f"[{group_name}] Test rows used: {len(test_df)}")
    print(f"[{group_name}] Power-law features: {POWER_LAW_FEATURES}")
    print(f"[{group_name}] Tree-model features: {TREE_MODEL_FEATURES}")
    print(f"[{group_name}] Class-balanced training weights: {balance_classes}")
    print(f"[{group_name}] Saved outputs in: {out_dir}")
    print(f"\n[{group_name}] Benchmark results:")
    print(metrics_df.to_string(index=False))
    return metrics_df


def main() -> None:
    print("Running ml_residual_learning.py")
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train class-balanced residual-learning models and evaluate them with "
            "class-level species-block cross-validation."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/merge_phylo.csv"),
        help="Full input CSV path used to create a class-level species-block split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark"),
        help="Output directory. Results are stored under class-name subfolders.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for the generated species-block train/test files.",
    )
    parser.add_argument(
        "--test-species-ratio",
        type=float,
        default=0.3,
        help="Per-class ratio of species blocks held out for testing.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if not 0 < args.test_species_ratio < 1:
        raise ValueError("--test-species-ratio must be in (0, 1).")

    input_path = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    split_out_dir = (
        args.split_output_dir if args.split_output_dir.is_absolute() else root / args.split_output_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    split_out_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_full_data(input_path)
    train_df, test_df_all, split_summary = make_class_species_block_split(
        full_df,
        test_species_ratio=args.test_species_ratio,
        random_state=args.random_state,
    )
    train_df.to_csv(split_out_dir / "train.csv", index=False, encoding="utf-8")
    test_df_all.to_csv(split_out_dir / "test.csv", index=False, encoding="utf-8")
    split_summary.to_csv(split_out_dir / "class_species_block_split_summary.csv", index=False, encoding="utf-8")
    split_summary.to_csv(out_dir / "class_species_block_split_summary.csv", index=False, encoding="utf-8")

    summary_rows: list[dict[str, float | str]] = []
    for group_name, class_name in GROUP_CLASS_FILTERS.items():
        group_test_df = test_df_all[test_df_all["class"] == class_name].copy()
        if group_test_df.empty:
            raise ValueError(f"Group {group_name} has no rows for class={class_name}.")

        group_out_dir = out_dir / group_name
        metrics_df = run_single_group(
            group_name=group_name,
            train_df=train_df,
            test_df=group_test_df,
            out_dir=group_out_dir,
            random_state=args.random_state,
            balance_classes=True,
        )
        best_row = metrics_df.sort_values("rmse").iloc[0]
        summary_rows.append(
            {
                "group": group_name,
                "class_filter": class_name,
                "test_rows": int(len(group_test_df)),
                "best_model": str(best_row["model"]),
                "best_rmse": float(best_row["rmse"]),
                "best_mae": float(best_row["mae"]),
                "best_r2": float(best_row["r2"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "benchmark_summary_groups.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nSaved species-block train: {split_out_dir / 'train.csv'}")
    print(f"Saved species-block test: {split_out_dir / 'test.csv'}")
    print(f"Saved split summary: {split_out_dir / 'class_species_block_split_summary.csv'}")
    print(f"\nSaved group summary: {summary_path}")


if __name__ == "__main__":
    main()
