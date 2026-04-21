
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

FEATURES = ["wet_Mass_kg", "temperature", "pc1", "pc2", "pc3", "pc4", "pc5"]
TARGET = "BMR"
MODEL_NAMES = ["power_law_3_4", "random_forest_residual", "xgboost_residual"]
POWER_LAW_FEATURES = ["wet_Mass_kg"]
TREE_MODEL_FEATURES = ["wet_Mass_kg", "temperature", "pc1", "pc2", "pc3", "pc4", "pc5"]


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["taxon_name", *FEATURES, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    for col in FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out[TARGET] > 0)].copy()
    out = out[out["taxon_name"] != ""].copy()
    return out.reset_index(drop=True)


def fit_alpha_three_quarter(mass_kg: np.ndarray, bmr_w: np.ndarray) -> float:
    log_m = np.log(mass_kg)
    log_y = np.log(bmr_w)
    return float(np.mean(log_y - 0.75 * log_m))


def residual_feature_frame(base_df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, np.ndarray]:
    mass_col = POWER_LAW_FEATURES[0]
    mass = base_df[mass_col].to_numpy()
    base_log_pred = alpha + 0.75 * np.log(mass)
    feature_data: dict[str, np.ndarray] = {"log_mass": np.log(mass)}
    for col in TREE_MODEL_FEATURES:
        if col == mass_col:
            continue
        feature_data[col] = base_df[col].to_numpy()
    feature_data["base_log_pred"] = base_log_pred
    X_res = pd.DataFrame(feature_data, index=base_df.index)
    return X_res, base_log_pred


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int
) -> tuple[dict[str, np.ndarray], dict[str, object], pd.DataFrame, pd.DataFrame]:
    y_train = train_df[TARGET].to_numpy()
    alpha = fit_alpha_three_quarter(train_df[POWER_LAW_FEATURES[0]].to_numpy(), y_train)

    X_train_res, train_log_base = residual_feature_frame(train_df, alpha)
    X_test_res, test_log_base = residual_feature_frame(test_df, alpha)
    residual_train = np.log(y_train) - train_log_base

    yhat_base = np.exp(test_log_base)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_res, residual_train)
    yhat_rf = np.exp(test_log_base + rf.predict(X_test_res))

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train_res, residual_train)
    yhat_xgb = np.exp(test_log_base + xgb.predict(X_test_res))

    preds = {
        "power_law_3_4": yhat_base,
        "random_forest_residual": yhat_rf,
        "xgboost_residual": yhat_xgb,
    }
    models = {"random_forest_residual": rf, "xgboost_residual": xgb}
    return preds, models, X_train_res, X_test_res


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_loss_curve(
    train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path, random_state: int
) -> pd.DataFrame:
    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()
    alpha = fit_alpha_three_quarter(train_df["wet_Mass_kg"].to_numpy(), y_train)
    X_train_res, train_log_base = residual_feature_frame(train_df, alpha)
    X_test_res, test_log_base = residual_feature_frame(test_df, alpha)
    residual_train = np.log(y_train) - train_log_base
    residual_test = np.log(y_test) - test_log_base

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
        eval_set=[(X_train_res, residual_train), (X_test_res, residual_test)],
        verbose=False,
    )

    evals = xgb_curve.evals_result()
    train_rmse = np.asarray(evals["validation_0"]["rmse"], dtype=float)
    test_rmse = np.asarray(evals["validation_1"]["rmse"], dtype=float)
    lc_df = pd.DataFrame(
        {
            "iteration": np.arange(1, len(train_rmse) + 1, dtype=int),
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        }
    )
    lc_df.to_csv(out_dir / "loss_curve_data.csv", index=False, encoding="utf-8")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    plt.plot(lc_df["iteration"], lc_df["train_rmse"], label="xgboost_train_rmse", linewidth=2)
    plt.plot(lc_df["iteration"], lc_df["test_rmse"], label="xgboost_test_rmse", linewidth=2)
    plt.xlabel("Boosting Iteration")
    plt.ylabel("RMSE (Residual Space)")
    plt.title("XGBoost Loss Curve")
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
        plt.scatter(
            pred_df["y_true"],
            pred_df["y_true"],
            s=12,
            alpha=0.35,
            color="#ff7f0e",
            label="observed",
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
    X_test_res: pd.DataFrame,
) -> None:
    shap_candidates = ["random_forest_residual", "xgboost_residual"]
    best = (
        metrics_df[metrics_df["model"].isin(shap_candidates)]
        .sort_values("rmse", ascending=True)
        .iloc[0]["model"]
    )
    model = models[best]

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


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BMR models on fixed train/test splits with residual learning "
            "and output required diagnostic figures."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train/train.csv"),
        help="Train CSV path.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/test/test.csv"),
        help="Test CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark"),
        help="Output directory.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_path = args.train if args.train.is_absolute() else root / args.train
    test_path = args.test if args.test.is_absolute() else root / args.test
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split_data(train_path)
    test_df = load_split_data(test_path)

    preds, models, _X_train_res, X_test_res = train_and_predict(
        train_df=train_df,
        test_df=test_df,
        random_state=args.random_state,
    )
    y_test = test_df[TARGET].to_numpy()

    metrics_rows = []
    for model in MODEL_NAMES:
        metrics_rows.append({"model": model, **evaluate(y_test, preds[model])})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    metrics_df.to_csv(out_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")

    pred_df = test_df[["taxon_name", *FEATURES]].copy()
    pred_df["y_true"] = y_test
    for model in MODEL_NAMES:
        pred_df[model] = preds[model]
    pred_df.to_csv(out_dir / "benchmark_predictions_test.csv", index=False, encoding="utf-8")

    save_loss_curve(
        train_df=train_df,
        test_df=test_df,
        out_dir=out_dir,
        random_state=args.random_state,
    )
    save_pred_and_residual_plots(out_dir=out_dir, pred_df=pred_df)
    save_performance_boxplot(
        out_dir=out_dir,
        y_true=y_test,
        pred_df=pred_df,
        random_state=args.random_state,
    )
    save_shap_outputs(
        out_dir=out_dir,
        metrics_df=metrics_df,
        models=models,
        X_test_res=X_test_res,
    )

    print(f"Train rows used: {len(train_df)}")
    print(f"Test rows used: {len(test_df)}")
    print(f"Power-law features: {POWER_LAW_FEATURES}")
    print(f"Tree-model features: {TREE_MODEL_FEATURES}")
    print(f"Saved outputs in: {out_dir}")
    print("\nBenchmark results:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
