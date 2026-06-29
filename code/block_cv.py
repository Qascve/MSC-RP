#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

K_BOLTZMANN_EV_PER_K = 8.617e-5
TARGET = "BMR"
LOG_TARGET = "log_BMR"
CLASS_COL = "class"
SPECIES_COL = "taxon_name"
PHYLO_COLS = ["pc1", "pc2", "pc3", "pc4", "pc5"]
BASE_COLUMNS = [
    CLASS_COL,
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
    SPECIES_COL,
    *PHYLO_COLS,
]
OUTPUT_COLUMNS = [*BASE_COLUMNS, "log_mass", LOG_TARGET, "inv_kT"]

SCHEME_CONFIG: dict[str, dict[str, str]] = {
    "class_balanced_species": {
        "block_column": SPECIES_COL,
        "description": "Class-balanced folds with whole species as blocks.",
    },
    "species_out": {
        "block_column": SPECIES_COL,
        "description": "Leave-species-out folds.",
    },
    "genus_out": {
        "block_column": "Genus",
        "description": "Leave-genus-out folds.",
    },
    "family_out": {
        "block_column": "family",
        "description": "Leave-family-out folds.",
    },
    "phylo_cluster_out": {
        "block_column": "phylo_cluster",
        "description": "Leave-phylogenetic-embedding-cluster-out folds.",
    },
}

TARGET_CLASS_GROUPS: dict[str, str] = {
    "A": "Teleostei",
    "B": "Mammalia",
    "C": "Insecta",
}


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_model_api(root: Path):
    code_dir = root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from ml_residual_learning import MODEL_NAMES, train_and_predict

    return MODEL_NAMES, train_and_predict


def clean_input_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[BASE_COLUMNS].copy()
    text_cols = [CLASS_COL, "order", "family", "Genus", "species", SPECIES_COL]
    for col in text_cols:
        out[col] = out[col].astype("string").str.strip().replace("", pd.NA)

    numeric_cols = ["wet_Mass_g", "wet_Mass_kg", TARGET, "temperature", *PHYLO_COLS]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=BASE_COLUMNS).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out[TARGET] > 0)].copy()
    out = out[(out["temperature"] + 273.15) > 0].copy()

    temp_k = out["temperature"] + 273.15
    out["log_mass"] = np.log(out["wet_Mass_kg"].to_numpy())
    out[LOG_TARGET] = np.log(out[TARGET].to_numpy())
    out["inv_kT"] = 1.0 / (K_BOLTZMANN_EV_PER_K * temp_k.to_numpy())
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=OUTPUT_COLUMNS).copy()
    out = out.reset_index(drop=True)
    out["cv_row_id"] = np.arange(len(out), dtype=int)
    if out.empty:
        raise ValueError("No valid rows left after block-CV cleaning.")
    return out


def add_phylo_clusters(df: pd.DataFrame, n_blocks: int, seed: int) -> pd.DataFrame:
    out = df.copy()
    n_clusters = min(max(2, n_blocks), len(out))
    X = out[PHYLO_COLS].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X)
    out["_phylo_cluster_raw"] = pd.Series(labels, index=out.index).map(lambda x: f"phylo_cluster_{x:03d}")

    target_block_rows = max(1, int(np.ceil(len(out) / n_blocks)))
    final_labels = pd.Series(index=out.index, dtype="string")
    for cluster_id, cluster_df in out.groupby("_phylo_cluster_raw", sort=True):
        if len(cluster_df) <= target_block_rows:
            final_labels.loc[cluster_df.index] = cluster_id
            continue

        species_counts = (
            cluster_df.groupby(SPECIES_COL, dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="rows")
        )
        n_parts = max(2, int(np.ceil(len(cluster_df) / target_block_rows)))
        part_loads = np.zeros(n_parts, dtype=int)
        species_to_part: dict[str, int] = {}
        for species_row in species_counts.itertuples(index=False):
            part_idx = int(np.argmin(part_loads))
            species_name = str(getattr(species_row, SPECIES_COL))
            species_to_part[species_name] = part_idx
            part_loads[part_idx] += int(species_row.rows)

        cluster_species = cluster_df[SPECIES_COL].astype("string").map(lambda x: str(x))
        part_ids = cluster_species.map(species_to_part).astype(int)
        final_labels.loc[cluster_df.index] = [
            f"{cluster_id}_part_{part_idx:02d}" for part_idx in part_ids
        ]

    out["phylo_cluster"] = final_labels.astype("string")
    out = out.drop(columns=["_phylo_cluster_raw"])
    return out


def _block_class_counts(df: pd.DataFrame, block_col: str) -> pd.DataFrame:
    counts = (
        df.groupby([block_col, CLASS_COL], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    block_totals = counts.groupby(block_col, dropna=False)["rows"].sum().rename("block_rows")
    dominant = (
        counts.sort_values([block_col, "rows"], ascending=[True, False])
        .drop_duplicates(block_col)
        .set_index(block_col)[CLASS_COL]
        .rename("dominant_class")
    )
    return counts.join(block_totals, on=block_col).join(dominant, on=block_col)


def assign_blocks_balanced(
    df: pd.DataFrame,
    block_col: str,
    n_splits: int,
    seed: int,
    class_balanced: bool,
) -> pd.DataFrame:
    block_counts = _block_class_counts(df, block_col)
    block_rows = (
        block_counts[[block_col, "block_rows", "dominant_class"]]
        .drop_duplicates(block_col)
        .sort_values(["block_rows", block_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(seed)
    same_size_groups = []
    for _rows, group in block_rows.groupby("block_rows", sort=False):
        same_size_groups.append(group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))))
    block_rows = pd.concat(same_size_groups, ignore_index=True)

    fold_total_load = np.zeros(n_splits, dtype=int)
    fold_class_load: dict[str, np.ndarray] = {}
    assignments: list[dict[str, object]] = []

    for row in block_rows.itertuples(index=False):
        block_id = getattr(row, block_col)
        block_slice = block_counts[block_counts[block_col] == block_id]
        class_vector = {
            str(r[CLASS_COL]): int(r["rows"])
            for _, r in block_slice.iterrows()
        }

        best_fold = 0
        best_score: tuple[float, float, float] | None = None
        for fold_idx in range(n_splits):
            total_after = fold_total_load[fold_idx] + int(row.block_rows)
            if class_balanced:
                class_pressure = 0.0
                for class_name, class_rows in class_vector.items():
                    loads = fold_class_load.setdefault(class_name, np.zeros(n_splits, dtype=int))
                    class_pressure += float(loads[fold_idx] + class_rows)
            else:
                class_pressure = 0.0
            score = (class_pressure, float(total_after), float(fold_idx))
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold_idx

        fold_total_load[best_fold] += int(row.block_rows)
        for class_name, class_rows in class_vector.items():
            fold_class_load.setdefault(class_name, np.zeros(n_splits, dtype=int))[best_fold] += class_rows
        assignments.append(
            {
                "block_id": str(block_id),
                "block_column": block_col,
                "fold": best_fold + 1,
                "rows": int(row.block_rows),
                "dominant_class": str(row.dominant_class),
            }
        )

    return pd.DataFrame(assignments)


def make_scheme_folds(
    df: pd.DataFrame,
    scheme: str,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = SCHEME_CONFIG[scheme]
    block_col = config["block_column"]
    if block_col not in df.columns:
        raise KeyError(f"Scheme {scheme} requires missing block column: {block_col}")

    class_balanced = scheme == "class_balanced_species"
    block_assignments = assign_blocks_balanced(
        df=df,
        block_col=block_col,
        n_splits=n_splits,
        seed=seed,
        class_balanced=class_balanced,
    )
    fold_map = block_assignments.set_index("block_id")["fold"].to_dict()
    row_folds = df[["cv_row_id", CLASS_COL, SPECIES_COL, "Genus", "family", "order"]].copy()
    row_folds["scheme"] = scheme
    row_folds["block_column"] = block_col
    row_folds["block_id"] = df[block_col].astype("string").to_numpy()
    row_folds["fold"] = row_folds["block_id"].map(fold_map).astype(int)
    return row_folds, block_assignments


def write_fold_files(
    df: pd.DataFrame,
    row_folds: pd.DataFrame,
    split_dir: Path,
    scheme: str,
    n_splits: int,
) -> pd.DataFrame:
    scheme_dir = split_dir / scheme
    scheme_dir.mkdir(parents=True, exist_ok=True)
    data_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    summary_rows: list[dict[str, object]] = []

    for fold in range(1, n_splits + 1):
        fold_dir = scheme_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_ids = set(row_folds.loc[row_folds["fold"] == fold, "cv_row_id"].astype(int))
        test_df = df[df["cv_row_id"].isin(test_ids)].copy()
        train_df = df[~df["cv_row_id"].isin(test_ids)].copy()
        train_df[data_cols].to_csv(fold_dir / "train.csv", index=False, encoding="utf-8")
        test_df[data_cols].to_csv(fold_dir / "test.csv", index=False, encoding="utf-8")

        summary_rows.append(
            {
                "scheme": scheme,
                "fold": fold,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_classes": train_df[CLASS_COL].nunique(),
                "test_classes": test_df[CLASS_COL].nunique(),
                "test_blocks": row_folds.loc[row_folds["fold"] == fold, "block_id"].nunique(),
            }
        )

    return pd.DataFrame(summary_rows)


def safe_evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": np.nan,
    }
    if len(y_true) >= 2 and not np.isclose(y_true.var(), 0.0):
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def summarize_prediction_metrics(
    pred_df: pd.DataFrame,
    model_names: list[str],
    block_col_name: str = "block_id",
    max_class_weight: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_class_rows: list[dict[str, object]] = []
    per_block_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for model in model_names:
        micro = safe_evaluate(pred_df["y_true"].to_numpy(), pred_df[model].to_numpy())
        summary_rows.append({"model": model, "metric_scope": "micro", "group": "ALL", **micro})

        class_metrics = []
        for class_name, group in pred_df.groupby(CLASS_COL, sort=True):
            metrics = safe_evaluate(group["y_true"].to_numpy(), group[model].to_numpy())
            row = {
                "model": model,
                CLASS_COL: class_name,
                "rows": len(group),
                **metrics,
            }
            per_class_rows.append(row)
            class_metrics.append(row)

        class_df = pd.DataFrame(class_metrics)
        eligible = class_df[class_df["rows"] >= 2].copy()
        if not eligible.empty:
            macro = eligible[["rmse", "mae", "r2"]].mean(numeric_only=True).to_dict()
            summary_rows.append({"model": model, "metric_scope": "macro_class", "group": "ALL", **macro})

            raw_weights = eligible["rows"].to_numpy(dtype=float)
            raw_weights = raw_weights / raw_weights.sum()
            capped = np.minimum(raw_weights, max_class_weight)
            weights = capped / capped.sum()
            weighted = {}
            for metric in ["rmse", "mae", "r2"]:
                values = eligible[metric].to_numpy(dtype=float)
                valid = ~np.isnan(values)
                if bool(valid.any()):
                    metric_weights = weights[valid] / weights[valid].sum()
                    weighted[metric] = float(np.sum(values[valid] * metric_weights))
                else:
                    weighted[metric] = np.nan
            summary_rows.append(
                {
                    "model": model,
                    "metric_scope": "weighted_macro_class_capped",
                    "group": "ALL",
                    **weighted,
                }
            )

        for block_id, group in pred_df.groupby(block_col_name, sort=True):
            metrics = safe_evaluate(group["y_true"].to_numpy(), group[model].to_numpy())
            per_block_rows.append(
                {
                    "model": model,
                    block_col_name: block_id,
                    "rows": len(group),
                    "dominant_class": group[CLASS_COL].mode(dropna=True).iloc[0],
                    **metrics,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(per_class_rows), pd.DataFrame(per_block_rows)


def run_residual_learning_cv(
    df: pd.DataFrame,
    row_folds: pd.DataFrame,
    results_dir: Path,
    scheme: str,
    random_state: int,
    root: Path,
    balance_classes: bool = False,
) -> None:
    model_names, train_and_predict = load_model_api(root)
    scheme_dir = results_dir / scheme
    scheme_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, object]] = []
    data_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]

    for fold in sorted(row_folds["fold"].unique()):
        test_ids = set(row_folds.loc[row_folds["fold"] == fold, "cv_row_id"].astype(int))
        train_df = df.loc[~df["cv_row_id"].isin(test_ids), data_cols].copy()
        test_df = df.loc[df["cv_row_id"].isin(test_ids), data_cols].copy()
        fold_meta = row_folds[row_folds["fold"] == fold].set_index("cv_row_id")

        preds, _models, _shap_inputs = train_and_predict(
            train_df=train_df,
            test_df=test_df,
            random_state=random_state,
            balance_classes=balance_classes,
        )
        pred_df = test_df[[SPECIES_COL, CLASS_COL, "Genus", "family", "order"]].copy()
        pred_df["scheme"] = scheme
        pred_df["fold"] = int(fold)
        pred_df["cv_row_id"] = df.loc[df["cv_row_id"].isin(test_ids), "cv_row_id"].to_numpy()
        pred_df["block_column"] = pred_df["cv_row_id"].map(fold_meta["block_column"])
        pred_df["block_id"] = pred_df["cv_row_id"].map(fold_meta["block_id"])
        pred_df["y_true"] = test_df[TARGET].to_numpy()
        for model in model_names:
            pred_df[model] = preds[model]
            fold_metrics = safe_evaluate(pred_df["y_true"].to_numpy(), pred_df[model].to_numpy())
            fold_metric_rows.append(
                {
                    "scheme": scheme,
                    "fold": int(fold),
                    "model": model,
                    "test_rows": len(pred_df),
                    **fold_metrics,
                }
            )
        prediction_frames.append(pred_df)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(scheme_dir / "cv_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(fold_metric_rows).to_csv(scheme_dir / "fold_metrics.csv", index=False, encoding="utf-8")

    summary, per_class, per_block = summarize_prediction_metrics(all_predictions, list(model_names))
    summary.to_csv(scheme_dir / "metric_summary.csv", index=False, encoding="utf-8")
    per_class.to_csv(scheme_dir / "per_class_metrics.csv", index=False, encoding="utf-8")
    per_block.to_csv(scheme_dir / "per_block_metrics.csv", index=False, encoding="utf-8")


def write_single_split(train_df: pd.DataFrame, test_df: pd.DataFrame, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    data_cols = [c for c in OUTPUT_COLUMNS if c in train_df.columns]
    train_df[data_cols].to_csv(split_dir / "train.csv", index=False, encoding="utf-8")
    test_df[data_cols].to_csv(split_dir / "test.csv", index=False, encoding="utf-8")


def save_group_predictions_and_metrics(
    pred_df: pd.DataFrame,
    model_names: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "benchmark_predictions_test.csv", index=False, encoding="utf-8")

    rows = []
    for model in model_names:
        rows.append({"model": model, **safe_evaluate(pred_df["y_true"], pred_df[model])})
    metrics_df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")
    return metrics_df


def run_leave_class_out(
    df: pd.DataFrame,
    split_dir: Path,
    results_dir: Path,
    random_state: int,
    root: Path,
    run_models: bool = True,
) -> pd.DataFrame:
    model_names, train_and_predict = load_model_api(root)
    data_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    summary_rows: list[dict[str, object]] = []

    for group_name, class_name in TARGET_CLASS_GROUPS.items():
        train_df = df[df[CLASS_COL] != class_name].copy()
        test_df = df[df[CLASS_COL] == class_name].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"Cannot build leave-class-out split for {group_name}={class_name}.")

        write_single_split(
            train_df=train_df,
            test_df=test_df,
            split_dir=split_dir / "leave_class_out" / group_name,
        )
        if not run_models:
            summary_rows.append(
                {
                    "group": group_name,
                    "held_out_class": class_name,
                    "model": "not_run",
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                }
            )
            continue

        preds, _models, _shap_inputs = train_and_predict(
            train_df=train_df[data_cols].copy(),
            test_df=test_df[data_cols].copy(),
            random_state=random_state,
            balance_classes=True,
        )

        pred_df = test_df[[SPECIES_COL, CLASS_COL, "Genus", "family", "order", "log_mass", "inv_kT"]].copy()
        pred_df["group"] = group_name
        pred_df["held_out_class"] = class_name
        pred_df["y_true"] = test_df[TARGET].to_numpy()
        for model in model_names:
            pred_df[model] = preds[model]

        group_out_dir = results_dir / "leave_class_out" / group_name
        metrics_df = save_group_predictions_and_metrics(pred_df, list(model_names), group_out_dir)
        for model in model_names:
            row = metrics_df[metrics_df["model"] == model].iloc[0]
            summary_rows.append(
                {
                    "group": group_name,
                    "held_out_class": class_name,
                    "model": model,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "r2": float(row["r2"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    out_dir = results_dir / "leave_class_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "benchmark_summary_groups.csv", index=False, encoding="utf-8")
    return summary_df


def run_fair_all(
    df: pd.DataFrame,
    split_dir: Path,
    results_dir: Path,
    n_splits: int,
    random_state: int,
    root: Path,
    run_models: bool = True,
) -> None:
    row_folds, block_assignments = make_scheme_folds(
        df=df,
        scheme="class_balanced_species",
        n_splits=n_splits,
        seed=random_state,
    )
    fold_summary = write_fold_files(
        df=df,
        row_folds=row_folds,
        split_dir=split_dir,
        scheme="fair_all",
        n_splits=n_splits,
    )
    fair_split_dir = split_dir / "fair_all"
    row_folds.to_csv(fair_split_dir / "row_fold_assignments.csv", index=False, encoding="utf-8")
    block_assignments.to_csv(fair_split_dir / "block_fold_assignments.csv", index=False, encoding="utf-8")
    fold_summary.to_csv(fair_split_dir / "fold_summary.csv", index=False, encoding="utf-8")

    fair_results_dir = results_dir / "fair_all"
    fair_results_dir.mkdir(parents=True, exist_ok=True)
    fold_summary.to_csv(fair_results_dir / "fold_summary.csv", index=False, encoding="utf-8")
    if not run_models:
        return

    run_residual_learning_cv(
        df=df,
        row_folds=row_folds.assign(scheme="fair_all"),
        results_dir=results_dir,
        scheme="fair_all",
        random_state=random_state,
        root=root,
        balance_classes=True,
    )


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate two BMR block-CV datasets: fair all-class CV and "
            "leave-class-out A/B/C predictions."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("data/merge_phylo.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/block_cv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/block_cv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Only write train/test CSVs; skip residual-learning model evaluation.",
    )
    parser.add_argument(
        "--legacy-extra-schemes",
        action="store_true",
        help="Also generate the broader species/genus/family/phylo exploratory schemes.",
    )
    args = parser.parse_args()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")

    input_path = resolve_path(root, args.input)
    split_dir = resolve_path(root, args.split_dir)
    results_dir = resolve_path(root, args.results_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = clean_input_data(input_path)

    run_fair_all(
        df=df,
        split_dir=split_dir,
        results_dir=results_dir,
        n_splits=args.n_splits,
        random_state=args.seed,
        root=root,
        run_models=not args.skip_models,
    )
    leave_class_summary = run_leave_class_out(
        df=df,
        split_dir=split_dir,
        results_dir=results_dir,
        random_state=args.seed,
        root=root,
        run_models=not args.skip_models,
    )

    if args.legacy_extra_schemes:
        legacy_df = add_phylo_clusters(df, n_blocks=20, seed=args.seed)
        all_fold_summaries: list[pd.DataFrame] = []
        all_row_assignments: list[pd.DataFrame] = []
        all_block_assignments: list[pd.DataFrame] = []
        legacy_schemes = ["species_out", "genus_out", "family_out", "phylo_cluster_out"]
        for scheme in legacy_schemes:
            row_folds, block_assignments = make_scheme_folds(
                df=legacy_df,
                scheme=scheme,
                n_splits=args.n_splits,
                seed=args.seed,
            )
            fold_summary = write_fold_files(
                df=legacy_df,
                row_folds=row_folds,
                split_dir=split_dir / "legacy_extra",
                scheme=scheme,
                n_splits=args.n_splits,
            )
            scheme_split_dir = split_dir / "legacy_extra" / scheme
            row_folds.to_csv(scheme_split_dir / "row_fold_assignments.csv", index=False, encoding="utf-8")
            block_assignments.to_csv(scheme_split_dir / "block_fold_assignments.csv", index=False, encoding="utf-8")
            fold_summary.to_csv(scheme_split_dir / "fold_summary.csv", index=False, encoding="utf-8")
            all_fold_summaries.append(fold_summary)
            all_row_assignments.append(row_folds)
            all_block_assignments.append(block_assignments.assign(scheme=scheme))

        legacy_results_dir = results_dir / "legacy_extra"
        legacy_results_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(all_fold_summaries, ignore_index=True).to_csv(
            legacy_results_dir / "fold_summary_all_schemes.csv",
            index=False,
            encoding="utf-8",
        )
        pd.concat(all_row_assignments, ignore_index=True).to_csv(
            legacy_results_dir / "row_fold_assignments_all_schemes.csv",
            index=False,
            encoding="utf-8",
        )
        pd.concat(all_block_assignments, ignore_index=True).to_csv(
            legacy_results_dir / "block_fold_assignments_all_schemes.csv",
            index=False,
            encoding="utf-8",
        )

    manifest_rows = [
        {
            "dataset": "fair_all",
            "purpose": "All classes appear in training and testing; species blocks and class-balanced sample weights reduce class-size dominance.",
            "split_path": str(split_dir / "fair_all"),
            "result_path": str(results_dir / "fair_all"),
        },
        {
            "dataset": "leave_class_out",
            "purpose": "A/B/C target classes are completely absent from training and predicted as held-out classes.",
            "split_path": str(split_dir / "leave_class_out"),
            "result_path": str(results_dir / "leave_class_out"),
        },
    ]
    pd.DataFrame(manifest_rows).to_csv(results_dir / "block_cv_manifest.csv", index=False, encoding="utf-8")
    leave_class_summary.to_csv(results_dir / "leave_class_out_summary.csv", index=False, encoding="utf-8")

    print(f"Saved intended block-CV splits under: {split_dir}")
    print(f"Saved intended block-CV results under: {results_dir}")
    print("Primary outputs:")
    print(f"- {results_dir / 'fair_all'}")
    print(f"- {results_dir / 'leave_class_out'}")


if __name__ == "__main__":
    main()
