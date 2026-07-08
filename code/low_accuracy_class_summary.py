#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def fold_accuracy(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    y_true_arr = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    y_pred_arr = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr) & (y_true_arr > 0) & (y_pred_arr > 0)

    out = np.full(len(y_true_arr), np.nan, dtype=float)
    ratio = y_pred_arr[valid] / y_true_arr[valid]
    out[valid] = np.minimum(ratio, 1.0 / ratio)
    return np.clip(out, 0.0, 1.0)


def taxon_to_tip_label(taxon_name: str) -> str:
    return str(taxon_name).strip().replace(" ", "_")


def build_summary(predictions_path: Path, tree_path: Path, threshold: float) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_path)
    required = ["taxon_name", "class", "y_true", "xgboost"]
    missing = [col for col in required if col not in pred_df.columns]
    if missing:
        raise KeyError(f"{predictions_path.name} missing required columns: {', '.join(missing)}")

    pred_df = pred_df[required].copy()
    pred_df["tip_label"] = pred_df["taxon_name"].map(taxon_to_tip_label)
    pred_df["accuracy"] = fold_accuracy(pred_df["y_true"], pred_df["xgboost"])
    pred_df = pred_df.dropna(subset=["tip_label", "class", "accuracy"]).drop_duplicates(
        "tip_label",
        keep="last",
    )

    tree = Phylo.read(str(tree_path), "newick")
    tree_tips = {clade.name for clade in tree.get_terminals()}
    pred_df = pred_df[pred_df["tip_label"].isin(tree_tips)].copy()
    if pred_df.empty:
        raise ValueError("No prediction rows matched tree tip labels.")

    low_df = pred_df[pred_df["accuracy"] < threshold].copy()
    summary = pred_df.groupby("class").agg(total_species=("tip_label", "nunique"))
    low_summary = low_df.groupby("class").agg(low_species=("tip_label", "nunique"))
    summary = summary.join(low_summary)
    summary["low_species"] = summary["low_species"].fillna(0).astype(int)
    summary["low_ratio"] = summary["low_species"] / summary["total_species"]
    summary = summary.reset_index().sort_values(
        ["low_ratio", "low_species", "class"],
        ascending=[False, False, True],
    )
    return summary


def write_text_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        f"{row['class']:<22} {int(row['low_species']):>3}/{int(row['total_species']):<3} {row['low_ratio'] * 100:>7.1f}%"
        for _, row in summary.iterrows()
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description="Summarize low-accuracy XGB residual-learning species counts by class."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/benchmark/all/benchmark_predictions_test.csv"),
        help="Prediction CSV with taxon_name, class, y_true, and xgboost columns.",
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path("data/phylogeny/unique_taxon_names.nwk"),
        help="Newick tree used to keep only tree-matched test species.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Low-accuracy threshold. Default: 0.40.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/explore/low_accuracy_class_summary.txt"),
        help="Output text file.",
    )
    args = parser.parse_args()

    predictions_path = resolve_path(root, args.predictions)
    tree_path = resolve_path(root, args.tree)
    output_path = resolve_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(predictions_path, tree_path, args.threshold)
    write_text_summary(summary, output_path)
    print(f"Saved low-accuracy class summary: {output_path}")


if __name__ == "__main__":
    main()
