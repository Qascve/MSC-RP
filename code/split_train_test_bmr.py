#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

K_BOLTZMANN_EV_PER_K = 8.617e-5

BASE_COLUMNS = [
    "class",
    "order",
    "family",
    "Genus",
    "species",
    "wet_Mass_g",
    "wet_Mass_kg",
    "BMR",
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

DERIVED_COLUMNS = [
    "log_mass",
    "log_BMR",
    "inv_kT",
]

OUTPUT_COLUMNS = [*BASE_COLUMNS, *DERIVED_COLUMNS]


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate train-test split "
            "Each class contributes samples to both train and test when possible."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/merge_phylo.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits/stratified"),
        help="Output directory containing train/test CSV files.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.3,
        help="Per-class test ratio (default: 0.3).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be in (0, 1).")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df[BASE_COLUMNS].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    out["class"] = out["class"].astype("string").str.strip()
    numeric_cols = [
        "wet_Mass_g",
        "wet_Mass_kg",
        "BMR",
        "temperature",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep only valid rows and avoid fabricated data by strict filtering.
    out = out.dropna(subset=BASE_COLUMNS).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out["BMR"] > 0)].copy()
    out = out[out["taxon_name"] != ""].copy()
    out = out[out["class"] != ""].copy()

    out = out[(out["temperature"] + 273.15) > 0].copy()
    temp_k = out["temperature"] + 273.15
    out["log_mass"] = np.log(out["wet_Mass_kg"].to_numpy())
    out["log_BMR"] = np.log(out["BMR"].to_numpy())
    out["inv_kT"] = 1.0 / (K_BOLTZMANN_EV_PER_K * temp_k.to_numpy())
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=OUTPUT_COLUMNS).copy()

    out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid rows left after filtering required columns.")

    class_counts = out["class"].value_counts(dropna=False)
    singleton_classes = class_counts[class_counts == 1].index.tolist()
    if singleton_classes:
        out = out[~out["class"].isin(singleton_classes)].copy()
        out = out.reset_index(drop=True)
        print(
            "Dropped singleton classes (rows_total=1): "
            + ", ".join(str(c) for c in sorted(singleton_classes))
        )

    if out.empty:
        raise ValueError("No rows left after dropping singleton classes.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out.reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=int)
    rng = np.random.default_rng(args.seed)

    test_ids: list[int] = []
    class_rows: list[dict[str, str | int | float]] = []
    for class_name, group in out.groupby("class", sort=True):
        group_ids = group["row_id"].to_numpy()
        n = len(group_ids)
        n_test = int(round(n * args.test_ratio))
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0
        if n_test > 0:
            picked = rng.choice(group_ids, size=n_test, replace=False)
            test_ids.extend(picked.tolist())
        class_rows.append(
            {
                "class": str(class_name),
                "rows_total": n,
                "rows_train": n - n_test,
                "rows_test": n_test,
                "test_ratio_actual": (n_test / n) if n > 0 else 0.0,
            }
        )

    test_id_set = set(test_ids)
    test_df = out[out["row_id"].isin(test_id_set)].copy()
    train_df = out[~out["row_id"].isin(test_id_set)].copy()

    if test_df.empty or train_df.empty:
        raise RuntimeError("Split failed: empty train or test set.")
    if set(train_df["row_id"]).intersection(set(test_df["row_id"])):
        raise RuntimeError("Leakage detected: overlapping row_id between train/test.")

    train_df = train_df.drop(columns=["row_id"])
    test_df = test_df.drop(columns=["row_id"])

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    class_summary = pd.DataFrame(class_rows).sort_values("rows_total", ascending=False)
    class_summary_path = out_dir / "class_split_summary.csv"
    class_summary.to_csv(class_summary_path, index=False, encoding="utf-8")

    print(f"Saved train: {train_path}")
    print(f"Saved test: {test_path}")
    print(f"Saved class summary: {class_summary_path}")
    print(f"Rows total: {len(out)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Classes total: {out['class'].nunique()}")
    print(f"Classes in train: {train_df['class'].nunique()}")
    print(f"Classes in test: {test_df['class'].nunique()}")


if __name__ == "__main__":
    main()
