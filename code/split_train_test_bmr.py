#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CLASS_HOLDOUT_SPLITS = {
    "A": "Teleostei",
    "B": "Mammalia",
    "C": "Insecta",
}
OUTPUT_COLUMNS = [
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
            "Generate A/B/C train-test splits by class holdout: "
            "A=Teleostei, B=Mammalia, C=Insecta."
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
        default=Path("data/splits/class_holdout"),
        help="Output directory containing A/B/C subfolders.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df[OUTPUT_COLUMNS].copy()
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
    out = out.dropna(subset=OUTPUT_COLUMNS).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out["BMR"] > 0)].copy()
    out = out[out["taxon_name"] != ""].copy()
    out = out[out["class"] != ""].copy()
    out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid rows left after filtering required columns.")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows: list[dict[str, str | int | float]] = []
    for split_name, test_class in CLASS_HOLDOUT_SPLITS.items():
        test_df = out[out["class"] == test_class].copy()
        train_df = out[out["class"] != test_class].copy()

        if test_df.empty:
            raise ValueError(f"Split {split_name} has empty test set for class={test_class}.")
        if train_df.empty:
            raise ValueError(f"Split {split_name} has empty train set for class={test_class}.")
        if set(train_df.index).intersection(set(test_df.index)):
            raise RuntimeError(f"Split {split_name} leakage detected by overlapping indices.")

        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        split_rows.append(
            {
                "split": split_name,
                "test_class": test_class,
                "rows_total": len(out),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_species": int(train_df["taxon_name"].nunique()),
                "test_species": int(test_df["taxon_name"].nunique()),
            }
        )

        print(f"[Split {split_name}] test_class={test_class}")
        print(f"  Saved train: {train_path}")
        print(f"  Saved test: {test_path}")
        print(f"  Train rows: {len(train_df)}")
        print(f"  Test rows: {len(test_df)}")

    summary_df = pd.DataFrame(split_rows)
    summary_path = out_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved split summary: {summary_path}")


if __name__ == "__main__":
    main()
