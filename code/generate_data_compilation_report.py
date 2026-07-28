#!/usr/bin/env python3
"""
Generate a compact data-flow table for §5.1 with three stages only:

1. After merging the three sources and dropping incomplete / duplicate rows
2. After class whitelist + dropping classes with <7 species
3. After phylogeny embedding join + dropping classes with <7 species again

Outputs: results/data_compilation/data_flow_table.csv
Columns: stage, unique_species, classes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def binomial_series(df: pd.DataFrame) -> pd.Series:
    if "taxon_name" in df.columns:
        names = df["taxon_name"].astype("string").str.strip()
        if names.notna().any() and (names != "").any():
            return names

    genus = df["Genus"].astype("string").str.strip() if "Genus" in df.columns else ""
    species = df["species"].astype("string").str.strip() if "species" in df.columns else ""
    return (genus.fillna("") + " " + species.fillna("")).str.strip().replace({"": pd.NA})


def stage_counts(df: pd.DataFrame) -> dict[str, int]:
    species = binomial_series(df).dropna()
    species = species[species != ""]
    if "class" in df.columns:
        classes = (
            df["class"]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
            .dropna()
        )
        n_classes = int(classes.nunique())
    else:
        n_classes = 0
    return {
        "unique_species": int(species.nunique()),
        "classes": n_classes,
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_data_flow(root: Path) -> pd.DataFrame:
    pipeline = [
        (
            "after_merge_dedup_dropna",
            root / "data" / "cleaning" / "merged_bmr_mass_temperature.csv",
        ),
        (
            "after_drop_classes_lt7_species",
            root / "data" / "cleaning" / "filtered_data.csv",
        ),
        (
            "after_phylogeny_join_drop_classes_lt7_species",
            root / "data" / "merge_phylo.csv",
        ),
    ]

    rows: list[dict[str, object]] = []
    for stage, path in pipeline:
        counts = stage_counts(load_csv(path))
        rows.append(
            {
                "stage": stage,
                "unique_species": counts["unique_species"],
                "classes": counts["classes"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_compilation"),
        help="Directory for CSV output",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    flow = build_data_flow(root)
    csv_path = output_dir / "data_flow_table.csv"
    flow.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(flow.to_string(index=False))
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
