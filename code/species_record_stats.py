#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def main() -> None:
    root = find_root()
    csv_path = root / "data" / "merged_bmr_mass_temperature.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    species_col = "Species"
    if species_col not in df.columns:
        raise KeyError(f"Missing required column: {species_col}")

    valid_species = df[species_col].astype("string").str.strip()
    valid_species = valid_species[valid_species.notna() & (valid_species != "")]

    unique_species = valid_species.nunique()
    total_records = len(df)
    avg_records_per_species = total_records / unique_species if unique_species > 0 else 0.0

    print(f"file: {csv_path.relative_to(root)}")
    print(f"total_records: {total_records}")
    print(f"unique_species: {unique_species}")
    print(f"avg_records_per_species: {avg_records_per_species:.6f}")


if __name__ == "__main__":
    main()
