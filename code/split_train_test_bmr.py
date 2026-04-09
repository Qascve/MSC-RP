#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def allocate_test_counts(
    group_sizes: pd.Series, test_ratio: float, rng: np.random.Generator
) -> pd.Series:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1).")

    mandatory = pd.Series(1, index=group_sizes.index, dtype=int)
    total_rows = int(group_sizes.sum())
    species_count = int(len(group_sizes))
    desired_total_test = int(round(total_rows * test_ratio))
    desired_total_test = max(desired_total_test, species_count)
    desired_total_test = min(desired_total_test, total_rows)

    capacities = (group_sizes - 1).clip(lower=0).astype(int)
    extra_needed = desired_total_test - species_count
    extras = pd.Series(0, index=group_sizes.index, dtype=int)

    if extra_needed > 0:
        total_capacity = int(capacities.sum())
        if total_capacity == 0:
            # All species have exactly one row, test set must include all rows.
            return group_sizes.astype(int)

        extra_needed = min(extra_needed, total_capacity)
        raw = capacities / total_capacity * extra_needed
        base = np.floor(raw).astype(int)
        extras += base
        remainder = int(extra_needed - int(base.sum()))

        if remainder > 0:
            frac = (raw - base).to_numpy()
            # Add tiny random noise to break ties reproducibly.
            frac = frac + rng.uniform(0.0, 1e-9, size=len(frac))
            order = np.argsort(-frac)
            idx = capacities.index.to_numpy()
            given = 0
            for pos in order:
                name = idx[pos]
                if extras[name] < capacities[name]:
                    extras[name] += 1
                    given += 1
                    if given == remainder:
                        break

    test_counts = mandatory + extras
    test_counts = np.minimum(test_counts.to_numpy(), group_sizes.to_numpy())
    return pd.Series(test_counts, index=group_sizes.index, dtype=int)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Split merge_phylo.csv into train/test with 7:3 ratio, "
            "ensuring each species has at least one sample in test."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/merge_phylo.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/train/train.csv"),
        help="Output train CSV path.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/test/test.csv"),
        help="Output test CSV path.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.3,
        help="Test ratio (default: 0.3).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    train_path = (
        args.train_output if args.train_output.is_absolute() else root / args.train_output
    )
    test_path = args.test_output if args.test_output.is_absolute() else root / args.test_output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = [
        "taxon_name",
        "wet_Mass_kg",
        "temperature",
        "pc1",
        "pc2",
        "pc3",
        "pc4",
        "pc5",
        "BMR",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df[required_cols].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    numeric_cols = ["wet_Mass_kg", "temperature", "pc1", "pc2", "pc3", "pc4", "pc5", "BMR"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep only valid rows and avoid fabricated data by strict filtering.
    out = out.dropna(subset=required_cols).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out["BMR"] > 0)].copy()
    out = out[out["taxon_name"] != ""].copy()
    out = out.reset_index(drop=True)
    out["row_id"] = out.index

    if out.empty:
        raise ValueError("No valid rows left after filtering required columns.")

    rng = np.random.default_rng(args.seed)
    group_sizes = out.groupby("taxon_name")["row_id"].size().sort_index()
    test_counts = allocate_test_counts(group_sizes, args.test_ratio, rng)

    test_ids: list[int] = []
    for taxon, group in out.groupby("taxon_name", sort=True):
        n_test = int(test_counts.loc[taxon])
        chosen = rng.choice(group["row_id"].to_numpy(), size=n_test, replace=False)
        test_ids.extend(chosen.tolist())

    test_id_set = set(test_ids)
    test_df = out[out["row_id"].isin(test_id_set)].copy()
    train_df = out[~out["row_id"].isin(test_id_set)].copy()

    # Safety checks for leakage and species-coverage constraints.
    if set(train_df["row_id"]).intersection(set(test_df["row_id"])):
        raise RuntimeError("Leakage detected: overlapping row_id between train/test.")
    missing_species_in_test = set(group_sizes.index) - set(test_df["taxon_name"].unique())
    if missing_species_in_test:
        preview = ", ".join(list(sorted(missing_species_in_test))[:5])
        raise RuntimeError(f"Species missing in test split: {preview}")

    train_df = train_df.drop(columns=["row_id"])
    test_df = test_df.drop(columns=["row_id"])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"Saved train: {train_path}")
    print(f"Saved test: {test_path}")
    print(f"Rows total: {len(out)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train ratio: {len(train_df) / len(out):.4f}")
    print(f"Test ratio: {len(test_df) / len(out):.4f}")
    print(f"Species total: {len(group_sizes)}")
    print(f"Species in test: {test_df['taxon_name'].nunique()}")


if __name__ == "__main__":
    main()
