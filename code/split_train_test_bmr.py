#!/usr/bin/env python3
"""
Create a species-block 40-40-20 split.

Per class, each species (taxon_name) is assigned wholly to one of:
  A (~40%), B (~40%), T (~20% held-out test).

Fold design (no species leakage):
  Fold 1: train = A, eval = B
  Fold 2: train = B, eval = A
  Test:   train = A ∪ B (80%), eval = T (20%)

Also writes class weights: class_rows / total_rows.
"""

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


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
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
    return out.reset_index(drop=True)


def _allocate_counts(n: int, fold_frac: float, test_frac: float) -> tuple[int, int, int]:
    """
    Allocate n species into A / B / T counts targeting fold_frac, fold_frac, test_frac.
    Guarantees each bucket has at least 1 when n >= 3.
    """
    if n < 3:
        raise ValueError("Need at least 3 species to form A/B/T buckets.")

    n_test = int(round(n * test_frac))
    n_test = max(1, min(n - 2, n_test))
    n_remain = n - n_test
    n_b = int(round(n_remain * 0.5))
    n_b = max(1, min(n_remain - 1, n_b))
    n_a = n_remain - n_b
    if n_a < 1 or n_b < 1 or n_test < 1:
        raise RuntimeError(f"Invalid allocation for n={n}: a={n_a}, b={n_b}, t={n_test}")
    if n_a + n_b + n_test != n:
        raise RuntimeError(f"Allocation sum mismatch for n={n}")
    return n_a, n_b, n_test


def assign_species_to_buckets(
    df: pd.DataFrame,
    fold_frac: float,
    test_frac: float,
    random_state: int,
) -> tuple[set[str], set[str], set[str], pd.DataFrame]:
    """
    Per class, assign each species wholly to A, B, or T.
    Returns (species_a, species_b, species_t, summary_df).
    """
    if not 0 < fold_frac < 1 or not 0 < test_frac < 1:
        raise ValueError("fold_frac and test_frac must be in (0, 1).")
    if abs(2 * fold_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Expected 2 * fold_frac + test_frac == 1 (e.g. 0.4 + 0.4 + 0.2).")

    rng = np.random.default_rng(random_state)
    half_a: set[str] = set()
    half_b: set[str] = set()
    holdout: set[str] = set()
    summary_rows: list[dict[str, object]] = []

    for class_name, class_df in df.groupby("class", sort=True):
        species_names = (
            class_df.groupby("taxon_name").size().sort_values(ascending=False).index.astype(str).to_numpy()
        )
        n_species = len(species_names)
        if n_species < 3:
            summary_rows.append(
                {
                    "class": str(class_name),
                    "species_total": n_species,
                    "species_a": 0,
                    "species_b": 0,
                    "species_test": 0,
                    "rows_total": len(class_df),
                    "rows_a": 0,
                    "rows_b": 0,
                    "rows_test": 0,
                    "included": False,
                }
            )
            continue

        n_a, n_b, n_t = _allocate_counts(n_species, fold_frac, test_frac)
        shuffled = species_names.copy()
        rng.shuffle(shuffled)
        picked_t = set(shuffled[:n_t].tolist())
        picked_b = set(shuffled[n_t : n_t + n_b].tolist())
        picked_a = set(shuffled[n_t + n_b :].tolist())
        if len(picked_a) != n_a:
            raise RuntimeError(f"A-count mismatch in class {class_name}")

        half_a.update(picked_a)
        half_b.update(picked_b)
        holdout.update(picked_t)

        mask_a = class_df["taxon_name"].astype(str).isin(picked_a)
        mask_b = class_df["taxon_name"].astype(str).isin(picked_b)
        mask_t = class_df["taxon_name"].astype(str).isin(picked_t)
        summary_rows.append(
            {
                "class": str(class_name),
                "species_total": n_species,
                "species_a": len(picked_a),
                "species_b": len(picked_b),
                "species_test": len(picked_t),
                "rows_total": len(class_df),
                "rows_a": int(mask_a.sum()),
                "rows_b": int(mask_b.sum()),
                "rows_test": int(mask_t.sum()),
                "included": True,
            }
        )

    if not half_a or not half_b or not holdout:
        raise RuntimeError("Species-block split failed: one of A/B/T is empty.")

    summary = pd.DataFrame(summary_rows).sort_values("rows_total", ascending=False)
    return half_a, half_b, holdout, summary


def assert_no_species_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    leaked = set(train_df["taxon_name"].astype(str)).intersection(
        set(test_df["taxon_name"].astype(str))
    )
    if leaked:
        raise RuntimeError(f"Species leakage detected: {sorted(leaked)[:5]}")


def write_class_weights_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    total_rows = len(df)
    counts = df["class"].value_counts(dropna=False)
    weight_df = pd.DataFrame(
        {
            "class": counts.index.astype(str),
            "rows": counts.to_numpy(dtype=int),
            "weight": (counts.to_numpy(dtype=float) / total_rows) if total_rows else 0.0,
        }
    ).sort_values("weight", ascending=False)
    weight_df.to_csv(path, index=False, encoding="utf-8")
    return weight_df


def _write_pair(directory: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(directory / "train.csv", index=False, encoding="utf-8")
    test_df.to_csv(directory / "test.csv", index=False, encoding="utf-8")


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Create species-block 40-40-20 splits: "
            "fold1 (train A / eval B), fold2 (train B / eval A), "
            "test (train A∪B / eval T)."
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
        default=Path("data/splits"),
        help="Output directory for fold1/fold2/test CSVs and class weights.",
    )
    parser.add_argument(
        "--fold-frac",
        type=float,
        default=0.4,
        help="Target fraction of species for each CV half A and B (default: 0.4).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Target fraction of species for held-out test T (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if abs(2 * args.fold_frac + args.test_frac - 1.0) > 1e-9:
        raise ValueError("--fold-frac * 2 + --test-frac must equal 1.0")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    out = prepare_modeling_frame(raw)
    if out.empty:
        raise ValueError("No valid rows left after filtering required columns.")

    # Need ≥3 species/class for non-empty A/B/T.
    species_per_class = out.groupby("class")["taxon_name"].nunique()
    drop_classes = species_per_class[species_per_class < 3].index.tolist()
    if drop_classes:
        out = out[~out["class"].isin(drop_classes)].copy().reset_index(drop=True)
        print(
            "Dropped classes with fewer than 3 species: "
            + ", ".join(str(c) for c in sorted(drop_classes))
        )
    if out.empty:
        raise ValueError("No rows left after dropping classes with <3 species.")

    half_a, half_b, holdout, summary = assign_species_to_buckets(
        out,
        fold_frac=args.fold_frac,
        test_frac=args.test_frac,
        random_state=args.seed,
    )
    included_classes = set(summary.loc[summary["included"], "class"].astype(str))
    out = out[out["class"].astype(str).isin(included_classes)].copy().reset_index(drop=True)

    taxon = out["taxon_name"].astype(str)
    is_a = taxon.isin(half_a)
    is_b = taxon.isin(half_b)
    is_t = taxon.isin(holdout)
    if int((is_a | is_b | is_t).sum()) != len(out):
        raise RuntimeError("Some rows were not assigned to A/B/T.")
    if int((is_a & is_b).sum()) or int((is_a & is_t).sum()) or int((is_b & is_t).sum()):
        raise RuntimeError("Overlapping A/B/T assignments detected.")

    part_a = out[is_a].copy()
    part_b = out[is_b].copy()
    part_t = out[is_t].copy()

    fold1_train = part_a
    fold1_test = part_b
    fold2_train = part_b
    fold2_test = part_a
    final_train = pd.concat([part_a, part_b], ignore_index=True)
    final_test = part_t

    assert_no_species_leakage(fold1_train, fold1_test)
    assert_no_species_leakage(fold2_train, fold2_test)
    assert_no_species_leakage(final_train, final_test)
    assert_no_species_leakage(fold1_train, final_test)
    assert_no_species_leakage(fold1_test, final_test)

    if set(fold1_train["taxon_name"].astype(str)) != set(fold2_test["taxon_name"].astype(str)):
        raise RuntimeError("Fold complementarity broken: fold1 train != fold2 test.")
    if set(fold1_test["taxon_name"].astype(str)) != set(fold2_train["taxon_name"].astype(str)):
        raise RuntimeError("Fold complementarity broken: fold1 test != fold2 train.")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_pair(out_dir / "fold1", fold1_train, fold1_test)
    _write_pair(out_dir / "fold2", fold2_train, fold2_test)
    _write_pair(out_dir / "test", final_train, final_test)

    # Compatibility alias: top-level train.csv mirrors the 80% train used for held-out test.
    # Do not write top-level test.csv — it would conflict with the test/ directory name.
    final_train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")

    summary_path = out_dir / "class_species_block_split_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    weights_path = out_dir / "class_weights.csv"
    weight_df = write_class_weights_csv(out, weights_path)

    n_sp = out["taxon_name"].nunique()
    print(f"Saved fold1 train/test: {out_dir / 'fold1'}")
    print(f"Saved fold2 train/test: {out_dir / 'fold2'}")
    print(f"Saved test train/test (80/20): {out_dir / 'test'}")
    print(f"Saved compatibility train.csv (A∪B): {out_dir / 'train.csv'}")
    print(f"Saved split summary: {summary_path}")
    print(f"Saved class weights: {weights_path}")
    print(f"Rows total: {len(out)} | species total: {n_sp}")
    print(
        f"Species A/B/T: {len(half_a)} / {len(half_b)} / {len(holdout)} "
        f"({len(half_a)/n_sp:.1%} / {len(half_b)/n_sp:.1%} / {len(holdout)/n_sp:.1%})"
    )
    print(f"Fold1 train/eval rows: {len(fold1_train)} / {len(fold1_test)}")
    print(f"Fold2 train/eval rows: {len(fold2_train)} / {len(fold2_test)}")
    print(f"Test  train/eval rows: {len(final_train)} / {len(final_test)}")
    print(f"Classes: {out['class'].nunique()}")
    print("Class weights (top):")
    print(weight_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
