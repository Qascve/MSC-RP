#!/usr/bin/env python3
# Create a fixed species-blocked 4-fold development partition plus held-out test.
#
# This is a species-blocked holdout design (plus reusable development folds), not
# a claim that the single 20% test bucket alone is "cross-validation".
#
# Per taxonomic class, each species (taxon_name) is assigned wholly to one of:
# F1, F2, F3, F4, T (target ~20% each).
#
# Stratification details:
# - Within each class, species are shuffled (seeded) then placed into buckets
# with per-class quotas from _allocate_counts.
# - Remainder slots prefer currently lightest buckets (by global row totals).
# - Classes with n_species < 5 cannot appear in every bucket; some folds/test
# may lack that class. With n_species >= 5, all five buckets receive >=1 species.
#
# Fold usage:
# Fold i (HP/CV): train = other three development folds (~60%), eval = Fi (~20%)
# Test holdout:   train = F1∪F2∪F3∪F4 (~80%), eval = T (~20%)
#
# Also writes class_weights.csv with row shares. Training sample weights used by
# residual RF/XGB and explore M3-L/M4-L follow sklearn balanced:
# w_c = n / (n_classes * n_c).

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
    out["log_mass"] = np.log10(out["wet_Mass_kg"].to_numpy())
    out["log_BMR"] = np.log10(out["BMR"].to_numpy())
    out["inv_kT"] = 1.0 / (K_BOLTZMANN_EV_PER_K * temp_k.to_numpy())
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=OUTPUT_COLUMNS).copy()
    return out.reset_index(drop=True)


def _allocate_counts(n: int, n_buckets: int = 5) -> list[int]:
    # Allocate species as evenly as possible across fixed buckets.
    if n < 1:
        raise ValueError("Need at least one species.")
    base, remainder = divmod(n, n_buckets)
    counts = [base + int(i < remainder) for i in range(n_buckets)]
    if sum(counts) != n:
        raise RuntimeError(f"Invalid {n_buckets}-bucket allocation for n={n}: {counts}")
    return counts


def assign_species_to_buckets(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[list[set[str]], set[str], pd.DataFrame]:
    #     Per class, assign each species wholly to F1/F2/F3/F4/T.
    # Returns (four development-fold species sets, test species set, summary).
    #
    rng = np.random.default_rng(random_state)
    cv_folds: list[set[str]] = [set() for _ in range(4)]
    holdout: set[str] = set()
    bucket_row_totals = np.zeros(5, dtype=int)
    summary_rows: list[dict[str, object]] = []

    for class_name, class_df in df.groupby("class", sort=True):
        species_names = (
            class_df.groupby("taxon_name").size().sort_values(ascending=False).index.astype(str).to_numpy()
        )
        n_species = len(species_names)
        counts = _allocate_counts(n_species)
        # Assign any remainder slots to the currently lightest buckets instead
        # of always favoring the same fold, especially for classes with <5 species.
        base_count = min(counts)
        remainder = n_species - base_count * 5
        counts = [base_count] * 5
        bucket_priority = sorted(range(5), key=lambda i: (bucket_row_totals[i], i))
        for bucket_idx in bucket_priority[:remainder]:
            counts[bucket_idx] += 1
        species_rows = class_df.groupby("taxon_name").size().to_dict()
        shuffled = species_names.copy()
        rng.shuffle(shuffled)
        # Place large species first into the currently lightest eligible bucket.
        # Per-class quotas keep taxonomic representation balanced while global
        # row totals make all five buckets close to 20% of the full dataset.
        ordered = sorted(
            shuffled.tolist(),
            key=lambda name: int(species_rows[name]),
            reverse=True,
        )
        picked: list[set[str]] = [set() for _ in range(5)]
        remaining = counts.copy()
        for species_name in ordered:
            eligible = [i for i, slots in enumerate(remaining) if slots > 0]
            bucket_idx = min(
                eligible,
                key=lambda i: (bucket_row_totals[i], -remaining[i], i),
            )
            picked[bucket_idx].add(species_name)
            remaining[bucket_idx] -= 1
            bucket_row_totals[bucket_idx] += int(species_rows[species_name])
        if any(remaining):
            raise RuntimeError(
                f"Unfilled bucket quota in class {class_name}: {remaining}"
            )
        picked_folds = picked[:4]
        picked_t = picked[4]

        for fold_set, class_fold_set in zip(cv_folds, picked_folds):
            fold_set.update(class_fold_set)
        holdout.update(picked_t)

        fold_masks = [
            class_df["taxon_name"].astype(str).isin(class_fold_set)
            for class_fold_set in picked_folds
        ]
        mask_t = class_df["taxon_name"].astype(str).isin(picked_t)
        summary_rows.append(
            {
                "class": str(class_name),
                "species_total": n_species,
                **{
                    f"species_f{i}": len(picked_folds[i - 1])
                    for i in range(1, 5)
                },
                "species_test": len(picked_t),
                "rows_total": len(class_df),
                **{
                    f"rows_f{i}": int(fold_masks[i - 1].sum())
                    for i in range(1, 5)
                },
                "rows_test": int(mask_t.sum()),
                "included": True,
            }
        )

    if any(not fold_set for fold_set in cv_folds) or not holdout:
        raise RuntimeError("Species-block split failed: one of F1/F2/F3/F4/T is empty.")

    summary = pd.DataFrame(summary_rows).sort_values("rows_total", ascending=False)
    return cv_folds, holdout, summary


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
            "Create fixed species-block F1/F2/F3/F4/T buckets (~20% each): "
            "each CV split trains on three development folds and validates on one; "
            "test trains on all four development folds."
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
        help="Output directory for fold1..fold4/test CSVs and class weights.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path)
    out = prepare_modeling_frame(raw)
    if out.empty:
        raise ValueError("No valid rows left after filtering required columns.")

    # Classes with fewer than 7 species are removed earlier in
    # filter_target_classes.py (after whitelist, before phylogeny PCs).

    cv_species, holdout, summary = assign_species_to_buckets(out, random_state=args.seed)
    included_classes = set(summary.loc[summary["included"], "class"].astype(str))
    out = out[out["class"].astype(str).isin(included_classes)].copy().reset_index(drop=True)

    taxon = out["taxon_name"].astype(str)
    cv_masks = [taxon.isin(fold_set) for fold_set in cv_species]
    is_t = taxon.isin(holdout)
    assigned_count = sum(mask.astype(int) for mask in [*cv_masks, is_t])
    if not bool((assigned_count == 1).all()):
        raise RuntimeError("Rows must be assigned exactly once across F1/F2/F3/F4/T.")

    cv_parts = [out[mask].copy() for mask in cv_masks]
    part_t = out[is_t].copy()
    final_train = pd.concat(cv_parts, ignore_index=True)
    final_test = part_t

    assert_no_species_leakage(final_train, final_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, eval_part in enumerate(cv_parts):
        train_parts = [part for i, part in enumerate(cv_parts) if i != fold_idx]
        fold_train = pd.concat(train_parts, ignore_index=True)
        assert_no_species_leakage(fold_train, eval_part)
        assert_no_species_leakage(fold_train, final_test)
        _write_pair(out_dir / f"fold{fold_idx + 1}", fold_train, eval_part)
    _write_pair(out_dir / "test", final_train, final_test)

    # Compatibility alias: top-level train.csv mirrors the 80% train used for held-out test.
    # Do not write top-level test.csv — it would conflict with the test/ directory name.
    final_train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")

    summary_path = out_dir / "class_species_block_split_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    weights_path = out_dir / "class_weights.csv"
    weight_df = write_class_weights_csv(out, weights_path)

    n_sp = out["taxon_name"].nunique()
    for fold_idx in range(4):
        print(f"Saved fold{fold_idx + 1} train/eval: {out_dir / f'fold{fold_idx + 1}'}")
    print(f"Saved test train/test (80/20): {out_dir / 'test'}")
    print(f"Saved compatibility train.csv (F1∪F2∪F3∪F4): {out_dir / 'train.csv'}")
    print(f"Saved split summary: {summary_path}")
    print(f"Saved class weights: {weights_path}")
    print(f"Rows total: {len(out)} | species total: {n_sp}")
    print(
        "Species F1/F2/F3/F4/T: "
        + " / ".join(str(len(s)) for s in [*cv_species, holdout])
        + " ("
        + " / ".join(f"{len(s)/n_sp:.1%}" for s in [*cv_species, holdout])
        + ")"
    )
    for fold_idx, eval_part in enumerate(cv_parts):
        print(
            f"Fold{fold_idx + 1} train/eval rows: "
            f"{len(final_train) - len(eval_part)} / {len(eval_part)}"
        )
    print(f"Test  train/eval rows: {len(final_train)} / {len(final_test)}")
    print(f"Classes: {out['class'].nunique()}")
    print("Class weights (top):")
    print(weight_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
