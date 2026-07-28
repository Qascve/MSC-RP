#!/usr/bin/env python3
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


def load_embeddings(path: Path) -> pd.DataFrame:
    emb = pd.read_csv(path)
    if "taxon_name" not in emb.columns:
        raise KeyError("Embeddings CSV missing required column: taxon_name")
    emb["taxon_name"] = emb["taxon_name"].astype("string").str.strip()
    emb = emb[emb["taxon_name"].notna() & (emb["taxon_name"] != "")].copy()
    required_pc_cols = ["PC1", "PC2", "PC3", "PC4", "PC5"]
    missing_pc = [c for c in required_pc_cols if c not in emb.columns]
    if missing_pc:
        raise KeyError(f"Embeddings CSV missing required columns: {', '.join(missing_pc)}")
    emb = emb[["taxon_name", *required_pc_cols]].copy()
    emb = emb.rename(
        columns={
            "PC1": "pc1",
            "PC2": "pc2",
            "PC3": "pc3",
            "PC4": "pc4",
            "PC5": "pc5",
        }
    )
    emb = emb.drop_duplicates(subset=["taxon_name"], keep="first")
    return emb


def load_filtered_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "taxon_name" not in df.columns:
        raise KeyError("Filtered CSV missing required column: taxon_name")
    df["taxon_name"] = df["taxon_name"].astype("string").str.strip()
    return df


def drop_classes_below_min_species(
    df: pd.DataFrame,
    *,
    min_species: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop classes with fewer than `min_species` unique taxon_name values."""
    out = df.copy()
    if "class" not in out.columns:
        raise KeyError("Merged frame missing required column: class")
    if "taxon_name" not in out.columns:
        raise KeyError("Merged frame missing required column: taxon_name")

    out["class"] = out["class"].astype("string").str.strip()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    valid = out[out["taxon_name"].notna() & (out["taxon_name"] != "")]
    species_per_class = valid.groupby("class", dropna=False)["taxon_name"].nunique()
    drop_classes = [
        str(c) for c in species_per_class[species_per_class < min_species].index.tolist()
    ]
    if drop_classes:
        out = out[~out["class"].astype(str).isin(drop_classes)].copy()
    return out.reset_index(drop=True), sorted(drop_classes)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Merge phylogenetic PCA embeddings with filtered observations by taxon_name. "
            "Keep only taxa present in embedding file, then drop classes with fewer than "
            "N unique species."
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/phylogeny/phylogenetic_embeddings.csv"),
        help="Input phylogenetic embedding CSV path.",
    )
    parser.add_argument(
        "--filtered",
        type=Path,
        default=Path("data/cleaning/filtered_data.csv"),
        help="Input filtered observation CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/merge_phylo.csv"),
        help="Output merged CSV path (default: data/merge_phylo.csv).",
    )
    parser.add_argument(
        "--min-species-per-class",
        type=int,
        default=7,
        help=(
            "After phylogeny join, drop classes with fewer than this many unique "
            "taxon_name values (default: 7)."
        ),
    )
    args = parser.parse_args()

    emb_path = args.embeddings if args.embeddings.is_absolute() else root / args.embeddings
    filtered_path = args.filtered if args.filtered.is_absolute() else root / args.filtered
    out_path = args.output if args.output.is_absolute() else root / args.output

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered file not found: {filtered_path}")

    emb = load_embeddings(emb_path)
    filtered = load_filtered_data(filtered_path)

    # Remove old pc columns if rerun, then append new pc1-pc5 columns at the end.
    pc_cols = ["pc1", "pc2", "pc3", "pc4", "pc5"]
    filtered_base = filtered.drop(columns=[c for c in pc_cols if c in filtered.columns]).copy()
    merged = filtered_base.merge(emb, on="taxon_name", how="inner", validate="many_to_one")
    merged = merged[[*filtered_base.columns, *pc_cols]]
    rows_after_join = len(merged)

    merged, dropped_sparse = drop_classes_below_min_species(
        merged, min_species=args.min_species_per_class
    )
    if dropped_sparse:
        print(
            f"Dropped classes with fewer than {args.min_species_per_class} species "
            f"after phylogeny join: " + ", ".join(dropped_sparse)
        )
        print(
            f"Rows after post-join min-species filter: {len(merged)} "
            f"(removed {rows_after_join - len(merged)})"
        )
    else:
        print(
            f"No classes below {args.min_species_per_class} species after phylogeny join."
        )
    if merged.empty:
        raise ValueError(
            f"No rows left after dropping classes with fewer than "
            f"{args.min_species_per_class} species."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Embedding taxa: {len(emb)}")
    print(f"number of rows input: {len(filtered)}")
    print(f"number of rows after phylogeny join: {rows_after_join}")
    print(f"number of rows output: {len(merged)}")
    print(f"rows with matched pc1-5: {int(merged['pc1'].notna().sum())}")
    print(f"rows removed (no embedding match): {len(filtered) - rows_after_join}")
    print(
        "Remaining classes: "
        + ", ".join(sorted(merged["class"].dropna().astype(str).unique().tolist()))
    )


if __name__ == "__main__":
    main()
