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


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Merge phylogenetic PCA embeddings with filtered observations by taxon_name. "
            "Keep only taxa present in embedding file."
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Embedding taxa: {len(emb)}")
    print(f"number of rows input: {len(filtered)}")
    print(f"number of rows output: {len(merged)}")
    print(f"rows with matched pc1-5: {int(merged['pc1'].notna().sum())}")
    print(f"rows removed (no embedding match): {len(filtered) - len(merged)}")


if __name__ == "__main__":
    main()
