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


def load_embeddings_with_species(path: Path) -> pd.DataFrame:
    emb = pd.read_csv(path)
    if emb.empty:
        raise ValueError(f"Embedding file is empty: {path}")

    if "Species" in emb.columns:
        species_col = "Species"
    elif "species_normalized" in emb.columns:
        species_col = "species_normalized"
    else:
        # Usually the species index column written by pandas appears as Unnamed: 0.
        species_col = emb.columns[0]

    emb = emb.rename(columns={species_col: "Species"})
    emb["Species"] = emb["Species"].astype("string").str.strip()
    emb = emb[emb["Species"].notna() & (emb["Species"] != "")].copy()
    emb = emb.drop_duplicates(subset=["Species"], keep="first")
    return emb


def main() -> None:
    root = find_root()

    parser = argparse.ArgumentParser(
        description=(
            "Filter observations by species present in phylogenetic embeddings "
            "and merge embeddings into all matching observation rows."
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/phylogenetic_embeddings.csv"),
        help="Path to phylogenetic embeddings CSV.",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/merged_bmr_mass_temperature.csv"),
        help="Path to merged observations CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/merged_phylo_embeddings.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    emb_path = args.embeddings if args.embeddings.is_absolute() else root / args.embeddings
    obs_path = (
        args.observations if args.observations.is_absolute() else root / args.observations
    )
    out_path = args.output if args.output.is_absolute() else root / args.output

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"Observations file not found: {obs_path}")

    emb = load_embeddings_with_species(emb_path)
    obs = pd.read_csv(obs_path)
    if "Species" not in obs.columns:
        raise KeyError("Missing required column in observations CSV: Species")

    obs["Species"] = obs["Species"].astype("string").str.strip()
    obs = obs[obs["Species"].notna() & (obs["Species"] != "")].copy()

    matched_obs = obs[obs["Species"].isin(set(emb["Species"]))].copy()
    merged = matched_obs.merge(emb, on="Species", how="left", validate="many_to_one")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Embedding species: {len(emb)}")
    print(f"Matched observation rows: {len(matched_obs)}")
    print(f"Output rows: {len(merged)}")


if __name__ == "__main__":
    main()
