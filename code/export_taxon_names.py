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


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description="Export unique taxon_name values to txt (one per line)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cleaning/filtered_data.csv"),
        help="Input CSV path containing taxon_name column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phylogeny/unique_taxon_names.txt"),
        help="Output txt path (one taxon_name per line).",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "taxon_name" not in df.columns:
        raise KeyError("Missing required column: taxon_name")

    series = df["taxon_name"].astype("string").str.strip()
    series = series[series.notna() & (series != "")]
    unique_names = list(dict.fromkeys(series.tolist()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(unique_names) + "\n", encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"Unique taxon_name count: {len(unique_names)}")


if __name__ == "__main__":
    main()
