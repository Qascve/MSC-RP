#!/usr/bin/env python3
"""
Merge selected source files into one CSV with unified columns.

Sources:
- data/raw/pnas.2303764120.sd01.xlsx
- data/raw/observations.xlsx
- data/raw/41586_2010_BFnature08920_MOESM90_ESM.xls

Output columns (fixed order):
- Species
- wet_mass_g
- wet_mass_kg
- BMR
- BMR_unit
- temperature
- temperature_unit
- Food
- Habitat
- Torpor
- Islands
- Mountains
- source_file
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def find_root(start: Optional[Path] = None, marker: str = ".gitignore") -> Path:
    """
    Find project root by walking up directories until `marker` is found.

    Priority:
    1) caller-provided `start`
    2) current working directory
    3) this script location
    """
    anchors = [start] if start is not None else [Path.cwd(), Path(__file__).resolve().parent]

    checked = set()
    for anchor in anchors:
        current = anchor.resolve()
        if current.is_file():
            current = current.parent

        for candidate in [current, *current.parents]:
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / marker).exists():
                return candidate

    raise FileNotFoundError(
        f"Could not find project root: no '{marker}' found from {', '.join(str(a) for a in anchors if a is not None)}"
    )


def detect_header_row(path: Path, sheet_name: Optional[str] = None, max_rows: int = 50) -> int:
    """Heuristically detect the most likely header row for Excel files."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_rows)
    best_idx = 0
    best_score = (-1, -1)

    for i in range(len(raw)):
        values = [str(v).strip() for v in raw.iloc[i].tolist()]
        non_blank = [v for v in values if v and v.lower() != "nan"]
        score = (len(non_blank), len(set(non_blank)))
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make duplicate column names unique by appending suffixes."""
    cols = []
    seen = {}
    for col in df.columns:
        name = str(col).strip()
        idx = seen.get(name, 0)
        cols.append(name if idx == 0 else f"{name}__{idx}")
        seen[name] = idx + 1
    out = df.copy()
    out.columns = cols
    return out


def read_excel_auto_header(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    header_idx = detect_header_row(path, sheet_name=sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_idx)
    return dedupe_columns(df)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_output_frame(length: int) -> pd.DataFrame:
    cols = [
        "Species",
        "wet_mass_g",
        "wet_mass_kg",
        "BMR",
        "BMR_unit",
        "temperature",
        "temperature_unit",
        "Food",
        "Habitat",
        "Torpor",
        "Islands",
        "Mountains",
        "source_file",
    ]
    return pd.DataFrame({c: [np.nan] * length for c in cols})


def ensure_weight_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-calculate wet_mass_g/wet_mass_kg if one side is missing.
    kg = g / 1000
    g  = kg * 1000
    """
    out = df.copy()
    g = numeric(out["wet_mass_g"])
    kg = numeric(out["wet_mass_kg"])

    out["wet_mass_kg"] = np.where(kg.notna(), kg, np.where(g.notna(), g / 1000.0, np.nan))
    out["wet_mass_g"] = np.where(g.notna(), g, np.where(kg.notna(), kg * 1000.0, np.nan))
    return out


def clean_text_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return out


def drop_incomplete_core_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Drop rows missing any core field: wet mass, BMR, temperature.
    2) Remove duplicates for biologically-equivalent records.
    """
    out = df.copy()

    core_mask = (
        pd.to_numeric(out["wet_mass_g"], errors="coerce").notna()
        & pd.to_numeric(out["BMR"], errors="coerce").notna()
        & pd.to_numeric(out["temperature"], errors="coerce").notna()
    )
    out = out.loc[core_mask].copy()

    dedup_cols = [
        "Species",
        "wet_mass_g",
        "wet_mass_kg",
        "BMR",
        "BMR_unit",
        "temperature",
        "temperature_unit",
        "Food",
        "Habitat",
        "Torpor",
        "Islands",
        "Mountains",
    ]
    out = out.drop_duplicates(subset=dedup_cols, keep="first")
    return out


def parse_41586(path: Path) -> pd.DataFrame:
    df = read_excel_auto_header(path, sheet_name="McNab 2008 Edited.csv")
    out = make_output_frame(len(df))

    out["Species"] = df["Genus Species"] if "Genus Species" in df.columns else np.nan
    out["wet_mass_g"] = numeric(df["Mass (g)"]) if "Mass (g)" in df.columns else np.nan
    out["wet_mass_kg"] = np.nan
    out["BMR"] = numeric(df["BMR (W)"]) if "BMR (W)" in df.columns else np.nan
    out["BMR_unit"] = np.where(pd.to_numeric(out["BMR"], errors="coerce").notna(), "W", np.nan)
    out["temperature"] = numeric(df["Temperature (C)"]) if "Temperature (C)" in df.columns else np.nan
    out["temperature_unit"] = np.where(pd.to_numeric(out["temperature"], errors="coerce").notna(), "C", np.nan)
    out["Food"] = df["Food"] if "Food" in df.columns else np.nan
    out["Habitat"] = df["Habitat"] if "Habitat" in df.columns else np.nan
    out["Torpor"] = df["Torpor"] if "Torpor" in df.columns else np.nan
    out["Islands"] = df["Islands"] if "Islands" in df.columns else np.nan
    out["Mountains"] = df["Mountains"] if "Mountains" in df.columns else np.nan
    out["source_file"] = path.name

    return ensure_weight_pair(out)


def parse_pnas(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Metabolic_Data")
    df = dedupe_columns(df)
    out = make_output_frame(len(df))

    if "Publication Species Name" in df.columns:
        out["Species"] = df["Publication Species Name"]
    elif "Species" in df.columns:
        out["Species"] = df["Species"]
    else:
        out["Species"] = np.nan

    out["wet_mass_g"] = numeric(df["Wet Mass (g)"]) if "Wet Mass (g)" in df.columns else np.nan
    out["wet_mass_kg"] = np.nan

    bmr_col = "Metabolic Rate (W, at 25C)"
    out["BMR"] = numeric(df[bmr_col]) if bmr_col in df.columns else np.nan
    out["BMR_unit"] = np.where(pd.to_numeric(out["BMR"], errors="coerce").notna(), "W", np.nan)

    out["temperature"] = numeric(df["T (C)"]) if "T (C)" in df.columns else np.nan
    out["temperature_unit"] = np.where(pd.to_numeric(out["temperature"], errors="coerce").notna(), "C", np.nan)

    out["Food"] = np.nan
    out["Habitat"] = np.nan
    out["Torpor"] = np.nan
    out["Islands"] = np.nan
    out["Mountains"] = np.nan
    out["source_file"] = path.name

    return ensure_weight_pair(out)


def parse_observations(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Observations")
    df = dedupe_columns(df)
    out = make_output_frame(len(df))

    out["Species"] = df["species"] if "species" in df.columns else np.nan

    # observations.xlsx body mass is in "body mass" with units in "body mass - units"
    mass = numeric(df["body mass"]) if "body mass" in df.columns else pd.Series([np.nan] * len(df))
    mass_unit = df["body mass - units"] if "body mass - units" in df.columns else pd.Series([np.nan] * len(df))
    mass_unit = mass_unit.astype("string").str.strip().str.lower().fillna("")

    out["wet_mass_g"] = np.where(mass_unit == "kg", mass * 1000.0, np.where(mass_unit == "g", mass, np.nan))
    out["wet_mass_kg"] = np.where(mass_unit == "g", mass / 1000.0, np.where(mass_unit == "kg", mass, np.nan))

    mr = numeric(df["metabolic rate"]) if "metabolic rate" in df.columns else pd.Series([np.nan] * len(df))
    mr_unit = df["metabolic rate - units"] if "metabolic rate - units" in df.columns else pd.Series([np.nan] * len(df))
    out["BMR"] = mr
    out["BMR_unit"] = np.where(pd.to_numeric(out["BMR"], errors="coerce").notna(), mr_unit, np.nan)

    out["temperature"] = (
        numeric(df["original temperature"]) if "original temperature" in df.columns else np.nan
    )
    out["temperature_unit"] = np.where(pd.to_numeric(out["temperature"], errors="coerce").notna(), "C", np.nan)

    out["Food"] = np.nan
    out["Habitat"] = np.nan
    out["Torpor"] = np.nan
    out["Islands"] = np.nan
    out["Mountains"] = np.nan
    out["source_file"] = path.name

    return ensure_weight_pair(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge BMR/mass/temperature datasets into one CSV.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detected by searching for .gitignore).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <base-dir>/results/merged_bmr_mass_temperature.csv).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir if args.base_dir is not None else find_root()
    if args.output is None:
        output_path = base_dir / "data" / "merged_bmr_mass_temperature.csv"
    else:
        output_path = args.output if args.output.is_absolute() else base_dir / args.output

    pnas_path = base_dir / "data" / "raw" / "pnas.2303764120.sd01.xlsx"
    obs_path = base_dir / "data" / "raw" / "observations.xlsx"
    mcnab_path = base_dir / "data" / "raw" / "41586_2010_BFnature08920_MOESM90_ESM.xls"

    missing = [p for p in [pnas_path, obs_path, mcnab_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(str(p) for p in missing)}")

    merged = pd.concat(
        [parse_pnas(pnas_path), parse_observations(obs_path), parse_41586(mcnab_path)],
        ignore_index=True,
    )

    merged = clean_text_cols(
        merged,
        [
            "Species",
            "BMR_unit",
            "temperature_unit",
            "Food",
            "Habitat",
            "Torpor",
            "Islands",
            "Mountains",
            "source_file",
        ],
    )

    # Always keep only usable core rows and remove duplicate records.
    merged = drop_incomplete_core_and_deduplicate(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    try:
        saved_path = output_path.relative_to(base_dir)
    except ValueError:
        saved_path = output_path
    print(f"Saved: {saved_path}")
    print(f"Rows: {len(merged)}")
    print("Non-null counts:")
    for c in ["Species", "wet_mass_g", "wet_mass_kg", "BMR", "temperature"]:
        print(f"  {c}: {int(merged[c].notna().sum())}")


if __name__ == "__main__":
    main()
