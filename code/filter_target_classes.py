#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytaxon.pytaxon as pytaxon_module
from pygbif import species as gbif_species
from pytaxon import Pytaxon


DEFAULT_PYTAXON_CONFIG = {
    "withAllMatches": False,
    "withCapitalization": False,
    "withSpeciesGroup": False,
    "withUninomialFuzzyMatch": True,
    "withStats": True,
    "mainTaxonThreshold": 0.6,
}
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

EXCLUDED_CLASSES = {
    "Gammaproteobacteria",
    "Bacillariophyceae",
    "Cyanophyceae",
    "Bacilli",
    "Alphaproteobacteria",
    "Prymnesiophyceae",
    "Chlorophyceae",
    "Dinophyceae",
    "Actinobacteria",
    "Deltaproteobacteria",
    "Trebouxiophyceae",
    "Oscillatoriophycideae",
    "Betaproteobacteria",
    "Xanthophyceae",
    "Chrysophyceae",
    "Saccharomycetes",
    "Mollicutes",
    "Thermoplasmata",
    "Kinetoplastea",
    "Euglenophyceae",
    "Methanobacteria",
    "Methanomicrobia",
}


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def clean_text(value: Any) -> str:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    return " ".join(text.split())


@contextmanager
def ensure_pytaxon_config(config_path: Path):
    # Keep config persisted under code/ for pytaxon runtime use.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(DEFAULT_PYTAXON_CONFIG, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    yield


@contextmanager
def working_directory(target: Path):
    prev = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(prev)


def retry_call(fn, retries: int = 3, delay_seconds: float = 0.35):
    last_exc: Exception | None = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i < retries - 1:
                time.sleep(delay_seconds * (i + 1))
    raise RuntimeError(f"Retry failed after {retries} attempts: {last_exc}") from last_exc


def split_binomial(name: str) -> tuple[str, str]:
    parts = clean_text(name).split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def standardize_names_with_pytaxon(
    names: list[str],
    source_id: int = 4,
    timeout_seconds: float = 20.0,
    pause_seconds: float = 0.01,
) -> dict[str, str]:
    original_post = pytaxon_module.requests.post

    def patched_post(*args, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return original_post(*args, **kwargs)

    pytaxon_module.requests.post = patched_post
    config_path = CONFIG_PATH
    code_dir = config_path.parent
    mapping: dict[str, str] = {}
    try:
        with ensure_pytaxon_config(config_path), working_directory(code_dir):
            resolver = Pytaxon(source_id=source_id)
            for idx, raw_name in enumerate(names, start=1):
                try:
                    data = retry_call(lambda: resolver.verify_taxon(raw_name))
                except Exception:  # noqa: BLE001
                    data = None

                standard_name = raw_name
                if isinstance(data, dict):
                    scientific = data.get("scientificName", ["", ""])
                    if len(scientific) > 0 and clean_text(scientific[0]):
                        standard_name = clean_text(scientific[0])

                mapping[raw_name] = standard_name
                if pause_seconds > 0:
                    time.sleep(pause_seconds)
                if idx % 200 == 0:
                    print(f"[pytaxon] processed: {idx}/{len(names)}", flush=True)
    finally:
        pytaxon_module.requests.post = original_post
    return mapping


def gbif_class_for_name(name: str, timeout_seconds: float = 20.0) -> str:
    data = retry_call(
        lambda: gbif_species.name_backbone(
            scientificName=name, verbose=True, timeout=timeout_seconds
        )
    )
    classification = data.get("classification", []) if isinstance(data, dict) else []
    for node in classification:
        if clean_text(node.get("rank", "")).upper() == "CLASS":
            return clean_text(node.get("name", ""))
    return ""


def is_excluded_class(class_name: str) -> bool:
    c = clean_text(class_name)
    if not c:
        return False
    if c.startswith("Tree "):
        return True
    return c in EXCLUDED_CLASSES


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Filter taxa in three steps: "
            "(1) remove taxa with <2 records, "
            "(2) standardize names via pytaxon, "
            "(3) remove excluded classes (tree/fungi/bacteria/algae lists)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cleaning/merged_bmr_mass_temperature.csv"),
        help="Input merged CSV path.",
    )
    parser.add_argument(
        "--standard-output",
        type=Path,
        default=Path("data/cleaning/standard_data.csv"),
        help="Output CSV after standardization.",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=Path("data/cleaning/filtered_data.csv"),
        help="Output CSV after class filtering.",
    )
    parser.add_argument(
        "--ncbi-source-id",
        type=int,
        default=4,
        help="pytaxon source id for NCBI (default: 4).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="API timeout seconds (default: 20).",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.01,
        help="Pause between API calls (default: 0.01).",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else root / args.input
    standard_path = (
        args.standard_output if args.standard_output.is_absolute() else root / args.standard_output
    )
    filtered_path = (
        args.filtered_output if args.filtered_output.is_absolute() else root / args.filtered_output
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    rows_initial = len(df)
    for col in ["Genus", "species"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        df[col] = df[col].astype("string").str.strip()

    # Step 0: delete blacklist classes first.
    if "class" in df.columns:
        df["class"] = df["class"].astype("string").str.strip()
        df = df[~df["class"].map(lambda x: is_excluded_class(str(x)))].copy()
    rows_after_blacklist = len(df)
    print(f"Rows after blacklist class removal: {rows_after_blacklist}", flush=True)

    # Keep rows with explicit genus/species before species-level counting.
    df = df[df["Genus"].notna() & (df["Genus"] != "") & df["species"].notna() & (df["species"] != "")]
    df = df.copy()
    raw_taxon_name = (df["Genus"] + " " + df["species"]).str.strip()

    # Step 1: remove taxa with record count < 2
    counts = raw_taxon_name.value_counts()
    keep_mask = raw_taxon_name.map(counts).fillna(0).astype(int) >= 2
    df = df.loc[keep_mask].copy()
    raw_taxon_name = raw_taxon_name.loc[keep_mask]
    print(f"Rows after count filter (>=2): {len(df)}", flush=True)

    unique_names = sorted(raw_taxon_name.unique().tolist())
    print(f"Unique taxa to standardize: {len(unique_names)}", flush=True)

    # Step 2: pytaxon standardization
    raw_to_standard = standardize_names_with_pytaxon(
        unique_names,
        source_id=args.ncbi_source_id,
        timeout_seconds=args.timeout_seconds,
        pause_seconds=args.pause_seconds,
    )
    standardized_taxon = raw_taxon_name.map(raw_to_standard).fillna(raw_taxon_name)
    df["taxon_name"] = standardized_taxon.values
    # Keep output clean: only append taxon_name as new column.
    ordered_cols = [c for c in df.columns if c != "taxon_name"] + ["taxon_name"]
    df = df[ordered_cols]
    rows_standard = len(df)
    standard_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(standard_path, index=False, encoding="utf-8")
    print(f"Saved: {standard_path}", flush=True)
    print(
        "Rows removed: "
        f"{rows_initial - rows_standard}",
        flush=True,
    )

    # Step 3: final filtering (blacklist classes by GBIF class as well)
    unique_standard_names = sorted(df["taxon_name"].astype(str).unique().tolist())
    class_map: dict[str, str] = {}
    for idx, name in enumerate(unique_standard_names, start=1):
        try:
            class_map[name] = gbif_class_for_name(name, timeout_seconds=args.timeout_seconds)
        except Exception:  # noqa: BLE001
            class_map[name] = ""
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)
        if idx % 200 == 0:
            print(f"[gbif] processed: {idx}/{len(unique_standard_names)}", flush=True)

    filtered = df.copy()
    # Safety guard: remove blacklist classes from original class and GBIF class.
    if "class" in filtered.columns:
        filtered = filtered[~filtered["class"].map(lambda x: is_excluded_class(str(x)))].copy()
    gbif_class_series = filtered["taxon_name"].map(class_map).fillna("")
    filtered = filtered[~gbif_class_series.map(lambda x: is_excluded_class(str(x)))].copy()

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_path, index=False, encoding="utf-8")
    print(f"Saved: {filtered_path}", flush=True)
    print(f"After filtering: {len(filtered)}", flush=True)
    print(
        "Rows removed: "
        f"{rows_standard - len(filtered)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
