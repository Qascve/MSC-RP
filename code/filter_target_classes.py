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

KEEP_CLASSES = {
    "Teleostei",
    "Insecta",
    "Mammalia",
    "Malacostraca",
    "Reptilia",
    "Amphibia",
    "Maxillopoda",
    "Aves",
    "Arachnida",
    "Chondrichthyes",
    "Cephalaspidomorphi",
    "Chondrostei",
    "Branchiopoda",
    "Cephalopoda",
    "Sagittoidea",
    "Hydrozoa",
    "Dipnotetrapodomorpha",
    "Ostracoda",
    "Scyphozoa",
    "Myxini",
    "Chilopoda",
    "Cladistei",
    "Gastropoda",
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


def is_allowed_class(class_name: str, *, allow_empty: bool = False) -> bool:
    c = clean_text(class_name)
    if not c:
        return allow_empty
    return c in KEEP_CLASSES


def ensure_wet_mass_g(df: pd.DataFrame) -> pd.DataFrame:
    """Derive wet_Mass_g from wet_Mass_kg when only kg is present (merged schema)."""
    out = df.copy()
    if "wet_Mass_kg" in out.columns:
        kg = pd.to_numeric(out["wet_Mass_kg"], errors="coerce")
        if "wet_Mass_g" not in out.columns:
            out["wet_Mass_g"] = kg * 1000.0
        else:
            g = pd.to_numeric(out["wet_Mass_g"], errors="coerce")
            out["wet_Mass_g"] = g.where(g.notna(), kg * 1000.0)
        # Prefer historical column order: ... species, wet_Mass_g, wet_Mass_kg, BMR ...
        cols = list(out.columns)
        if "wet_Mass_g" in cols and "wet_Mass_kg" in cols:
            cols = [c for c in cols if c != "wet_Mass_g"]
            kg_idx = cols.index("wet_Mass_kg")
            cols.insert(kg_idx, "wet_Mass_g")
            out = out[cols]
    return out


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Filter taxa in two steps: "
            "(1) keep whitelist classes only, "
            "(2) standardize names via pytaxon and write filtered outputs."
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
    parser.add_argument(
        "--reuse-taxon-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with Genus/species/taxon_name to reuse prior NCBI mappings. "
            "Unmapped names still go through pytaxon."
        ),
    )
    parser.add_argument(
        "--skip-pytaxon",
        action="store_true",
        help="Skip pytaxon API calls; use Genus+species (or reused map) as taxon_name.",
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
    df = ensure_wet_mass_g(df)
    rows_initial = len(df)
    for col in ["Genus", "species"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        df[col] = df[col].astype("string").str.strip()

    # Step 0: keep whitelist classes only.
    if "class" not in df.columns:
        raise KeyError("Missing required column: class")
    df["class"] = df["class"].astype("string").str.strip()
    df = df[df["class"].map(lambda x: is_allowed_class(str(x)))].copy()
    rows_after_whitelist = len(df)
    print(f"Rows after whitelist class filter: {rows_after_whitelist}", flush=True)
    print(f"Kept classes: {sorted(KEEP_CLASSES)}", flush=True)

    # Keep rows with explicit genus/species before species-level counting.
    df = df[df["Genus"].notna() & (df["Genus"] != "") & df["species"].notna() & (df["species"] != "")]
    df = df.copy()
    raw_taxon_name = (df["Genus"] + " " + df["species"]).str.strip()

    unique_names = sorted(raw_taxon_name.unique().tolist())
    print(f"Unique taxa to standardize: {len(unique_names)}", flush=True)

    raw_to_standard: dict[str, str] = {name: name for name in unique_names}
    if args.reuse_taxon_csv is not None:
        reuse_path = (
            args.reuse_taxon_csv
            if args.reuse_taxon_csv.is_absolute()
            else root / args.reuse_taxon_csv
        )
        if not reuse_path.exists():
            raise FileNotFoundError(f"Reuse taxon CSV not found: {reuse_path}")
        reuse_df = pd.read_csv(reuse_path)
        needed = {"Genus", "species", "taxon_name"}
        missing_reuse = needed.difference(reuse_df.columns)
        if missing_reuse:
            raise KeyError(
                f"Reuse CSV missing columns: {', '.join(sorted(missing_reuse))}"
            )
        reuse_df = reuse_df.copy()
        reuse_df["Genus"] = reuse_df["Genus"].astype("string").str.strip()
        reuse_df["species"] = reuse_df["species"].astype("string").str.strip()
        reuse_df["taxon_name"] = reuse_df["taxon_name"].astype("string").str.strip()
        reuse_df["raw_binomial"] = (
            reuse_df["Genus"] + " " + reuse_df["species"]
        ).str.strip()
        reuse_df = reuse_df[
            reuse_df["raw_binomial"].notna()
            & (reuse_df["raw_binomial"] != "")
            & reuse_df["taxon_name"].notna()
            & (reuse_df["taxon_name"] != "")
        ]
        reuse_map = (
            reuse_df.drop_duplicates(subset=["raw_binomial"], keep="first")
            .set_index("raw_binomial")["taxon_name"]
            .to_dict()
        )
        hit = 0
        for name in unique_names:
            if name in reuse_map:
                raw_to_standard[name] = reuse_map[name]
                hit += 1
        print(f"Reused taxon_name for {hit}/{len(unique_names)} binomials", flush=True)

    need_api = [n for n in unique_names if raw_to_standard.get(n, n) == n]
    if args.skip_pytaxon:
        print(
            f"Skipping pytaxon ({len(need_api)} names keep Genus+species or reused map).",
            flush=True,
        )
    elif need_api and args.reuse_taxon_csv is not None:
        # Only resolve names that were not covered by the reuse map.
        unresolved = [n for n in unique_names if n not in reuse_map]
        if unresolved:
            print(f"Resolving {len(unresolved)} new taxa via pytaxon", flush=True)
            api_map = standardize_names_with_pytaxon(
                unresolved,
                source_id=args.ncbi_source_id,
                timeout_seconds=args.timeout_seconds,
                pause_seconds=args.pause_seconds,
            )
            raw_to_standard.update(api_map)
        else:
            print("All taxa covered by reuse map; skipping pytaxon.", flush=True)
    else:
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

    # Step 3: final filtering by original class whitelist.
    # Do not drop rows when GBIF returns an alternate class name
    # (e.g. Elasmobranchii vs Chondrichthyes); source class is authoritative here.
    filtered = df[df["class"].map(lambda x: is_allowed_class(str(x)))].copy()
    filtered = ensure_wet_mass_g(filtered)

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_path, index=False, encoding="utf-8")
    print(f"Saved: {filtered_path}", flush=True)
    print(f"After filtering: {len(filtered)}", flush=True)
    print(
        "Rows removed: "
        f"{rows_standard - len(filtered)}",
        flush=True,
    )
    print(
        "Remaining classes: "
        f"{sorted(filtered['class'].dropna().astype(str).unique().tolist())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
