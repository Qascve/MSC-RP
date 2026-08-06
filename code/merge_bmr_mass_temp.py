#!/usr/bin/env python3
# Merge three source datasets into one CSV with unified columns.
#
# Sources:
# - data/raw/pnas.2303764120.sd01.xlsx
# - data/raw/observations.xlsx
# - data/raw/41586_2010_BFnature08920_MOESM90_ESM.xls
#
# Keep every observation row that simultaneously has:
# Genus, species, temperature, and BMR.
# No per-species record-count limit is applied.
#
# Output columns (fixed order):
# - class
# - order
# - family
# - Genus
# - species
# - wet_Mass_kg
# - BMR
# - BMR_unit
# - temperature
# - temperature_unit
# - Reference
#
# Mass is standardized to kg. Metabolic rate is kept only when reported in W
# (no kJ/h, kJ/s, mW/kW, or other unit conversions). PNAS rows are restricted
# to Type of Metabolic Rate == Basal; AnimalTraits rows are restricted to
# metabolic rate - method == "basal metabolic rate". Genus/species are kept
# as parsed from source binomials without renaming.

from __future__ import annotations
import argparse
import re
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd
from pygbif import species as gbif_species

GENUS_COL = "Genus"
SPECIES_COL = "species"
WET_G_COL = "wet_Mass_g"
WET_KG_COL = "wet_Mass_kg"

OUTPUT_COLS = [
    "class",
    "order",
    "family",
    GENUS_COL,
    SPECIES_COL,
    WET_KG_COL,
    "BMR",
    "BMR_unit",
    "temperature",
    "temperature_unit",
    "Reference",
]


def find_root(start: Optional[Path] = None, marker: str = ".gitignore") -> Path:
    #     Find project root by walking up directories until `marker` is found.
    #
    # Priority:
    # 1) caller-provided `start`
    # 2) current working directory
    # 3) this script location
    #
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
    # Heuristically detect the most likely header row for Excel files.
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
    # Make duplicate column names unique by appending suffixes.
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
    # Keep wet_Mass_g only as an internal helper during conversion.
    cols = [
        "class",
        "order",
        "family",
        GENUS_COL,
        SPECIES_COL,
        WET_G_COL,
        WET_KG_COL,
        "BMR",
        "BMR_unit",
        "temperature",
        "temperature_unit",
        "Reference",
    ]
    return pd.DataFrame({c: [np.nan] * length for c in cols})


def ensure_weight_pair(df: pd.DataFrame) -> pd.DataFrame:
    #     Auto-calculate wet mass g/wet mass kg if one side is missing.
    # kg = g / 1000
    # g  = kg * 1000
    #
    out = df.copy()
    g = numeric(out[WET_G_COL])
    kg = numeric(out[WET_KG_COL])

    out[WET_KG_COL] = np.where(kg.notna(), kg, np.where(g.notna(), g / 1000.0, np.nan))
    out[WET_G_COL] = np.where(g.notna(), g, np.where(kg.notna(), kg * 1000.0, np.nan))
    return out


def convert_mass_value_unit_to_g_kg(
    mass_value: pd.Series, mass_unit: pd.Series
) -> tuple[pd.Series, pd.Series]:
    value = numeric(mass_value)
    unit = mass_unit.astype("string").str.strip().str.lower().fillna("")
    unit = unit.str.replace(".", "", regex=False).str.replace(" ", "", regex=False)

    is_kg = unit.isin(["kg", "kilogram", "kilograms"])
    is_g = unit.isin(["g", "gram", "grams"])
    is_mg = unit.isin(["mg", "milligram", "milligrams"])

    g = np.where(is_kg, value * 1000.0, np.where(is_g, value, np.where(is_mg, value / 1000.0, np.nan)))
    kg = np.where(is_kg, value, np.where(is_g, value / 1000.0, np.where(is_mg, value / 1_000_000.0, np.nan)))
    return pd.Series(g, index=mass_value.index), pd.Series(kg, index=mass_value.index)


def normalize_bmr_unit(unit: object) -> str:
    text = normalize_text_value(unit).lower()
    text = text.replace(".", "").replace(" ", "")
    text = text.replace("watts", "w").replace("watt", "w")
    return text


def convert_bmr_value_unit_to_w(
    bmr_value: pd.Series, bmr_unit: pd.Series
) -> tuple[pd.Series, pd.Series]:
    #     Keep metabolic rate only when the reported unit is watts (W).
    #
    # No conversion from mW/kW, J/s, kJ/h, kJ/s, O2 volume rates, or other
    # energy/time units: non-W units are treated as missing.
    #
    value = numeric(bmr_value)
    unit = bmr_unit.map(normalize_bmr_unit)
    is_w = unit.isin(["w", "watt", "watts"])

    watts = np.where(is_w, value, np.nan)
    unit_out = pd.Series(
        np.where(pd.notna(watts), "W", pd.NA),
        index=bmr_value.index,
        dtype="string",
    )
    return pd.Series(watts, index=bmr_value.index), unit_out


def parse_temperature_series(series: pd.Series) -> pd.Series:
    #     Parse numeric temperatures and simple ranges (e.g. '25-35', '4-20C') to midpoint.
    # Non-numeric placeholders such as ENDO/ND become NaN.
    #
    values: list[float] = []
    for raw in series.tolist():
        text = normalize_text_value(raw)
        if not text:
            values.append(np.nan)
            continue
        upper = text.upper()
        if upper in {"ENDO", "ND", "NA", "N/A", "NAN", "NONE"}:
            values.append(np.nan)
            continue

        direct = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
        if pd.notna(direct):
            values.append(float(direct))
            continue

        cleaned = re.sub(r"[°º]?\s*[Cc]\b", "", text)
        cleaned = cleaned.replace("–", "-").replace("—", "-")
        match = re.fullmatch(
            r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*",
            cleaned,
        )
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            values.append((low + high) / 2.0)
        else:
            values.append(np.nan)

    return pd.Series(values, index=series.index, dtype="float64")


def infer_unit_from_colname(colname: str) -> str:
    name = normalize_text_value(colname).lower()
    if "(kg)" in name or name.endswith("_kg") or " kg" in name:
        return "kg"
    if "(mg)" in name or name.endswith("_mg") or " mg" in name:
        return "mg"
    if "(g)" in name or name.endswith("_g") or " g" in name:
        return "g"
    return ""


def is_mass_value_col(colname: str) -> bool:
    name = normalize_text_value(colname).lower()
    if "mass" not in name:
        return False
    deny = ["specific", "metadata", "method", "comment", "minimum", "maximum", "min", "max", "specificepithet"]
    return not any(token in name for token in deny)


def find_unit_col_for_value_col(df: pd.DataFrame, value_col: str) -> Optional[str]:
    candidates = [
        f"{value_col} - units",
        f"{value_col}-units",
        f"{value_col}_units",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    value_name = normalize_text_value(value_col).lower()
    for col in df.columns:
        low = normalize_text_value(col).lower()
        if "unit" in low and value_name in low:
            return col
    return None


def mass_from_candidates(
    df: pd.DataFrame,
    candidates: list[tuple[str, Optional[str], Optional[str]]],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    #     Convert mass from candidate (value_col, unit_col, default_unit) triples.
    # Returns:
    #   - wet mass in g
    #   - wet mass in kg
    #   - raw mass fallback values (for rows still unresolved)
    #
    n = len(df)
    out_g = pd.Series([np.nan] * n)
    out_kg = pd.Series([np.nan] * n)
    raw_fallback = pd.Series([np.nan] * n)

    for value_col, unit_col, default_unit in candidates:
        if value_col not in df.columns:
            continue
        values = numeric(df[value_col])
        if unit_col is not None and unit_col in df.columns:
            g, kg = convert_mass_value_unit_to_g_kg(values, df[unit_col])
        else:
            unit_guess = default_unit or infer_unit_from_colname(value_col)
            if unit_guess == "kg":
                g, kg = values * 1000.0, values
            elif unit_guess == "mg":
                g, kg = values / 1000.0, values / 1_000_000.0
            elif unit_guess == "g":
                g, kg = values, values / 1000.0
            else:
                g, kg = pd.Series([np.nan] * n), pd.Series([np.nan] * n)

        g_series = pd.Series(g, index=df.index)
        kg_series = pd.Series(kg, index=df.index)
        values_series = pd.Series(values, index=df.index)

        out_g = out_g.where(pd.to_numeric(out_g, errors="coerce").notna(), g_series)
        out_kg = out_kg.where(pd.to_numeric(out_kg, errors="coerce").notna(), kg_series)
        raw_fallback = raw_fallback.where(
            pd.to_numeric(raw_fallback, errors="coerce").notna(), values_series
        )

    return out_g, out_kg, raw_fallback


def build_general_mass_candidates(df: pd.DataFrame) -> list[tuple[str, Optional[str], Optional[str]]]:
    #     Build flexible mass candidates for current and future datasets.
    # Priority:
    #   1) explicit common columns
    #   2) any mass-like column + matched unit column
    #
    candidates: list[tuple[str, Optional[str], Optional[str]]] = []
    explicit = [
        ("Wet Mass (g)", None, "g"),
        ("Wet Mass (kg)", None, "kg"),
        ("Mass (g)", None, "g"),
        ("Mass (kg)", None, "kg"),
        ("body mass", "body mass - units", None),
        ("original body mass", "original body mass - units", None),
    ]
    for value_col, unit_col, default_unit in explicit:
        if value_col in df.columns:
            candidates.append((value_col, unit_col, default_unit))

    for col in df.columns:
        if not is_mass_value_col(col):
            continue
        unit_col = find_unit_col_for_value_col(df, col)
        candidates.append((col, unit_col, None))

    seen = set()
    deduped: list[tuple[str, Optional[str], Optional[str]]] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def normalize_text_value(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    return " ".join(text.split())


def gbif_name_backbone_with_retry(
    scientific_name: str, timeout_seconds: float, retries: int, retry_delay_seconds: float
) -> dict:
    last_exc: Exception | None = None
    for i in range(max(1, retries)):
        try:
            data = gbif_species.name_backbone(
                scientificName=scientific_name,
                verbose=True,
                timeout=timeout_seconds,
            )
            if isinstance(data, dict):
                return data
            return {}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if i < max(1, retries) - 1:
                time.sleep(retry_delay_seconds * (i + 1))
    if last_exc is not None:
        return {}
    return {}


def extract_rank_from_gbif(data: dict, rank: str) -> str:
    direct = normalize_text_value(data.get(rank, ""))
    if direct:
        return direct
    classification = data.get("classification", []) if isinstance(data, dict) else []
    for node in classification:
        if normalize_text_value(node.get("rank", "")).lower() == rank.lower():
            value = normalize_text_value(node.get("name", ""))
            if value:
                return value
    return ""


def fill_missing_taxonomy_with_gbif(
    df: pd.DataFrame,
    timeout_seconds: float = 20.0,
    retries: int = 3,
    retry_delay_seconds: float = 0.35,
    pause_seconds: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    out = df.copy()
    taxonomy_cols = ["class", "order", "family"]

    for col in taxonomy_cols:
        out[col] = out[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    missing_before = {col: int(out[col].isna().sum()) for col in taxonomy_cols}
    if all(v == 0 for v in missing_before.values()):
        return out, missing_before, missing_before

    binomial = (
        out[GENUS_COL].astype("string").str.strip().fillna("")
        + " "
        + out[SPECIES_COL].astype("string").str.strip().fillna("")
    ).str.strip()
    binomial = binomial.where(binomial != "", pd.NA)
    missing_any = out[taxonomy_cols].isna().any(axis=1)
    target_names = sorted(binomial[missing_any & binomial.notna()].astype(str).unique().tolist())

    if not target_names:
        return out, missing_before, missing_before

    taxonomy_map: dict[str, dict[str, str]] = {}
    for idx, name in enumerate(target_names, start=1):
        gbif_data = gbif_name_backbone_with_retry(
            scientific_name=name,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        taxonomy_map[name] = {
            "class": extract_rank_from_gbif(gbif_data, "class"),
            "order": extract_rank_from_gbif(gbif_data, "order"),
            "family": extract_rank_from_gbif(gbif_data, "family"),
        }
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        if idx % 200 == 0:
            print(f"[gbif taxonomy fill] processed: {idx}/{len(target_names)}", flush=True)

    for col in taxonomy_cols:
        missing_mask = out[col].isna()
        fill_map = {name: values[col] for name, values in taxonomy_map.items() if values[col] != ""}
        if fill_map:
            fill_series = binomial.map(fill_map)
            fill_mask = missing_mask & fill_series.notna()
            out.loc[fill_mask, col] = fill_series.loc[fill_mask]
        out[col] = out[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    missing_after = {col: int(out[col].isna().sum()) for col in taxonomy_cols}
    return out, missing_before, missing_after


def extract_reference_series(df: pd.DataFrame) -> pd.Series:
    def clean_ref(col: str) -> pd.Series:
        ref = df[col].astype("string").str.strip()
        return ref.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    # Case-insensitive reference lookup:
    # supports names like "Reference", "XXReference", "Reference xx", etc.
    for col in df.columns:
        key = str(col).strip().lower()
        if "reference" in key:
            return clean_ref(str(col))

    return pd.Series([pd.NA] * len(df), dtype="string")


def extract_genus_species(
    genus_series: Optional[pd.Series],
    species_series: Optional[pd.Series],
    fallback_series: Optional[pd.Series] = None,
) -> tuple[pd.Series, pd.Series]:
    if genus_series is None and species_series is None and fallback_series is None:
        raise ValueError("At least one species-related series must be provided.")

    length = 0
    for candidate in [genus_series, species_series, fallback_series]:
        if candidate is not None:
            length = len(candidate)
            break
    genus_out: list[object] = []
    species_out: list[object] = []
    for i in range(length):
        genus = normalize_text_value(genus_series.iloc[i]) if genus_series is not None else ""
        species = normalize_text_value(species_series.iloc[i]) if species_series is not None else ""
        fallback = normalize_text_value(fallback_series.iloc[i]) if fallback_series is not None else ""

        g = ""
        s = ""
        if genus and species:
            # If species column already contains full binomial, re-split it.
            if species.lower().startswith(f"{genus.lower()} ") or len(species.split()) >= 2:
                parts = species.split()
                g = parts[0]
                s = parts[1]
            else:
                g = genus
                s = species.split()[0]
        elif fallback:
            parts = fallback.split()
            if len(parts) >= 2:
                g = parts[0]
                s = parts[1]
        elif species and len(species.split()) >= 2:
            parts = species.split()
            g = parts[0]
            s = parts[1]
        else:
            g = ""
            s = ""

        genus_out.append(g if g else pd.NA)
        species_out.append(s if s else pd.NA)

    return pd.Series(genus_out, dtype="string"), pd.Series(species_out, dtype="string")


def clean_text_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return out


def is_valid_taxon_token(series: pd.Series, *, allow_numeric: bool = False) -> pd.Series:
    # Reject blank / placeholder genus-species tokens such as NA, unknown, or bare numbers.
    text = series.astype("string").str.strip()
    lower = text.str.lower()
    invalid = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "unknown",
        "unidentified",
        "sp",
        "sp.",
        "spp",
        "spp.",
    }
    ok = text.notna() & ~lower.isin(invalid)
    if not allow_numeric:
        ok = ok & ~text.str.fullmatch(r"\d+")
    # Reject tokens that start with NA placeholder patterns: "NA", "NA 1", "NA2"
    ok = ok & ~lower.str.match(r"^na(?:[\s_-]*\d+)?$", na=False)
    return ok.fillna(False)


def drop_incomplete_core_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    #     1) Keep rows that simultaneously have Genus, species, temperature, and BMR.
    #    Also require wet_Mass_kg so body mass can be retained in kg.
    #    No per-species record-count limit is applied.
    # 2) Remove exact-duplicate biologically-equivalent records.
    #
    out = df.copy()

    core_mask = (
        is_valid_taxon_token(out[GENUS_COL])
        & is_valid_taxon_token(out[SPECIES_COL])
        & pd.to_numeric(out[WET_KG_COL], errors="coerce").notna()
        & pd.to_numeric(out["BMR"], errors="coerce").notna()
        & out["BMR_unit"].astype("string").str.strip().notna()
        & (out["BMR_unit"].astype("string").str.strip() != "")
        & pd.to_numeric(out["temperature"], errors="coerce").notna()
    )
    out = out.loc[core_mask].copy()

    dedup_cols = [
        GENUS_COL,
        SPECIES_COL,
        "class",
        "order",
        "family",
        WET_KG_COL,
        "BMR",
        "BMR_unit",
        "temperature",
        "temperature_unit",
        "Reference",
    ]
    out = out.drop_duplicates(subset=dedup_cols, keep="first")
    return out


def unique_classes_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Summarize unique class values after merge (Genus/species unchanged).
    out = df.copy()
    out["class"] = out["class"].astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "NaN": pd.NA}
    )
    binomial = (
        out[GENUS_COL].astype("string").str.strip().fillna("")
        + " "
        + out[SPECIES_COL].astype("string").str.strip().fillna("")
    ).str.strip()
    out["binomial"] = binomial.where(binomial != "", pd.NA)

    rows: list[dict[str, object]] = []
    counts = out["class"].dropna().value_counts()
    for class_name, n_rows in counts.items():
        subset = out.loc[out["class"] == class_name]
        rows.append(
            {
                "class": str(class_name),
                "rows": int(n_rows),
                "unique_genus_species": int(subset["binomial"].nunique(dropna=True)),
            }
        )

    missing = out["class"].isna()
    if missing.any():
        rows.append(
            {
                "class": "(missing)",
                "rows": int(missing.sum()),
                "unique_genus_species": int(out.loc[missing, "binomial"].nunique(dropna=True)),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            by=["rows", "class"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
    return result


def parse_41586(path: Path) -> pd.DataFrame:
    df = read_excel_auto_header(path, sheet_name="McNab 2008 Edited.csv")
    out = make_output_frame(len(df))

    genus_col = df["Genus"] if "Genus" in df.columns else None
    species_col = df["Species"] if "Species" in df.columns else None
    full_col = df["Genus Species"] if "Genus Species" in df.columns else None
    out[GENUS_COL], out[SPECIES_COL] = extract_genus_species(genus_col, species_col, full_col)
    out["class"] = np.nan
    out["order"] = df["Order"] if "Order" in df.columns else np.nan
    out["family"] = df["Family"] if "Family" in df.columns else np.nan

    mass_candidates = build_general_mass_candidates(df)
    mass_g, mass_kg, raw_mass = mass_from_candidates(df, mass_candidates)
    out[WET_G_COL] = mass_g
    out[WET_KG_COL] = mass_kg

    bmr_raw = numeric(df["BMR (W)"]) if "BMR (W)" in df.columns else pd.Series([np.nan] * len(df))
    bmr_unit_raw = pd.Series(["W"] * len(df), dtype="string")
    out["BMR"], out["BMR_unit"] = convert_bmr_value_unit_to_w(bmr_raw, bmr_unit_raw)

    out["temperature"] = (
        parse_temperature_series(df["Temperature (C)"])
        if "Temperature (C)" in df.columns
        else np.nan
    )
    temp_mask = pd.to_numeric(out["temperature"], errors="coerce").notna()
    out["temperature_unit"] = pd.Series("C", index=out.index, dtype="string").where(
        temp_mask, pd.NA
    )
    out["Reference"] = extract_reference_series(df)
    fallback_mask = (
        pd.to_numeric(out[WET_G_COL], errors="coerce").isna()
        & pd.to_numeric(out["BMR"], errors="coerce").notna()
        & pd.to_numeric(raw_mass, errors="coerce").notna()
    )
    # If wet/dry is unspecified but BMR exists, default unresolved mass as wet mass in grams.
    out.loc[fallback_mask, WET_G_COL] = raw_mass.loc[fallback_mask]
    out.loc[fallback_mask, WET_KG_COL] = raw_mass.loc[fallback_mask] / 1000.0

    return ensure_weight_pair(out)


def parse_pnas(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Metabolic_Data")
    df = dedupe_columns(df)

    # Keep basal metabolic rate only (drop Field / Maximum / Dark respiration / etc.).
    if "Type of Metabolic Rate" not in df.columns:
        raise KeyError(
            "PNAS sheet Metabolic_Data is missing required column "
            "'Type of Metabolic Rate'."
        )
    mr_type = df["Type of Metabolic Rate"].astype("string").str.strip()
    basal_mask = mr_type.str.casefold() == "basal"
    n_before = len(df)
    df = df.loc[basal_mask].copy().reset_index(drop=True)
    print(
        f"PNAS Type of Metabolic Rate filter: kept {len(df):,}/{n_before:,} "
        f"Basal rows "
        f"(dropped {n_before - len(df):,} non-Basal/missing)."
    )

    out = make_output_frame(len(df))

    genus_col = df["Genus"] if "Genus" in df.columns else None
    species_col = df["Species"] if "Species" in df.columns else None
    fallback_col = (
        df["Publication Species Name"] if "Publication Species Name" in df.columns else None
    )
    out[GENUS_COL], out[SPECIES_COL] = extract_genus_species(genus_col, species_col, fallback_col)
    out["class"] = df["Class"] if "Class" in df.columns else np.nan
    out["order"] = df["Order"] if "Order" in df.columns else np.nan
    out["family"] = df["Family"] if "Family" in df.columns else np.nan

    mass_candidates = build_general_mass_candidates(df)
    mass_g, mass_kg, raw_mass = mass_from_candidates(df, mass_candidates)
    out[WET_G_COL] = mass_g
    out[WET_KG_COL] = mass_kg

    # Prefer Metabolic Rate (W, at T) paired with T (C).
    # If that pair is incomplete, fall back to Metabolic Rate (W, at 25C)
    # with temperature fixed at 25 C (Basal rows only; already filtered above).
    if "Metabolic Rate (W, at T)" not in df.columns:
        raise KeyError(
            "PNAS sheet Metabolic_Data is missing required column "
            "'Metabolic Rate (W, at T)'."
        )
    if "T (C)" not in df.columns:
        raise KeyError(
            "PNAS sheet Metabolic_Data is missing required column 'T (C)'."
        )
    if "Metabolic Rate (W, at 25C)" not in df.columns:
        raise KeyError(
            "PNAS sheet Metabolic_Data is missing required column "
            "'Metabolic Rate (W, at 25C)'."
        )

    bmr_at_t = numeric(df["Metabolic Rate (W, at T)"])
    temp_at_t = parse_temperature_series(df["T (C)"])
    paired_ok = bmr_at_t.notna() & temp_at_t.notna()

    bmr_25 = numeric(df["Metabolic Rate (W, at 25C)"])
    use_25 = (~paired_ok) & bmr_25.notna()

    bmr_raw = bmr_at_t.where(paired_ok, bmr_25.where(use_25, np.nan))
    bmr_unit_raw = pd.Series(["W"] * len(df), dtype="string")
    out["BMR"], out["BMR_unit"] = convert_bmr_value_unit_to_w(bmr_raw, bmr_unit_raw)

    out["temperature"] = temp_at_t.where(paired_ok, np.where(use_25, 25.0, np.nan))
    temp_mask = pd.to_numeric(out["temperature"], errors="coerce").notna()
    out["temperature_unit"] = pd.Series("C", index=out.index, dtype="string").where(
        temp_mask, pd.NA
    )
    # Rows with neither at-T pair nor 25C rate remain missing.
    keep_rate = paired_ok | use_25
    out.loc[~keep_rate, "BMR"] = np.nan
    out.loc[~keep_rate, "BMR_unit"] = pd.NA
    out.loc[~keep_rate, "temperature"] = np.nan
    out.loc[~keep_rate, "temperature_unit"] = pd.NA
    print(
        f"PNAS rate/temperature pairing: at-T={int(paired_ok.sum()):,}, "
        f"25C-fallback={int(use_25.sum()):,}, "
        f"neither={int((~keep_rate).sum()):,}"
    )
    out["Reference"] = extract_reference_series(df)
    fallback_mask = (
        pd.to_numeric(out[WET_G_COL], errors="coerce").isna()
        & pd.to_numeric(out["BMR"], errors="coerce").notna()
        & pd.to_numeric(raw_mass, errors="coerce").notna()
    )
    out.loc[fallback_mask, WET_G_COL] = raw_mass.loc[fallback_mask]
    out.loc[fallback_mask, WET_KG_COL] = raw_mass.loc[fallback_mask] / 1000.0

    return ensure_weight_pair(out)


def parse_observations(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Observations")
    df = dedupe_columns(df)

    # Keep explicit basal metabolic rate only (drop standard/resting/field/
    # not specified/missing method labels).
    if "metabolic rate - method" not in df.columns:
        raise KeyError(
            "AnimalTraits sheet Observations is missing required column "
            "'metabolic rate - method'."
        )
    mr_method = df["metabolic rate - method"].astype("string").str.strip()
    basal_mask = mr_method.str.casefold() == "basal metabolic rate"
    n_before = len(df)
    df = df.loc[basal_mask].copy().reset_index(drop=True)
    print(
        f"AnimalTraits metabolic rate - method filter: kept {len(df):,}/{n_before:,} "
        f"'basal metabolic rate' rows "
        f"(dropped {n_before - len(df):,} non-basal/missing)."
    )

    out = make_output_frame(len(df))

    genus_col = df["genus"] if "genus" in df.columns else None
    species_col = df["specificEpithet"] if "specificEpithet" in df.columns else None
    full_col = df["species"] if "species" in df.columns else None
    out[GENUS_COL], out[SPECIES_COL] = extract_genus_species(genus_col, species_col, full_col)
    out["class"] = df["class"] if "class" in df.columns else np.nan
    out["order"] = df["order"] if "order" in df.columns else np.nan
    out["family"] = df["family"] if "family" in df.columns else np.nan

    mass_candidates = build_general_mass_candidates(df)
    mass_g, mass_kg, raw_mass = mass_from_candidates(df, mass_candidates)
    out[WET_G_COL] = mass_g
    out[WET_KG_COL] = mass_kg

    mr = (
        numeric(df["metabolic rate"])
        if "metabolic rate" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    mr_unit = (
        df["metabolic rate - units"]
        if "metabolic rate - units" in df.columns
        else pd.Series([pd.NA] * len(df), dtype="string")
    )
    out["BMR"], out["BMR_unit"] = convert_bmr_value_unit_to_w(mr, mr_unit)

    out["temperature"] = (
        parse_temperature_series(df["original temperature"])
        if "original temperature" in df.columns
        else np.nan
    )
    temp_mask = pd.to_numeric(out["temperature"], errors="coerce").notna()
    out["temperature_unit"] = pd.Series("C", index=out.index, dtype="string").where(
        temp_mask, pd.NA
    )
    out["Reference"] = extract_reference_series(df)

    # If mass unit is missing/unknown but mass+BMR exist, default to wet mass in grams.
    fallback_mask = (
        pd.to_numeric(out[WET_G_COL], errors="coerce").isna()
        & pd.to_numeric(out["BMR"], errors="coerce").notna()
        & pd.to_numeric(raw_mass, errors="coerce").notna()
    )
    out.loc[fallback_mask, WET_G_COL] = raw_mass.loc[fallback_mask]
    out.loc[fallback_mask, WET_KG_COL] = raw_mass.loc[fallback_mask] / 1000.0

    return ensure_weight_pair(out)


def main() -> None:
    print("Merge start")
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
        help="Output CSV path (default: <base-dir>/data/cleaning/merged_bmr_mass_temperature.csv).",
    )
    parser.add_argument(
        "--classes-output",
        type=Path,
        default=None,
        help="Unique-class CSV path (default: <base-dir>/data/cleaning/merged_unique_classes.csv).",
    )
    parser.add_argument(
        "--gbif-timeout-seconds",
        type=float,
        default=20.0,
        help="GBIF API timeout in seconds for taxonomy fill.",
    )
    parser.add_argument(
        "--gbif-retries",
        type=int,
        default=3,
        help="GBIF query retries for taxonomy fill.",
    )
    parser.add_argument(
        "--gbif-retry-delay-seconds",
        type=float,
        default=0.35,
        help="Base retry delay for GBIF taxonomy fill.",
    )
    parser.add_argument(
        "--gbif-pause-seconds",
        type=float,
        default=0.01,
        help="Pause between GBIF queries for taxonomy fill.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir if args.base_dir is not None else find_root()
    if args.output is None:
        output_path = base_dir / "data" / "cleaning" / "merged_bmr_mass_temperature.csv"
    else:
        output_path = args.output if args.output.is_absolute() else base_dir / args.output
    if args.classes_output is None:
        classes_output_path = base_dir / "data" / "cleaning" / "merged_unique_classes.csv"
    else:
        classes_output_path = (
            args.classes_output
            if args.classes_output.is_absolute()
            else base_dir / args.classes_output
        )

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
            GENUS_COL,
            SPECIES_COL,
            "class",
            "order",
            "family",
            "BMR_unit",
            "temperature_unit",
            "Reference",
        ],
    )

    # Always keep only usable core rows and remove duplicate records.
    # Multiple observations per Genus+species are intentionally retained.
    merged = drop_incomplete_core_and_deduplicate(merged)

    merged, missing_before, missing_after = fill_missing_taxonomy_with_gbif(
        merged,
        timeout_seconds=args.gbif_timeout_seconds,
        retries=args.gbif_retries,
        retry_delay_seconds=args.gbif_retry_delay_seconds,
        pause_seconds=args.gbif_pause_seconds,
    )
    print(
        "Taxonomy missing before fill: "
        f"class={missing_before['class']}, "
        f"order={missing_before['order']}, "
        f"family={missing_before['family']}"
    )
    print(
        "Taxonomy missing after fill: "
        f"class={missing_after['class']}, "
        f"order={missing_after['order']}, "
        f"family={missing_after['family']}"
    )

    merged = merged.reindex(columns=OUTPUT_COLS)
    classes_df = unique_classes_frame(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")
    classes_df.to_csv(classes_output_path, index=False, encoding="utf-8")

    try:
        saved_path = output_path.relative_to(base_dir)
    except ValueError:
        saved_path = output_path
    try:
        classes_saved_path = classes_output_path.relative_to(base_dir)
    except ValueError:
        classes_saved_path = classes_output_path

    binomial = (
        merged[GENUS_COL].astype("string").str.strip().fillna("")
        + " "
        + merged[SPECIES_COL].astype("string").str.strip().fillna("")
    ).str.strip()
    print(f"Saved: {saved_path}")
    print(f"Saved unique classes: {classes_saved_path}")
    print(f"Rows: {len(merged)}")
    print(f"Unique Genus+species: {int(binomial.replace('', pd.NA).nunique(dropna=True))}")
    print(f"Unique classes: {int(merged['class'].nunique(dropna=True))}")
    print("Non-null counts:")
    for c in [GENUS_COL, SPECIES_COL, WET_KG_COL, "BMR", "temperature"]:
        print(f"  {c}: {int(merged[c].notna().sum())}")
    print(f"BMR units: {sorted(merged['BMR_unit'].dropna().astype(str).unique().tolist())}")


if __name__ == "__main__":
    main()
