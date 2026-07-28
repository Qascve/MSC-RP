#!/usr/bin/env python3
"""
Generate §5.1 data-compilation artefacts:

1. data_flow_table.csv — records / unique species / classes after each major stage
2. data_compilation_report.txt — precise notes on sources, variables, BMR
   definitions, units, taxonomy harmonisation, duplicates, missing values,
   and retention of multiple observations per species
"""

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


def binomial_series(df: pd.DataFrame) -> pd.Series:
    if "taxon_name" in df.columns:
        names = df["taxon_name"].astype("string").str.strip()
        if names.notna().any() and (names != "").any():
            return names

    genus = df["Genus"].astype("string").str.strip() if "Genus" in df.columns else ""
    species = df["species"].astype("string").str.strip() if "species" in df.columns else ""
    return (genus.fillna("") + " " + species.fillna("")).str.strip().replace({"": pd.NA})


def stage_counts(df: pd.DataFrame) -> dict[str, int]:
    species = binomial_series(df).dropna()
    species = species[species != ""]
    if "class" in df.columns:
        classes = (
            df["class"]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
            .dropna()
        )
        n_classes = int(classes.nunique())
    else:
        n_classes = 0
    return {
        "records": int(len(df)),
        "unique_species": int(species.nunique()),
        "classes": n_classes,
    }


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_modeling_frame(split_dir: Path) -> pd.DataFrame:
    train_path = split_dir / "test" / "train.csv"
    test_path = split_dir / "test" / "test.csv"
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing fixed-split files: " + ", ".join(missing))
    return pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)


def raw_source_rows(root: Path) -> list[dict[str, object]]:
    """Optional raw-sheet row counts (before core filtering)."""
    rows: list[dict[str, object]] = []
    sources = [
        (
            "01_raw_pnas_hoehler",
            root / "data" / "raw" / "pnas.2303764120.sd01.xlsx",
            "Metabolic_Data",
        ),
        (
            "02_raw_animaltraits",
            root / "data" / "raw" / "observations.xlsx",
            "Observations",
        ),
    ]
    for stage, path, sheet in sources:
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        class_col = "Class" if "Class" in df.columns else ("class" if "class" in df.columns else None)
        if class_col is not None:
            n_classes = int(
                df[class_col].astype("string").str.strip().replace({"": pd.NA}).dropna().nunique()
            )
        else:
            n_classes = 0
        # Raw sheets do not share a single binomial field; report row count only for species.
        rows.append(
            {
                "stage": stage,
                "records": int(len(df)),
                "unique_species": pd.NA,
                "classes": n_classes if n_classes else pd.NA,
                "notes": f"Raw sheet '{sheet}' row count before parse/filter",
            }
        )

    mcnab_path = root / "data" / "raw" / "41586_2010_BFnature08920_MOESM90_ESM.xls"
    if mcnab_path.exists():
        # Header is auto-detected in merge_bmr_mass_temp; approximate with sheet rows minus notes.
        raw = pd.read_excel(mcnab_path, sheet_name="McNab 2008 Edited.csv", header=None)
        rows.append(
            {
                "stage": "03_raw_mcnab",
                "records": int(len(raw)),
                "unique_species": pd.NA,
                "classes": pd.NA,
                "notes": "Raw sheet rows including citation header block; mammals only",
            }
        )
    return rows


def build_data_flow(root: Path, include_raw: bool) -> pd.DataFrame:
    stages: list[dict[str, object]] = []

    if include_raw:
        stages.extend(raw_source_rows(root))

    pipeline = [
        (
            "04_after_merge_core_filter_dedup",
            root / "data" / "cleaning" / "merged_bmr_mass_temperature.csv",
            "Merged three sources; required Genus/species/mass/BMR/temperature; exact-row dedup; GBIF taxonomy fill",
        ),
        (
            "05_after_class_whitelist_taxon_standardise",
            root / "data" / "cleaning" / "filtered_data.csv",
            "Restricted to animal KEEP_CLASSES; NCBI/pytaxon taxon_name added",
        ),
        (
            "06_after_phylogeny_embedding_join",
            root / "data" / "merge_phylo.csv",
            "Inner join to phylogenetic PC1-PC5 embeddings by taxon_name",
        ),
        (
            "07_after_modeling_filters_min7_species",
            None,
            "Positive mass/BMR; valid T; drop classes with <7 species; train+test fixed split",
        ),
    ]

    for stage, path, notes in pipeline:
        if stage.endswith("min7_species"):
            df = load_modeling_frame(root / "data" / "splits")
        else:
            assert path is not None
            df = load_csv(path)
        counts = stage_counts(df)
        stages.append(
            {
                "stage": stage,
                "records": counts["records"],
                "unique_species": counts["unique_species"],
                "classes": counts["classes"],
                "notes": notes,
            }
        )

    return pd.DataFrame(stages)


def format_flow_markdown(flow: pd.DataFrame) -> str:
    lines = [
        "| stage | records | unique_species | classes |",
        "|---|---:|---:|---:|",
    ]
    for _, row in flow.iterrows():
        def fmt(v: object) -> str:
            if pd.isna(v):
                return "—"
            return f"{int(v):,}"

        lines.append(
            f"| {row['stage']} | {fmt(row['records'])} | "
            f"{fmt(row['unique_species'])} | {fmt(row['classes'])} |"
        )
    return "\n".join(lines)


def build_report_text(flow: pd.DataFrame) -> str:
    flow_md = format_flow_markdown(flow)
    final = flow.loc[flow["stage"].str.contains("modeling_filters")].iloc[-1]

    return f"""5.1 Data compilation and harmonisation
======================================

This report documents the compilation pipeline implemented in:
  code/merge_bmr_mass_temp.py
  code/filter_target_classes.py
  code/merge_phylo_embedding.py
  code/split_train_test_bmr.py

Final modelling frame (train + test):
  {int(final['records']):,} records · {int(final['unique_species']):,} unique species · {int(final['classes'])} classes


1. Source databases and taxonomic scope
---------------------------------------
Three source compilations were merged:

  (i) Hoehler et al. (2023), PNAS supplementary metabolic data
      File: data/raw/pnas.2303764120.sd01.xlsx (sheet Metabolic_Data)
      DOI: 10.1073/pnas.2303764120
      Scope: cross-domain metabolic rates spanning Bacteria, Archaea and
      Eukaryota (animals, plants, protists, etc.). Class field is provided.

  (ii) AnimalTraits (Herberstein et al. / Sci Data)
      File: data/raw/observations.xlsx (sheet Observations)
      DOI: 10.1038/s41597-022-01364-9
      Scope: animals (e.g. Insecta, Mammalia, Aves, Amphibia, Reptilia,
      Arachnida, Malacostraca, and related invertebrate classes).

  (iii) McNab mammalian BMR compilation (redistributed via Nature 2010 supplement)
      File: data/raw/41586_2010_BFnature08920_MOESM90_ESM.xls
           (sheet "McNab 2008 Edited.csv")
      Primary citation: McNab (2008) Comp. Biochem. Physiol. A
      Scope: mammals only. Class was missing in the source and was filled
      later via GBIF name backbone lookup.


2. Variables extracted
----------------------
Each source was mapped onto a common observation schema:

  Taxonomy: class, order, family, Genus, species
  Traits:   wet_Mass_kg (and wet_Mass_g), BMR, BMR_unit,
            temperature, temperature_unit
  Provenance: Reference

Later pipeline stages additionally add:
  taxon_name (NCBI/pytaxon-standardised binomial)
  pc1–pc5 (phylogenetic embeddings)
  log_mass, log_BMR, inv_kT (modelling transforms)


3. Biological definition of “BMR” in each dataset
-------------------------------------------------
The unified column is named BMR. Source inclusion rules:

  Hoehler / PNAS
    Only rows with Type of Metabolic Rate == "Basal" are retained
    (Field / Maximum / Dark respiration / Endogenous / etc. are dropped).
    Metabolic rate is taken preferentially from "Metabolic Rate (W, at T)"
    paired with "T (C)". When that pair is incomplete, "Metabolic Rate
    (W, at 25C)" is used and temperature is set to 25 C.

  AnimalTraits
    Only rows with metabolic rate - method == "basal metabolic rate" are
    retained (standard / resting / field / not specified / missing are
    dropped). Metabolic rate is kept only when metabolic rate - units is W;
    non-W rows are treated as missing.

  McNab
    Explicit mammalian basal metabolic rate (BMR) compilation, already in W.


4. Original and final units, with conversion steps
--------------------------------------------------
  Body mass
    Original: kg / g / mg (unit inferred from source unit columns or column
              names such as "Wet Mass (g)", "Mass (g)", "body mass - units").
    Final:    wet_Mass_kg (also stored as wet_Mass_g).
    Steps:    kg → g×1000; g unchanged; mg → g/1000; then kg = g/1000.
              If wet/dry status was unspecified but a numeric mass and BMR
              existed, unresolved mass was treated as wet mass in grams.

  Metabolic rate
    Original accepted unit: W only (watt / watts).
    Final:    BMR in watts; BMR_unit = "W".
    Steps:    No conversion from mW, kW, J/s, kJ/h, kJ/s, O2 volume rates,
              or other energy/time units — those rows are treated as missing.
              PNAS prefers "Metabolic Rate (W, at T)" with matching "T (C)";
              otherwise uses "Metabolic Rate (W, at 25C)" with temperature=25.

  Temperature
    Original: °C numerics, simple ranges (e.g. 25-35), placeholders ENDO/ND.
    Final:    temperature in °C; temperature_unit = "C".
    Steps:    ranges → midpoint; ENDO/ND/NA → missing (row dropped later).


5. Harmonisation of species names and taxonomic synonyms
--------------------------------------------------------
  (a) Genus and species strings were taken from source binomials and were not
      overwritten during merge.
  (b) Missing class / order / family were filled with GBIF
      species.name_backbone lookups (critical for McNab, which lacked class).
  (c) NCBI standardisation via pytaxon (source_id=4) produced taxon_name
      (fuzzy uninomial match enabled; mainTaxonThreshold=0.6). Genus/species
      columns were retained unchanged; taxon_name is the join key for
      phylogeny.
  (d) Observations were restricted to a fixed animal-class whitelist
      (KEEP_CLASSES in filter_target_classes.py).


6. Identification of duplicate records (within and across databases)
--------------------------------------------------------------------
After concatenation, exact duplicates were removed on the biologically
equivalent key:

  Genus, species, class, order, family,
  wet_Mass_kg, BMR, BMR_unit, temperature, temperature_unit, Reference

keep="first". There is no source-priority rule (e.g. McNab over PNAS).
Cross-database duplicates are removed only when all key fields match.
Same-species records that differ in mass, temperature, rate or reference
are retained as distinct observations.
Phylogenetic embeddings were de-duplicated on taxon_name before the join.


7. Handling of missing values
-----------------------------
  Merge / core filter
    Rows lacking valid Genus, species, wet_Mass_kg, BMR (+ unit), or
    temperature were dropped. Placeholder taxon tokens (NA, unknown, sp.,
    bare numbers, etc.) were rejected.

  Taxonomy
    GBIF fill attempted for missing ranks; rows still without an allowed
    class were removed at the whitelist step.

  Phylogeny
    Inner join on taxon_name: species without tree embeddings were dropped.

  Modelling frame
    Required BASE columns must be non-missing; mass>0; BMR>0;
    (T+273.15)>0; classes with fewer than 7 unique species were dropped.


8. Retention of multiple observations per species
-------------------------------------------------
Multiple observations for the same Genus+species (and later taxon_name) were
intentionally retained. No per-species record-count limit was applied at
compilation. Within-species replicates capture variation in body mass,
measurement temperature and study conditions that enter the MTE predictors.
Leakage control for prediction uses species-block splits (entire species
assigned to train or test), rather than collapsing to one row per species
during cleaning.


Data-flow table
---------------
Species are the blocking unit in the prediction task, so unique_species is
reported alongside records and classes after each major filter.

{flow_md}

Outputs written by this script:
  results/data_compilation/data_flow_table.csv
  results/data_compilation/data_compilation_report.txt
"""


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_compilation"),
        help="Directory for CSV and report outputs",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        default=True,
        help="Include raw Excel sheet row counts (default: on)",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip reading raw Excel files; only use cleaned CSVs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    include_raw = bool(args.include_raw) and not bool(args.no_raw)
    flow = build_data_flow(root, include_raw=include_raw)
    report = build_report_text(flow)

    csv_path = output_dir / "data_flow_table.csv"
    txt_path = output_dir / "data_compilation_report.txt"
    flow.to_csv(csv_path, index=False, encoding="utf-8-sig")
    txt_path.write_text(report, encoding="utf-8")

    print(flow.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
