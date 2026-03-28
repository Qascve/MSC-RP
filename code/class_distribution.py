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


def normalize_class_series(df: pd.DataFrame, class_col: str) -> pd.Series:
    if class_col not in df.columns:
        raise KeyError(f"Missing class column: {class_col}")
    cls = df[class_col].astype("string").str.strip()
    cls = cls.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return cls


def summarize_single_csv(path: Path, class_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    total_rows = len(df)
    cls = normalize_class_series(df, class_col)
    non_null_rows = int(cls.notna().sum())
    counts = cls.dropna().value_counts()

    rows: list[dict[str, object]] = []
    for class_name, count in counts.items():
        rows.append(
            {
                "source_file": path.name,
                "class": str(class_name),
                "rows": int(count),
                "ratio_in_total": float(count / total_rows) if total_rows > 0 else 0.0,
            }
        )

    missing_count = total_rows - non_null_rows
    rows.append(
        {
            "source_file": path.name,
            "class": "(missing)",
            "rows": int(missing_count),
            "ratio_in_total": float(missing_count / total_rows) if total_rows > 0 else 0.0,
        }
    )

    out = pd.DataFrame(rows)
    return out.sort_values(by=["source_file", "rows", "class"], ascending=[True, False, True])


def summarize_multi_csv(paths: list[Path], class_col: str) -> pd.DataFrame:
    parts = [summarize_single_csv(path, class_col) for path in paths]
    if not parts:
        return pd.DataFrame(
            columns=[
                "source_file",
                "class",
                "rows",
                "ratio_in_total",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description="Compute class distribution for one or more CSV files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[Path("data/cleaning/standard_data.csv")],
        help="One or more input CSV paths.",
    )
    parser.add_argument(
        "--class-col",
        type=str,
        default="class",
        help="Class column name (default: class).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Only print first N rows (0 means print all).",
    )
    args = parser.parse_args()

    csv_paths: list[Path] = []
    for p in args.inputs:
        rp = p if p.is_absolute() else root / p
        if not rp.exists():
            raise FileNotFoundError(f"CSV not found: {rp}")
        csv_paths.append(rp)
    summary = summarize_multi_csv(csv_paths, class_col=args.class_col)

    out_path = root / "data" / "cleaning" / "class_distribution.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    if args.head > 0:
        summary = summary.head(args.head)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
