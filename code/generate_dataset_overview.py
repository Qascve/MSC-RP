#!/usr/bin/env python3
# Generate PPT-ready summary assets for the current fixed-split dataset.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def load_current_dataset(split_dir: Path) -> pd.DataFrame:
    train_path = split_dir / "test" / "train.csv"
    test_path = split_dir / "test" / "test.csv"
    missing = [str(path) for path in [train_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing current fixed-split files: " + ", ".join(missing))

    frames = [pd.read_csv(train_path), pd.read_csv(test_path)]
    data = pd.concat(frames, ignore_index=True)
    required = {"class", "taxon_name"}
    absent = required.difference(data.columns)
    if absent:
        raise KeyError(f"Missing required columns: {', '.join(sorted(absent))}")

    data["class"] = data["class"].astype("string").str.strip()
    data["taxon_name"] = data["taxon_name"].astype("string").str.strip()
    return data.dropna(subset=["class", "taxon_name"]).reset_index(drop=True)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    total = len(data)
    summary = (
        data.groupby("class", sort=False)
        .agg(species_count=("taxon_name", "nunique"), observations=("class", "size"))
        .reset_index()
        .sort_values(["observations", "class"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary["share_percent"] = summary["observations"] / total * 100
    return summary


def autopct_for_large_slices(values: pd.Series):
    percentages = values / values.sum() * 100
    index = iter(percentages)

    def format_slice(_: float) -> str:
        percentage = next(index)
        return f"{percentage:.1f}%" if percentage >= 2 else ""

    return format_slice


def draw_pie(summary: pd.DataFrame, total_species: int, output_path: Path) -> None:
    colors = list(plt.colormaps["tab20"].colors)
    if len(summary) > len(colors):
        colors = [plt.colormaps["turbo"](i / len(summary)) for i in range(len(summary))]

    fig, ax = plt.subplots(figsize=(13.333, 7.5), facecolor="white")
    wedges, _, autotexts = ax.pie(
        summary["observations"],
        colors=colors[: len(summary)],
        startangle=90,
        counterclock=False,
        labels=None,
        autopct=autopct_for_large_slices(summary["observations"]),
        pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 8, "color": "black", "weight": "bold"},
    )
    for text in autotexts:
        text.set_path_effects([])

    legend_labels = [
        f"{row['class']}  {int(row['observations']):,} ({row['share_percent']:.2f}%)"
        for _, row in summary.iterrows()
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Class · observations",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=20,
        title_fontsize=20,
        labelspacing=0.65,
        handlelength=1.2,
    )
    ax.set_title("Observations by clade", fontsize=25, weight="bold", pad=18)
    ax.text(
        0.5,
        0.02,
        f"Total: {summary['observations'].sum():,} observations · "
        f"{total_species:,} species · {len(summary)} classes",
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=25,
        color="#404040",
    )
    ax.axis("equal")
    fig.subplots_adjust(left=0.03, right=0.72, top=0.88, bottom=0.08)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_overview"))
    args = parser.parse_args()

    split_dir = args.split_dir if args.split_dir.is_absolute() else root / args.split_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_current_dataset(split_dir)
    summary = build_summary(data)
    total_species = int(data["taxon_name"].nunique())

    summary_path = output_dir / "dataset_class_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    chart_path = output_dir / "class_observation_share_pie.pdf"
    draw_pie(summary, total_species, chart_path)

    overview_path = output_dir / "dataset_overview.txt"
    lines = [
        f"Observations: {len(data):,}",
        f"Species: {total_species:,}",
        f"Classes: {data['class'].nunique():,}",
        "",
        summary.to_string(index=False, formatters={"share_percent": lambda x: f"{x:.2f}%"}),
    ]
    overview_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Observations: {len(data):,}")
    print(f"Species: {total_species:,}")
    print(f"Classes: {data['class'].nunique():,}")
    print(summary.to_string(index=False, formatters={"share_percent": lambda x: f"{x:.2f}%"}))
    print(f"Saved: {summary_path}")
    print(f"Saved: {chart_path}")
    print(f"Saved: {overview_path}")


if __name__ == "__main__":
    main()
