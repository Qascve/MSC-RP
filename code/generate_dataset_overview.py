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


def draw_pie(summary: pd.DataFrame, output_path: Path) -> None:
    # Match the sorted-class tab10 mapping used in slope_estimates Panel B.
    class_levels = sorted(summary["class"].astype(str).unique())
    palette = list(plt.colormaps["tab10"].colors)
    class_colors = {
        class_name: palette[index]
        for index, class_name in enumerate(class_levels)
    }
    colors = [class_colors[str(class_name)] for class_name in summary["class"]]

    fig, ax = plt.subplots(figsize=(25, 12), facecolor="white")
    wedges, _ = ax.pie(
        summary["observations"],
        colors=colors,
        startangle=90,
        counterclock=False,
        labels=None,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )

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
        fontsize=28,
        title_fontsize=28,
        labelspacing=0.65,
        handlelength=1.2,
    )
    ax.axis("equal")
    fig.subplots_adjust(left=0.03, right=0.72, top=0.98, bottom=0.02)
    fig.savefig(output_path, facecolor="white")
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
    draw_pie(summary, chart_path)

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
