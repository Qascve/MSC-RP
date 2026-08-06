#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo import PhyloXML


LOW_COLOR = "#7AD7D3"
MID_COLOR = "#8A8A8A"
HIGH_COLOR = "#F1055B"
MISSING_COLOR = "#D0D0D0"


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def taxon_to_tip_label(taxon_name: str) -> str:
    return str(taxon_name).strip().replace(" ", "_")


def fold_accuracy(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    # Multiplicative accuracy on log10(BMR): 10^(-|pred - true|).
    y_true_arr = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    y_pred_arr = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)

    out = np.full(len(y_true_arr), np.nan, dtype=float)
    out[valid] = 10.0 ** (-np.abs(y_pred_arr[valid] - y_true_arr[valid]))
    return np.clip(out, 0.0, 1.0)


def load_prediction_table(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    required = ["taxon_name", "y_true", "xgboost"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{predictions_path.name} missing required columns: {', '.join(missing)}")

    optional = [col for col in ["class", "order", "family"] if col in df.columns]
    df = df[[*required, *optional]].copy()
    df["tip_label"] = df["taxon_name"].map(taxon_to_tip_label)
    df["accuracy"] = fold_accuracy(df["y_true"], df["xgboost"])
    return df.dropna(subset=["tip_label", "accuracy"]).drop_duplicates("tip_label", keep="last")


def load_tip_accuracy(predictions_path: Path) -> dict[str, float]:
    df = load_prediction_table(predictions_path)
    return df.dropna(subset=["tip_label", "accuracy"]).set_index("tip_label")["accuracy"].to_dict()


def load_tip_records(predictions_path: Path) -> dict[str, dict[str, object]]:
    df = load_prediction_table(predictions_path)
    return df.set_index("tip_label").to_dict(orient="index")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(channel)):02X}" for channel in rgb)


def interpolate_color(left: str, right: str, fraction: float) -> str:
    left_rgb = np.array(hex_to_rgb(left), dtype=float)
    right_rgb = np.array(hex_to_rgb(right), dtype=float)
    mixed = left_rgb + (right_rgb - left_rgb) * fraction
    return rgb_to_hex(tuple(mixed))


def accuracy_to_color(value: float) -> str:
    if not np.isfinite(value):
        return MISSING_COLOR
    value = float(np.clip(value, 0.0, 1.0))
    if value <= 0.5:
        return interpolate_color(LOW_COLOR, MID_COLOR, value / 0.5)
    return interpolate_color(MID_COLOR, HIGH_COLOR, (value - 0.5) / 0.5)


def annotate_clade(clade: PhyloXML.Clade, tip_accuracy: dict[str, float]) -> float:
    if clade.is_terminal():
        accuracy = tip_accuracy.get(clade.name, np.nan)
    else:
        child_values = [annotate_clade(child, tip_accuracy) for child in clade.clades]
        child_values = [value for value in child_values if np.isfinite(value)]
        accuracy = float(np.mean(child_values)) if child_values else np.nan

    color = accuracy_to_color(accuracy)
    clade.color = PhyloXML.BranchColor.from_hex(color)
    if np.isfinite(accuracy):
        clade.properties.append(
            PhyloXML.Property(
                value=f"{accuracy:.6f}",
                ref="msc:prediction_accuracy",
                applies_to="clade",
                datatype="xsd:decimal",
            )
        )
    clade.properties.append(
        PhyloXML.Property(
            value=color,
            ref="msc:prediction_accuracy_color",
            applies_to="clade",
            datatype="xsd:string",
        )
    )
    return accuracy


def prune_to_predicted_tips(tree: Phylo.BaseTree.Tree, tip_accuracy: dict[str, float]) -> int:
    predicted_tips = {name for name, value in tip_accuracy.items() if np.isfinite(value)}
    tree_tips = {clade.name for clade in tree.get_terminals()}
    keep_tips = tree_tips & predicted_tips
    if not keep_tips:
        raise ValueError("No prediction rows matched tree tip labels.")

    for clade in list(tree.get_terminals()):
        if clade.name not in keep_tips:
            tree.prune(clade)
    return len(keep_tips)


def write_species_accuracy_table(
    tree: Phylo.BaseTree.Tree,
    tip_accuracy: dict[str, float],
    output_path: Path,
) -> None:
    rows = [
        {
            "species": clade.name.replace("_", " "),
            "accuracy": tip_accuracy[clade.name],
        }
        for clade in tree.get_terminals()
        if clade.name in tip_accuracy and np.isfinite(tip_accuracy[clade.name])
    ]
    table = pd.DataFrame(rows).sort_values(["accuracy", "species"], ascending=[False, True])
    table.to_csv(output_path, sep="\t", index=False, float_format="%.6f")


def format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.6g}"


def assign_clade_layout(
    tree: Phylo.BaseTree.Tree,
) -> tuple[dict[int, float], dict[int, float], dict[int, int]]:
    terminals = tree.get_terminals()
    terminal_positions = {id(clade): idx for idx, clade in enumerate(terminals)}
    clade_positions: dict[int, float] = {}
    descendant_counts: dict[int, int] = {}

    def assign_position(clade: Phylo.BaseTree.Clade) -> float:
        if clade.is_terminal():
            clade_positions[id(clade)] = float(terminal_positions[id(clade)])
            descendant_counts[id(clade)] = 1
            return clade_positions[id(clade)]

        child_positions = [assign_position(child) for child in clade.clades]
        child_counts = [descendant_counts[id(child)] for child in clade.clades]
        descendant_counts[id(clade)] = int(sum(child_counts))
        clade_positions[id(clade)] = float(np.average(child_positions, weights=child_counts))
        return clade_positions[id(clade)]

    assign_position(tree.root)
    max_distance = max(tree.distance(clade) for clade in tree.find_clades()) or 1.0
    radius_by_clade = {
        id(clade): 55.0 + 390.0 * (tree.distance(clade) / max_distance)
        for clade in tree.find_clades()
    }
    n_tips = max(len(terminals), 1)
    angle_by_clade = {
        clade_id: (2.0 * math.pi * ((position + 0.5) / n_tips)) - (math.pi / 2.0)
        for clade_id, position in clade_positions.items()
    }
    return radius_by_clade, angle_by_clade, descendant_counts


def polar_to_xy(radius: float, angle: float, center: float = 500.0) -> tuple[float, float]:
    return center + radius * math.cos(angle), center + radius * math.sin(angle)


def edge_path(parent_radius: float, parent_angle: float, child_radius: float, child_angle: float) -> str:
    parent_x, parent_y = polar_to_xy(parent_radius, parent_angle)
    arc_x, arc_y = polar_to_xy(parent_radius, child_angle)
    child_x, child_y = polar_to_xy(child_radius, child_angle)
    delta = abs(child_angle - parent_angle)
    large_arc = 1 if delta > math.pi else 0
    sweep = 1 if child_angle >= parent_angle else 0
    return (
        f"M {parent_x:.3f} {parent_y:.3f} "
        f"A {parent_radius:.3f} {parent_radius:.3f} 0 {large_arc} {sweep} {arc_x:.3f} {arc_y:.3f} "
        f"L {child_x:.3f} {child_y:.3f}"
    )


def circular_edge_points(
    parent_radius: float,
    parent_angle: float,
    child_radius: float,
    child_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    arc_steps = max(3, int(abs(child_angle - parent_angle) / (2.0 * math.pi) * 240))
    arc_angles = np.linspace(parent_angle, child_angle, arc_steps)
    arc_x = parent_radius * np.cos(arc_angles)
    arc_y = parent_radius * np.sin(arc_angles)
    radial_r = np.linspace(parent_radius, child_radius, 3)
    radial_x = radial_r * math.cos(child_angle)
    radial_y = radial_r * math.sin(child_angle)
    return np.concatenate([arc_x, radial_x]), np.concatenate([arc_y, radial_y])


def clade_accuracy(clade: Phylo.BaseTree.Clade, tip_accuracy: dict[str, float]) -> float:
    if clade.is_terminal():
        return tip_accuracy.get(clade.name, np.nan)
    values = [clade_accuracy(child, tip_accuracy) for child in clade.clades]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def normalize_search_key(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s_]+", " ", text)
    return text


def clade_tooltip(
    clade: Phylo.BaseTree.Clade,
    accuracy: float,
    descendant_counts: dict[int, int],
    tip_records: dict[str, dict[str, object]],
) -> str:
    if clade.is_terminal():
        record = tip_records.get(clade.name, {})
        lines = [
            f"Species: {str(record.get('taxon_name', clade.name.replace('_', ' ')))}",
            f"Class: {str(record.get('class', 'NA'))}",
            f"Accuracy: {format_number(accuracy)}",
            f"Observed log10(BMR): {format_number(record.get('y_true'))}",
            f"XGB prediction (log10(BMR)): {format_number(record.get('xgboost'))}",
        ]
    else:
        lines = [
            "Internal clade",
            f"Descendant species: {descendant_counts.get(id(clade), 0)}",
            f"Mean accuracy: {format_number(accuracy)}",
        ]
    return "&#10;".join(html.escape(line) for line in lines)


def write_interactive_html(
    tree: Phylo.BaseTree.Tree,
    tip_accuracy: dict[str, float],
    tip_records: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    radius_by_clade, angle_by_clade, descendant_counts = assign_clade_layout(tree)
    clade_accuracy_cache = {id(clade): clade_accuracy(clade, tip_accuracy) for clade in tree.find_clades()}
    paths: list[str] = []
    tip_index: list[dict[str, object]] = []
    legend_x = 1055.0
    legend_y = 430.0

    for parent in tree.find_clades(order="level"):
        for child in parent.clades:
            accuracy = clade_accuracy_cache[id(child)]
            color = accuracy_to_color(accuracy)
            path_data = edge_path(
                radius_by_clade[id(parent)],
                angle_by_clade[id(parent)],
                radius_by_clade[id(child)],
                angle_by_clade[id(child)],
            )
            tooltip = clade_tooltip(child, accuracy, descendant_counts, tip_records)
            if child.is_terminal():
                record = tip_records.get(child.name, {})
                display_name = str(record.get("taxon_name", child.name.replace("_", " ")))
                search_key = normalize_search_key(display_name)
                tip_x, tip_y = polar_to_xy(
                    radius_by_clade[id(child)],
                    angle_by_clade[id(child)],
                )
                tip_index.append(
                    {
                        "key": search_key,
                        "label": display_name,
                        "x": round(tip_x, 3),
                        "y": round(tip_y, 3),
                        "tooltip": tooltip.replace("&#10;", "\n"),
                    }
                )
            paths.append(
                f'<path class="branch" d="{path_data}" stroke="{color}" data-tooltip="{tooltip}"></path>'
            )

    tip_index.sort(key=lambda item: str(item["label"]).lower())
    tip_options = "".join(
        f'<option value="{html.escape(str(item["label"]), quote=True)}"></option>'
        for item in tip_index
    )
    tips_json = json.dumps(tip_index, ensure_ascii=False)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XGB residual-learning accuracy tree</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #ffffff;
      color: #222;
    }}
    .page {{
      width: 1120px;
      margin: 20px auto;
      position: relative;
    }}
    h1 {{
      text-align: center;
      font-size: 22px;
      font-weight: 500;
      margin: 10px 0 0;
    }}
    .search-bar {{
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      margin: 14px 0 8px;
      flex-wrap: wrap;
    }}
    .search-bar input {{
      width: min(420px, 72vw);
      padding: 8px 10px;
      border: 1px solid #bbb;
      border-radius: 6px;
      font-size: 14px;
    }}
    .search-bar button {{
      padding: 8px 14px;
      border: 1px solid #888;
      border-radius: 6px;
      background: #f7f7f7;
      font-size: 14px;
      cursor: pointer;
    }}
    .search-bar button:hover {{
      background: #ececec;
    }}
    # search-status {{
      width: 100%;
      text-align: center;
      font-size: 13px;
      color: #666;
      min-height: 18px;
    }}
    svg {{
      display: block;
      margin: 0 auto;
    }}
    .branch {{
      fill: none;
      stroke-width: 1.05;
      stroke-linecap: round;
      opacity: 0.86;
      cursor: default;
      transition: stroke-width 0.15s ease, opacity 0.15s ease;
    }}
    .branch:hover {{
      stroke-width: 3.2;
      opacity: 1;
    }}
    .tip-focus-ring {{
      fill: none;
      stroke: rgba(241, 5, 91, 0.82);
      stroke-width: 2.4;
      pointer-events: none;
      visibility: hidden;
    }}
    # tooltip {{
      position: fixed;
      display: none;
      pointer-events: none;
      max-width: 300px;
      padding: 9px 11px;
      border: 1px solid #bbb;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.16);
      white-space: pre-line;
      font-size: 13px;
      line-height: 1.35;
      z-index: 10;
    }}
    .legend-title {{
      font-size: 16px;
    }}
    .legend-label {{
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>XGB residual-learning accuracy across test-set species</h1>
    <div class="search-bar">
      <input id="species-search" type="search" list="species-options"
             placeholder="Search species, e.g. Rhinella marina" autocomplete="off">
      <datalist id="species-options">
        {tip_options}
      </datalist>
      <button id="search-go" type="button">Go</button>
      <button id="search-reset" type="button">Reset view</button>
    </div>
    <div id="search-status"></div>
    <svg id="tree-svg" width="1120" height="1040" viewBox="0 0 1120 1040" role="img">
      <defs>
        <linearGradient id="accuracy-gradient" x1="0" x2="0" y1="1" y2="0">
          <stop offset="0%" stop-color="{LOW_COLOR}"/>
          <stop offset="50%" stop-color="{MID_COLOR}"/>
          <stop offset="100%" stop-color="{HIGH_COLOR}"/>
        </linearGradient>
      </defs>
      <g id="tree-stage" transform="translate(60, 20) scale(1)">
        {''.join(paths)}
        <circle id="tip-focus-ring" class="tip-focus-ring" cx="0" cy="0" r="18"></circle>
      </g>
      <g transform="translate({legend_x:.0f}, {legend_y:.0f})">
        <text class="legend-title" x="0" y="-18">Prediction</text>
        <text class="legend-title" x="0" y="0">accuracy</text>
        <rect x="0" y="16" width="26" height="150" fill="url(#accuracy-gradient)"></rect>
        <text class="legend-label" x="36" y="20">1.00</text>
        <text class="legend-label" x="36" y="58">0.75</text>
        <text class="legend-label" x="36" y="96">0.50</text>
        <text class="legend-label" x="36" y="134">0.25</text>
        <text class="legend-label" x="36" y="170">0.00</text>
      </g>
    </svg>
    <div id="tooltip"></div>
  </div>
  <script>
    const TIPS = {tips_json};
    const tooltip = document.getElementById("tooltip");
    const searchInput = document.getElementById("species-search");
    const searchStatus = document.getElementById("search-status");
    const treeStage = document.getElementById("tree-stage");
    const tipFocusRing = document.getElementById("tip-focus-ring");
    const defaultTransform = "translate(60, 20) scale(1)";
    const focusScale = 2.35;
    let pinnedTipKey = null;

    function getSvgMetrics() {{
      const svg = document.getElementById("tree-svg");
      const rect = svg.getBoundingClientRect();
      return {{
        rect,
        scaleX: rect.width / 1120,
        scaleY: rect.height / 1040,
      }};
    }}

    function getScreenCenter() {{
      return {{
        x: window.innerWidth / 2,
        y: window.innerHeight / 2,
      }};
    }}

    function normalizeSearchKey(value) {{
      return String(value || "")
        .trim()
        .toLowerCase()
        .replace(/[\\s_]+/g, " ");
    }}

    function showTooltip(text, clientX, clientY) {{
      tooltip.textContent = text;
      tooltip.style.display = "block";
      tooltip.style.left = `${{clientX + 14}}px`;
      tooltip.style.top = `${{clientY + 14}}px`;
    }}

    function hideTooltip() {{
      if (!pinnedTipKey) {{
        tooltip.style.display = "none";
      }}
    }}

    function focusTreeOnTip(tipX, tipY) {{
      const {{ rect, scaleX, scaleY }} = getSvgMetrics();
      const center = getScreenCenter();
      const tx = (center.x - rect.left) / scaleX - tipX * focusScale;
      const ty = (center.y - rect.top) / scaleY - tipY * focusScale;
      treeStage.setAttribute(
        "transform",
        `translate(${{tx.toFixed(3)}}, ${{ty.toFixed(3)}}) scale(${{focusScale}})`
      );
    }}

    function showTipFocusRing(tipX, tipY) {{
      tipFocusRing.setAttribute("cx", tipX.toFixed(3));
      tipFocusRing.setAttribute("cy", tipY.toFixed(3));
      tipFocusRing.style.visibility = "visible";
    }}

    function hideTipFocusRing() {{
      tipFocusRing.style.visibility = "hidden";
    }}

    function resetTreeView() {{
      treeStage.setAttribute("transform", defaultTransform);
      pinnedTipKey = null;
      hideTipFocusRing();
      searchStatus.textContent = "";
      tooltip.style.display = "none";
    }}

    function resolveTip(query) {{
      const key = normalizeSearchKey(query);
      if (!key) {{
        return null;
      }}
      const exact = TIPS.find((tip) => tip.key === key);
      if (exact) {{
        return exact;
      }}
      const prefixMatches = TIPS.filter((tip) => tip.key.startsWith(key));
      if (prefixMatches.length === 1) {{
        return prefixMatches[0];
      }}
      const containsMatches = TIPS.filter((tip) => tip.key.includes(key));
      if (containsMatches.length === 1) {{
        return containsMatches[0];
      }}
      return null;
    }}

    function jumpToSpecies(query) {{
      const tip = resolveTip(query);
      if (!tip) {{
        pinnedTipKey = null;
        hideTipFocusRing();
        tooltip.style.display = "none";
        searchStatus.textContent = "No matching species found.";
        return false;
      }}

      pinnedTipKey = tip.key;
      focusTreeOnTip(tip.x, tip.y);
      showTipFocusRing(tip.x, tip.y);
      searchInput.value = tip.label;
      searchStatus.textContent = `Focused on ${{tip.label}}`;

      const center = getScreenCenter();
      showTooltip(tip.tooltip, center.x, center.y);
      return true;
    }}

    document.querySelectorAll(".branch").forEach((branch) => {{
      branch.addEventListener("mouseenter", (event) => {{
        showTooltip(branch.dataset.tooltip, event.clientX, event.clientY);
      }});
      branch.addEventListener("mousemove", (event) => {{
        showTooltip(branch.dataset.tooltip, event.clientX, event.clientY);
      }});
      branch.addEventListener("mouseleave", () => {{
        hideTooltip();
      }});
    }});

    document.getElementById("search-go").addEventListener("click", () => {{
      jumpToSpecies(searchInput.value);
    }});
    document.getElementById("search-reset").addEventListener("click", resetTreeView);
    searchInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        event.preventDefault();
        jumpToSpecies(searchInput.value);
      }}
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def write_static_pdf(
    tree: Phylo.BaseTree.Tree,
    tip_accuracy: dict[str, float],
    output_path: Path,
) -> None:
    radius_by_clade, angle_by_clade, _descendant_counts = assign_clade_layout(tree)
    clade_accuracy_cache = {id(clade): clade_accuracy(clade, tip_accuracy) for clade in tree.find_clades()}

    fig, ax = plt.subplots(figsize=(10, 10))
    for parent in tree.find_clades(order="level"):
        for child in parent.clades:
            accuracy = clade_accuracy_cache[id(child)]
            color = accuracy_to_color(accuracy)
            x_values, y_values = circular_edge_points(
                radius_by_clade[id(parent)],
                angle_by_clade[id(parent)],
                radius_by_clade[id(child)],
                angle_by_clade[id(child)],
            )
            ax.plot(x_values, y_values, color=color, linewidth=0.55, alpha=0.88)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "prediction_accuracy",
        [LOW_COLOR, MID_COLOR, HIGH_COLOR],
    )
    sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01)
    cbar.set_label("Prediction\naccuracy", rotation=0, labelpad=18)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_title("XGB residual-learning accuracy across test-set species", fontsize=14, pad=18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    fig.tight_layout()
    cbar.ax.set_position([0.935, 0.24, 0.018, 0.52])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description="Export XGB residual-learning prediction accuracy as an annotated PhyloXML tree for iTOL."
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path("data/phylogeny/unique_taxon_names.nwk"),
        help="Input Newick tree.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/benchmark/all/test/benchmark_predictions_test.csv"),
        help="Residual-learning prediction CSV containing taxon_name, y_true, and xgboost columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/plots/xgb_residual_phylogeny_accuracy.phyloxml"),
        help="Output PhyloXML file for iTOL.",
    )
    parser.add_argument(
        "--species-output",
        type=Path,
        default=Path("results/plots/xgb_residual_phylogeny_accuracy_species.txt"),
        help="Output tab-delimited species accuracy table sorted by descending accuracy.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("results/plots/xgb_residual_phylogeny_accuracy.html"),
        help="Output standalone interactive HTML with hover tooltips.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("results/plots/xgb_residual_phylogeny_accuracy.pdf"),
        help="Output static PDF of the circular accuracy tree.",
    )
    args = parser.parse_args()

    tree_path = resolve_path(root, args.tree)
    predictions_path = resolve_path(root, args.predictions)
    output_path = resolve_path(root, args.output)
    species_output_path = resolve_path(root, args.species_output)
    html_output_path = resolve_path(root, args.html_output)
    pdf_output_path = resolve_path(root, args.pdf_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    species_output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

    tip_accuracy = load_tip_accuracy(predictions_path)
    tip_records = load_tip_records(predictions_path)
    tree = Phylo.read(str(tree_path), "newick")
    matched_tips = prune_to_predicted_tips(tree, tip_accuracy)
    write_species_accuracy_table(tree, tip_accuracy, species_output_path)
    write_interactive_html(tree, tip_accuracy, tip_records, html_output_path)
    write_static_pdf(tree, tip_accuracy, pdf_output_path)
    phylogeny = PhyloXML.Phylogeny.from_tree(tree)
    annotate_clade(phylogeny.root, tip_accuracy)

    Phylo.write(phylogeny, str(output_path), "phyloxml")
    print(f"Exported test-set tips: {matched_tips}")
    print(f"Saved PhyloXML: {output_path}")
    print(f"Saved species accuracy table: {species_output_path}")
    print(f"Saved interactive HTML: {html_output_path}")
    print(f"Saved static PDF: {pdf_output_path}")


if __name__ == "__main__":
    main()
