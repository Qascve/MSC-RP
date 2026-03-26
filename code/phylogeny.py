#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from io import StringIO
from multiprocessing import Manager
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from Bio import Phylo
from sklearn.decomposition import PCA

_WORKER_TREE = None
_WORKER_SPECIES: list[str] | None = None


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def normalize_tip_like_r_sub(name: str) -> str:
    # Match R sub("_", " ", x): only replace first underscore.
    return name.replace("_", " ", 1).strip()


def format_seconds(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _progress_line(
    processed_pairs: int, total_pairs: int, start: float, bar_width: int = 30
) -> str:
    now = time.perf_counter()
    elapsed = now - start
    pct = (processed_pairs / total_pairs * 100.0) if total_pairs else 100.0
    speed = (processed_pairs / elapsed) if elapsed > 0 else 0.0
    remaining = (total_pairs - processed_pairs) / speed if speed > 0 else 0.0
    filled = int(bar_width * processed_pairs / total_pairs) if total_pairs else bar_width
    bar = "#" * filled + "-" * (bar_width - filled)
    return (
        f"Distance progress [{bar}] {pct:6.2f}% "
        f"({processed_pairs}/{total_pairs}) "
        f"elapsed {format_seconds(elapsed)} | ETA {format_seconds(remaining)}"
    )


def _render_status(line: str, prev_len: int) -> int:
    out = line
    if len(out) < prev_len:
        out += " " * (prev_len - len(out))
    sys.stdout.write("\r" + out)
    sys.stdout.flush()
    return len(line)


def _worker_init(tree_newick: str, species_order: list[str]) -> None:
    global _WORKER_TREE, _WORKER_SPECIES
    _WORKER_TREE = Phylo.read(StringIO(tree_newick), "newick")
    for tip in _WORKER_TREE.get_terminals():
        if tip.name:
            tip.name = normalize_tip_like_r_sub(tip.name)

    if isinstance(species_order, str):
        raise TypeError("species_order must be a sequence of species names, got string.")
    _WORKER_SPECIES = [
        normalize_tip_like_r_sub(str(s)) for s in list(species_order) if str(s).strip() != ""
    ]
    tree_tip_set = {tip.name for tip in _WORKER_TREE.get_terminals() if tip.name}
    missing = [s for s in _WORKER_SPECIES if s not in tree_tip_set]
    if missing:
        preview = ", ".join(repr(s) for s in missing[:5])
        more = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5} more)"
        raise ValueError(
            f"Worker species not found in worker tree after normalization: {preview}{more}"
        )


def _compute_row_block(
    start_i: int, end_i: int, task_id: int, progress_dict
) -> tuple[int, int, np.ndarray, int]:
    if _WORKER_TREE is None or _WORKER_SPECIES is None:
        raise RuntimeError("Worker not initialized")
    n = len(_WORKER_SPECIES)
    total_pairs = sum(n - i for i in range(start_i, end_i))
    done_pairs = 0
    progress_dict[task_id] = (done_pairs, total_pairs)
    block = np.zeros((end_i - start_i, n), dtype=float)
    for offset, i in enumerate(range(start_i, end_i)):
        sp_i = _WORKER_SPECIES[i]
        for j in range(i, n):
            block[offset, j] = float(_WORKER_TREE.distance(sp_i, _WORKER_SPECIES[j]))
        done_pairs += (n - i)
        progress_dict[task_id] = (done_pairs, total_pairs)
    return start_i, end_i, block, task_id


def build_species_match_table(tree, csv_species: pd.Series) -> pd.DataFrame:
    # Tree side: keep first occurrence order after normalization.
    tree_ordered_unique = list(
        dict.fromkeys(
            normalize_tip_like_r_sub(t.name)
            for t in tree.get_terminals()
            if t.name and normalize_tip_like_r_sub(t.name)
        )
    )
    # CSV side: unique species only, no row-level duplicates.
    csv_ordered_unique_raw = list(
        dict.fromkeys(
            s.strip()
            for s in csv_species.astype("string").dropna().tolist()
            if str(s).strip() != ""
        )
    )
    csv_norm_to_raw: dict[str, str] = {}
    for raw in csv_ordered_unique_raw:
        norm = normalize_tip_like_r_sub(raw)
        if norm and norm not in csv_norm_to_raw:
            csv_norm_to_raw[norm] = raw

    rows: list[dict[str, str]] = []
    for idx, tree_norm in enumerate(tree_ordered_unique, start=1):
        if tree_norm in csv_norm_to_raw:
            rows.append(
                {
                    "match_id": idx,
                    "species_normalized": tree_norm,
                    "tree_species": tree_norm,
                    "csv_species": csv_norm_to_raw[tree_norm],
                }
            )
    return pd.DataFrame(rows)


def build_patristic_distance_matrix(
    tree, species_order: list[str], n_jobs: int = 1, chunk_rows: int = 24
) -> pd.DataFrame:
    n = len(species_order)
    total_pairs = n * (n + 1) // 2
    mat = np.zeros((n, n), dtype=float)
    start = time.perf_counter()
    processed_pairs = 0
    last_update = start
    status_len = 0

    if n_jobs <= 1:
        for i in range(n):
            for j in range(i, n):
                d = float(tree.distance(species_order[i], species_order[j]))
                mat[i, j] = d
                mat[j, i] = d
                processed_pairs += 1
            now = time.perf_counter()
            if (now - last_update) >= 0.5 or i == n - 1:
                last_update = now
                status_len = _render_status(
                    _progress_line(processed_pairs, total_pairs, start), status_len
                )
    else:
        if chunk_rows < 1:
            chunk_rows = 1
        handle = StringIO()
        Phylo.write([tree], handle, "newick")
        tree_newick = handle.getvalue()
        blocks: list[tuple[int, int]] = []
        for s in range(0, n, chunk_rows):
            e = min(s + chunk_rows, n)
            blocks.append((s, e))
        block_pairs = {
            (s, e): sum(n - i for i in range(s, e))
            for s, e in blocks
        }
        with Manager() as manager:
            progress_dict = manager.dict()
            with ProcessPoolExecutor(
                max_workers=n_jobs, initializer=_worker_init, initargs=(tree_newick, species_order)
            ) as executor:
                future_to_meta = {}
                for task_id, (s, e) in enumerate(blocks):
                    fut = executor.submit(_compute_row_block, s, e, task_id, progress_dict)
                    future_to_meta[fut] = (s, e, task_id)
                pending = set(future_to_meta.keys())
                while pending:
                    done_set, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        s, e, task_id = future_to_meta[fut]
                        s2, e2, block, task_id2 = fut.result()
                        if (s, e, task_id) != (s2, e2, task_id2):
                            raise RuntimeError("Parallel task metadata mismatch")
                        mat[s:e, :] = block
                        for i in range(s, e):
                            mat[:i, i] = mat[i, :i]
                        processed_pairs += block_pairs[(s, e)]
                    now = time.perf_counter()
                    if (now - last_update) >= 0.2 or not pending:
                        last_update = now
                        snapshot = dict(progress_dict)
                        live_processed = min(total_pairs, sum(done for done, _ in snapshot.values()))
                        line = _progress_line(live_processed, total_pairs, start)
                        status_len = _render_status(line, status_len)

    status_len = _render_status(_progress_line(processed_pairs, total_pairs, start), status_len)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return pd.DataFrame(mat, index=species_order, columns=species_order)


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description="Python version of 04_C_extract_phylogeny.R (DeepPhylo-style embedding)."
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path("data/unique_species.nwk"),
        help="Path to Newick tree (.nwk).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/merged_bmr_mass_temperature.csv"),
        help="Path to CSV with Species column.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/phylogenetic_embeddings.csv"),
        help="Output embeddings CSV path.",
    )
    parser.add_argument(
        "--matched-species-out",
        type=Path,
        default=Path("data/phylogeny_matched_species.csv"),
        help="Output CSV path for matched unique species between tree and data.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of PCA components (default: 10).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel worker count for distance calculation (default: 4).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=100,
        help="Rows per parallel block (default: 100).",
    )
    args = parser.parse_args()

    # Resolve paths from project root so script works from any cwd.
    tree_path = args.tree if args.tree.is_absolute() else root / args.tree
    data_path = args.data if args.data.is_absolute() else root / args.data
    out_path = args.out if args.out.is_absolute() else root / args.out
    matched_species_out = (
        args.matched_species_out
        if args.matched_species_out.is_absolute()
        else root / args.matched_species_out
    )

    if not tree_path.exists():
        raise FileNotFoundError(f"Newick tree file not found: {tree_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV file not found: {data_path}")

    tree = Phylo.read(str(tree_path), "newick")
    df = pd.read_csv(data_path)
    if "Species" not in df.columns:
        raise KeyError("Missing required column: Species")

    # Fix tip labels like R code: sub("_", " ", tip.label)
    for tip in tree.get_terminals():
        if tip.name:
            tip.name = normalize_tip_like_r_sub(tip.name)

    # Build unique-species match table (tree <-> CSV) after name normalization.
    match_df = build_species_match_table(tree, df["Species"])
    if match_df.empty:
        raise ValueError("No matched species found between tree and CSV after normalization.")
    matched_species_out.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(matched_species_out, index=False, encoding="utf-8")

    # Compute only on matched unique species from the match table.
    species_order = match_df["species_normalized"].tolist()
    if len(species_order) < 2:
        raise ValueError(f"Need at least 2 matched species, got {len(species_order)}.")

    # Dist matrix and reorder to dataframe species order.
    dist_matrix = build_patristic_distance_matrix(
        tree, species_order, n_jobs=args.n_jobs, chunk_rows=args.chunk_rows
    )

    # PCA on distance matrix (center=True, scale=False equivalent).
    n_components = min(args.n_components, dist_matrix.shape[0], dist_matrix.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full")
    embedding_values = pca.fit_transform(dist_matrix.to_numpy())

    embeddings = pd.DataFrame(
        embedding_values,
        index=species_order,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(out_path, index=True, encoding="utf-8")

    print("Phylogenetic embeddings created:")
    print(f"  Dimensions: {embeddings.shape[0]} x {embeddings.shape[1]}")
    print(f"  Matched species used: {len(species_order)}")
    print(f"  Matched species CSV: {matched_species_out}")
    print(f"  Variance explained: {float(np.sum(pca.explained_variance_ratio_)):.6f}")


if __name__ == "__main__":
    print(f"Project root: {find_root()}")
    print(f"Tree path: {Path('data/unique_species.nwk')}")
    print(f"phylogeny.py started")
    main()
    print(f"phylogeny.py finished")
