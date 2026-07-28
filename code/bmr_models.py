"""
BMR scaling models: lm / XGB / RF x m1-m4, evaluated under TWO data
pipelines and THREE train/test splitting strategies each.

=============================================================================
PIPELINE A — "avg"  (one row per species, averaged)
=============================================================================
Many species have multiple BMR measurements. If individual rows were split
into train/test, the same species could appear in both, and a model could
"memorize" a species' intercept rather than generalize. Instead of doing
row-level splits with group-aware folding, the dataset is first collapsed to
ONE ROW PER SPECIES: log_mass, log_BMR, and inv_kT are averaged across a
species' repeated measurements; taxonomy and PCA coordinates are already
constant within a species. With exactly one row per species, any ordinary
split is automatically leak-free.

=============================================================================
PIPELINE B — "nonleaky"  (all rows kept, species-grouped splits)
=============================================================================
All rows are kept so a model can learn from within-species replicate
variation -- but every split strategy assigns a whole SPECIES to a single
fold, so no species ever appears in both train and test.

=============================================================================
THE THREE SPLITTING STRATEGIES (all 5-fold, species-level)
=============================================================================
Density and clustering are both computed directly on the TREE's patristic
(cophenetic) distance matrix -- the actual phylogenetic distance -- rather
than on the PCA embedding space, which is only an approximation of it.

1. RANDOM     - species randomly shuffled into 5 folds. Baseline.
2. DENSITY    - Gaussian-kernel density from patristic distances
                (bandwidth = median pairwise distance). Fold 0 = rarest /
                most phylogenetically isolated species.
3. CLUSTERING - Ward hierarchical clustering on the condensed patristic
                distance matrix; each cluster = one fold (entire clade held
                out). Hardest / most extrapolative split.

=============================================================================
THE FOUR MODELS
=============================================================================
m1: log_BMR ~ log_mass
m2: log_BMR ~ log_mass + inv_kT
m3: log_BMR ~ (log_mass + inv_kT) * clade        [clade = taxonomic class]
m4: log_BMR ~ (log_mass + inv_kT) * phylogeny    [phylogeny = 5 PCA axes]

Each is fit with three algorithms (LinearRegression, XGBRegressor,
RandomForestRegressor) on the SAME design matrix.
"""

import itertools
import numpy as np
import pandas as pd
import dendropy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

RNG = 0
N_FOLDS = 5
BOLTZMANN_EV = 8.617333262e-5  # eV / K

# =============================================================================
# 1. LOAD + FEATURE ENGINEERING
#    Returns BOTH the full row-level df AND the species-averaged df so that
#    each pipeline can pick the representation it needs.
# =============================================================================

def load_data(trait_path, embed_path):
    """
    Returns
    -------
    df_full : pd.DataFrame
        One row per BMR measurement (used by the nonleaky pipeline).
    df_avg  : pd.DataFrame
        One row per species, numeric columns averaged (used by the avg pipeline).
    pc_cols : list[str]
    """
    traits = pd.read_csv(trait_path)
    embed  = pd.read_csv(embed_path).drop_duplicates(subset="taxon_name")

    # Restrict to species that have a phylogenetic embedding so all four
    # models are evaluated on the same set of species (fair comparison).
    df = traits.merge(embed, on="taxon_name", how="inner")

    df["log_mass"] = np.log10(df["wet_Mass_kg"])
    df["log_BMR"]  = np.log10(df["BMR"])
    df["inv_kT"]   = 1.0 / (BOLTZMANN_EV * (df["temperature"] + 273.15))

    pc_cols = [c for c in df.columns if c.startswith("PC")]

    # Drop rare classes (<7 unique species) -- singleton/near-singleton
    # classes add noise without enough data to estimate clade-specific effects.
    # Count by unique species (not rows) so the threshold is consistent.
    species_per_class = df.groupby("class")["taxon_name"].transform("nunique")
    df = df[species_per_class >= 7].reset_index(drop=True)
    df["clade"] = df["class"]

    # ---- Pipeline A: collapse to one row per species --------------------
    agg = {"log_mass": "mean", "log_BMR": "mean", "inv_kT": "mean", "class": "first", "clade": "first"}
    agg.update({pc: "first" for pc in pc_cols})
    df_avg = df.groupby("taxon_name", as_index=False).agg(agg).reset_index(drop=True)

    return df, df_avg, pc_cols


# =============================================================================
# 2. PHYLOGENETIC DISTANCE MATRIX  (shared by both pipelines)
# =============================================================================

def build_distance_matrix(tree_path, taxon_names):
    """Patristic (cophenetic) distance matrix between `taxon_names`, in that
    exact order, read off the phylogenetic tree."""
    tree = dendropy.Tree.get(path=tree_path, schema="newick")
    pdm  = tree.phylogenetic_distance_matrix()
    label_to_taxon = {t.label: t for t in tree.taxon_namespace}

    taxa = [label_to_taxon[name] for name in taxon_names]
    n = len(taxa)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = pdm.patristic_distance(taxa[i], taxa[j])
            D[i, j] = D[j, i] = d
    return D


# =============================================================================
# 3. FOLD ASSIGNMENT  (species-level; both pipelines share the same functions)
#
#    Every function returns a pd.Series  taxon_name -> fold_id.
#    The avg pipeline uses it directly (one row = one species).
#    The nonleaky pipeline maps it onto the full row-level df via taxon_name.
# =============================================================================

def folds_random(taxon_names, dist_matrix, n_folds=N_FOLDS, seed=RNG):
    rng  = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(taxon_names))
    return pd.Series(fold, index=taxon_names)


def folds_density(taxon_names, dist_matrix, n_folds=N_FOLDS):
    bandwidth = np.median(dist_matrix[dist_matrix > 0])
    density   = np.exp(-(dist_matrix ** 2) / (2 * bandwidth ** 2)).sum(axis=1)
    fold      = pd.qcut(density, n_folds, labels=False, duplicates="drop")
    return pd.Series(fold, index=taxon_names)


def folds_cluster(taxon_names, dist_matrix, n_folds=N_FOLDS):
    condensed = squareform(dist_matrix, checks=False)
    Z    = linkage(condensed, method="ward")
    fold = fcluster(Z, t=n_folds, criterion="maxclust")
    return pd.Series(fold, index=taxon_names)


SPLIT_STRATEGIES = {
    "random":   folds_random,
    "density":  folds_density,
    "cluster":  folds_cluster,
}


# =============================================================================
# 4. DESIGN MATRIX  (identical for both pipelines)
# =============================================================================

def build_design_matrix(df, model_name, pc_cols, algo_name, ref_columns=None):
    X = pd.DataFrame(index=df.index)
    X["log_mass"] = df["log_mass"]

    if model_name in ("m2", "m3", "m4"):
        X["inv_kT"] = df["inv_kT"]

    if model_name == "m3":
        dummies = pd.get_dummies(df["clade"], prefix="clade", drop_first=True)
        X = pd.concat([X, dummies], axis=1)
        if algo_name == "lm":
            for c in dummies.columns:
                X[f"log_mass_x_{c}"] = X["log_mass"] * X[c]
                X[f"invkT_x_{c}"]    = X["inv_kT"]   * X[c]

    if model_name == "m4":
        for pc in pc_cols:
            X[pc] = df[pc]
        if algo_name == "lm":
            for pc in pc_cols:
                X[f"log_mass_x_{pc}"] = X["log_mass"] * df[pc]
                X[f"invkT_x_{pc}"]    = X["inv_kT"]   * df[pc]

    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)
    return X


# =============================================================================
# 5. ALGORITHMS + HYPERPARAMETER GRIDS
# =============================================================================

MODELS = ["m1", "m2", "m3", "m4"]

PARAM_GRIDS = {
    "lm": {},
    "xgb": {
        "n_estimators":  [200, 300],
        "max_depth":     [3, 4, 6],
        "learning_rate": [0.05, 0.1],
    },
    "rf": {
        "n_estimators":    [300],
        "max_depth":       [6, 10, None],
        "min_samples_leaf": [1, 3],
    },
}


def expand_grid(grid):
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def make_algo(algo_name, params):
    if algo_name == "lm":
        return LinearRegression()
    if algo_name == "xgb":
        return XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=RNG, **params)
    if algo_name == "rf":
        return RandomForestRegressor(random_state=RNG, n_jobs=-1, **params)
    raise ValueError(f"Unknown algo: {algo_name}")


# =============================================================================
# 6. CROSS-VALIDATED EVALUATION
#
#    fold_map : pd.Series  taxon_name -> fold_id
#    df       : the dataframe to split (avg df or full df depending on pipeline)
#
#    For the avg pipeline  : df has one row per species; fold_map maps directly.
#    For the nonleaky pipeline: df has many rows per species; fold_map is
#    broadcast to rows via taxon_name before splitting.
#
#    Returns train AND test R²/RMSE (mean ± SD across folds) so the results
#    table exposes overfitting directly: a large train-test gap on a given
#    algo/model/split combination signals memorisation rather than
#    generalisation.
# =============================================================================

def evaluate(df, pc_cols, fold_map, model_name, algo_name, params):
    train_r2s,   test_r2s   = [], []
    train_rmses, test_rmses = [], []

    # Map species-level fold ids down to individual rows.
    # For the avg pipeline this is a 1-to-1 mapping; for nonleaky it fans out.
    fold_of_row = df["taxon_name"].map(fold_map)

    for k in sorted(fold_map.unique()):
        train = df[fold_of_row != k]
        test  = df[fold_of_row == k]
        if len(test) == 0 or len(train) == 0:
            continue

        X_train = build_design_matrix(train, model_name, pc_cols, algo_name)
        X_test  = build_design_matrix(test,  model_name, pc_cols, algo_name,
                                      ref_columns=X_train.columns)
        y_train = train["log_BMR"].values
        y_test  = test["log_BMR"].values

        model = make_algo(algo_name, params)
        model.fit(X_train, y_train)

        # ---- train performance (fitted values on the training fold) ----
        train_pred = model.predict(X_train)
        train_r2s.append(r2_score(y_train, train_pred))
        train_rmses.append(np.sqrt(mean_squared_error(y_train, train_pred)))

        # ---- test performance (predictions on the held-out fold) -------
        test_pred = model.predict(X_test)
        test_r2s.append(r2_score(y_test, test_pred))
        test_rmses.append(np.sqrt(mean_squared_error(y_test, test_pred)))

    return (
        np.mean(train_r2s),   np.std(train_r2s),
        np.mean(train_rmses), np.std(train_rmses),
        np.mean(test_r2s),    np.std(test_r2s),
        np.mean(test_rmses),  np.std(test_rmses),
    )


# =============================================================================
# 7. RUN ONE PIPELINE  (called twice from main)
# =============================================================================

def run_pipeline(pipeline_name, df, taxon_names, dist_matrix, pc_cols):
    """
    Parameters
    ----------
    pipeline_name : "avg" | "nonleaky"
    df            : the dataframe for this pipeline (df_avg or df_full)
    taxon_names   : ordered list of species names matching dist_matrix rows
    dist_matrix   : pre-built patristic distance matrix
    pc_cols       : list of PC column names

    Returns
    -------
    list of result dicts, each tagged with pipeline=pipeline_name
    """
    results = []
    for split_name, split_fn in SPLIT_STRATEGIES.items():
        fold_map = split_fn(taxon_names, dist_matrix)

        for model_name in MODELS:
            for algo_name, grid in PARAM_GRIDS.items():
                for params in expand_grid(grid):
                    (
                        train_r2_mean,   train_r2_sd,
                        train_rmse_mean, train_rmse_sd,
                        test_r2_mean,    test_r2_sd,
                        test_rmse_mean,  test_rmse_sd,
                    ) = evaluate(df, pc_cols, fold_map, model_name, algo_name, params)

                    row = dict(
                        pipeline=pipeline_name,
                        split=split_name,
                        model=model_name,
                        algo=algo_name,
                        params=str(params),
                        # ---- train ----
                        train_R2_mean=train_r2_mean,
                        train_R2_sd=train_r2_sd,
                        train_RMSE_mean=train_rmse_mean,
                        train_RMSE_sd=train_rmse_sd,
                        # ---- test ----
                        test_R2_mean=test_r2_mean,
                        test_R2_sd=test_r2_sd,
                        test_RMSE_mean=test_rmse_mean,
                        test_RMSE_sd=test_rmse_sd,
                    )
                    results.append(row)
                    print(
                        f"[{pipeline_name:8s}|{split_name:8s}] "
                        f"{model_name} {algo_name:4s} {params}\n"
                        f"  train  R2={train_r2_mean:.3f}±{train_r2_sd:.3f}"
                        f"  RMSE={train_rmse_mean:.3f}±{train_rmse_sd:.3f}\n"
                        f"  test   R2={test_r2_mean:.3f}±{test_r2_sd:.3f}"
                        f"  RMSE={test_rmse_mean:.3f}±{test_rmse_sd:.3f}"
                    )
    return results


# =============================================================================
# 8. MAIN
# =============================================================================

def main(
    trait_path="../data/cleaning/filtered_data.csv",
    embed_path="../data/phylogeny/phylogenetic_embeddings.csv",
    tree_path="../data/phylogeny/unique_taxon_names.nwk",
    out_path="../results/model_results_combined.csv",
):
    # ------------------------------------------------------------------
    # Load data -- returns full row-level df AND species-averaged df
    # ------------------------------------------------------------------
    df_full, df_avg, pc_cols = load_data(trait_path, embed_path)

    # Both pipelines operate on the same set of species, so the distance
    # matrix is built once from the averaged species list and reused.
    # (df_avg has exactly one row per species by construction.)
    taxon_names = df_avg["taxon_name"].tolist()

    print(f"Species: {len(taxon_names)}  |  Total rows (nonleaky): {len(df_full)}")
    print("Building phylogenetic distance matrix …")
    dist_matrix = build_distance_matrix(tree_path, taxon_names)
    print("Distance matrix ready.\n")

    # ------------------------------------------------------------------
    # Pipeline A: averaged (one row per species)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PIPELINE: avg  (one row per species, averaged)")
    print("=" * 70)
    results_avg = run_pipeline("avg", df_avg, taxon_names, dist_matrix, pc_cols)

    # ------------------------------------------------------------------
    # Pipeline B: nonleaky (all rows, species-grouped splits)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE: nonleaky  (all rows, species-grouped splits)")
    print("=" * 70)
    results_nonleaky = run_pipeline("nonleaky", df_full, taxon_names, dist_matrix, pc_cols)

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results_avg + results_nonleaky)

    # Enforce a logical column order: identifiers -> train metrics -> test metrics
    col_order = [
        "pipeline", "split", "model", "algo", "params",
        "train_R2_mean", "train_R2_sd", "train_RMSE_mean", "train_RMSE_sd",
        "test_R2_mean",  "test_R2_sd",  "test_RMSE_mean",  "test_RMSE_sd",
    ]
    results_df = results_df[col_order]

    results_df.to_csv(out_path, index=False)
    print(f"\nSaved combined results -> {out_path}")
    return results_df


if __name__ == "__main__":
    main()