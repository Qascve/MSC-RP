# MSC-RP Workflow Guide

## 1. Project Goal

This repository builds a reproducible workflow for basal metabolic rate (BMR) analysis. It merges raw BMR, body mass, temperature, taxonomy, and phylogeny data; creates a species-blocked development/test partition; and compares physics-inspired MTE models, tree-based residual-learning models, and phylogenetic PGLS.

All modeling targets use `log_BMR = log10(BMR)`.

Current processed dataset (`data/merge_phylo.csv`):
- Observations: 5,077
- Species: 1,594
- Classes: 8 (`Teleostei`, `Mammalia`, `Aves`, `Insecta`, `Malacostraca`, `Amphibia`, `Reptilia`, `Cephalopoda`)

Current species-blocked split (`data/splits/`):
- Development (F1∪F2∪F3∪F4): 4,063 rows / 1,277 species
- Holdout test (T): 1,014 rows / 317 species
- Largest classes in the holdout test: `Teleostei` 554, `Mammalia` 284, `Aves` 72

## 2. Environment Setup

Install Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

The PGLS step is run in R and additionally requires:

```r
install.packages(c("ape", "nlme", "phytools"))
```

Notes:
- `filter_target_classes.py` and `merge_bmr_mass_temp.py` may require internet access for taxonomy lookups through GBIF/pytaxon.
- All commands below are intended to be run from the project root.

## 3. Data Preparation Pipeline

### Step 1: Merge Raw Datasets

Script: `code/merge_bmr_mass_temp.py`

Inputs:
- `data/raw/pnas.2303764120.sd01.xlsx`
- `data/raw/observations.xlsx`
- `data/raw/41586_2010_BFnature08920_MOESM90_ESM.xls`

Output:
- `data/cleaning/merged_bmr_mass_temperature.csv`

```bash
python code/merge_bmr_mass_temp.py
```

This step unifies the raw schemas, removes invalid records and duplicates, and fills missing taxonomy fields where possible.

### Step 2: Standardize Taxa and Filter Classes

Script: `code/filter_target_classes.py`

Inputs:
- `data/cleaning/merged_bmr_mass_temperature.csv`
- `code/config.json`

Outputs:
- `data/cleaning/standard_data.csv`
- `data/cleaning/filtered_data.csv`

```bash
python code/filter_target_classes.py
```

This step standardizes species names, creates `taxon_name`, removes excluded classes, and applies a final taxonomy safety filter.

### Step 3: Export Species Names

Script: `code/export_taxon_names.py`

Input:
- `data/cleaning/filtered_data.csv`

Output:
- `data/phylogeny/unique_taxon_names.txt`

```bash
python code/export_taxon_names.py
```

The exported species list is used for phylogenetic tree matching.

### Step 4: Build Phylogenetic Embeddings

Script: `code/phylogeny.py`

Inputs:
- `data/phylogeny/unique_taxon_names.nwk`
- `data/phylogeny/unique_taxon_names.txt`

Outputs:
- `data/phylogeny/phylogenetic_embeddings.csv`
- `data/phylogeny/phylogeny_matched_species.csv`

```bash
python code/phylogeny.py
```

This step matches and prunes tree tips, computes patristic distances, and converts the distance matrix into `PC1`-`PC5` phylogenetic embedding features.

### Step 5: Merge Observations and Embeddings

Script: `code/merge_phylo_embedding.py`

Inputs:
- `data/cleaning/filtered_data.csv`
- `data/phylogeny/phylogenetic_embeddings.csv`

Output:
- `data/merge_phylo.csv`

```bash
python code/merge_phylo_embedding.py
```

This joins cleaned observations with `pc1`-`pc5` by `taxon_name`.

### Step 6: Create Species-Blocked Folds and Holdout

Script: `code/split_train_test_bmr.py`

Input:
- `data/merge_phylo.csv`

Outputs:
- `data/splits/fold1/` … `data/splits/fold4/` (`train.csv`, `test.csv`)
- `data/splits/test/` (`train.csv` = F1∪F2∪F3∪F4, `test.csv` = holdout T)
- `data/splits/train.csv` (alias of the 80% development set)
- `data/splits/class_species_block_split_summary.csv`
- `data/splits/class_weights.csv`

```bash
python code/split_train_test_bmr.py
```

Per taxonomic class, each species (`taxon_name`) is assigned wholly to one of F1–F4 or T (~20% each). Fold usage:
- Fold `i` (HP/CV): train = other three development folds (~60%), eval = Fi (~20%)
- Test holdout: train = F1∪F2∪F3∪F4 (~80%), eval = T (~20%)

This step also derives `log_mass`, `log_BMR` (`log10`), and `inv_kT`.

## 4. Model and Result Pipeline

### Step 7: Residual-Learning Benchmark

Script: `code/ml_residual_learning.py`

Input:
- `data/splits/` (fixed fold1–fold4 + test)

Output directory:
- `results/benchmark/`

```bash
python code/ml_residual_learning.py
```

The benchmark fits residual Random Forest and XGBoost on top of a fixed M3-L baseline (`log_BMR ~ log_mass + inv_kT + class`). Hyperparameters are tuned with 4-fold species-block CV on the development set, then models are refit on the full 80% development set and evaluated once on the holdout test. Training uses class-balanced sample weights.

It reports the full test set plus per-class subsets:
- `all`
- `Teleostei`, `Mammalia`, `Aves`, `Insecta`, `Malacostraca`, `Amphibia`, `Reptilia`, `Cephalopoda`

Key outputs (per group):
- `results/benchmark/<group>/cv/` — OOF CV metrics and predictions
- `results/benchmark/<group>/test/` — holdout metrics, predictions, SHAP plots
- `results/benchmark/<group>/species_accuracy.csv`
- `results/benchmark/all/test/models/` — saved RF/XGB joblib models
- `results/benchmark/xgb_best_params.csv`
- `results/benchmark/class_species_block_split_summary.csv`

Current best holdout result on the full test set (`log10(BMR)`):
- Model: `xgboost`
- RMSE: 0.3327
- MAE: 0.2332
- R2: 0.9511

### Step 8: M1-M4 Tree-Based ML Comparison

Script: `code/explore_ml.py`

Inputs:
- `data/splits/test/train.csv`
- `data/splits/test/test.csv`

Output directory:
- `results/explore/test/`

```bash
python code/explore_ml.py
```

This step compares Random Forest and XGBoost under M1–M4 feature settings:
- m1: `log_mass`
- m2: `log_mass + inv_kT`
- m3: `log_mass + inv_kT + class`
- m4: `log_mass + inv_kT + pc1`–`pc5`

Key outputs:
- `results/explore/test/explore_ml_metrics.csv`
- `results/explore/test/explore_ml_predictions_test.csv`
- `results/explore/explore_ml_species_accuracy.csv`

Current best model from this comparison:
- Model: `random_forest_m3`
- RMSE: 0.3884
- MAE: 0.2754
- R2: 0.9333

### Step 9: PGLS Model Comparison with `ape` and `nlme`

Script: `code/pgls_ape.R`

Inputs:
- `data/splits/test/train.csv`
- `data/splits/test/test.csv`
- `data/phylogeny/unique_taxon_names.nwk`

Output directory:
- `results/pgls_ape/test/` (when invoked by `explore.py`; standalone runs may write under `results/pgls_ape/`)

```bash
Rscript code/pgls_ape.R
```

This step fits PGLS models with the fixed-effects formula `log_BMR ~ log_mass + inv_kT` and compares phylogenetic correlation structures by AIC:
- Pagel lambda
- Brownian
- Martins
- Blomberg
- Grafen

Key outputs:
- `pgls_aic_scores.csv`
- `pgls_test_predictions.csv`
- `pgls_train_fitted.csv`
- `pgls_test_metrics.csv`
- `pgls_best_model_summary.txt`

Current holdout PGLS metrics (`log10(BMR)`, n=1,014):
- Best correlation structure: `pglsModel_Brownian`
- RMSE: 0.6956
- MAE: 0.5894
- R2: 0.7863

### Step 10: Integrated MTE, PGLS, and ML Comparison

Script: `code/explore.py`

Inputs:
- `data/splits/`
- residual-learning predictions from `results/benchmark/all/`
- explore_ml predictions from `results/explore/`
- PGLS via `code/pgls_ape.R` (called automatically if needed)

Output directory:
- `results/explore/`

```bash
python code/explore.py
```

This step fits linear MTE-style models (`M0-L` to `M4-L`), imports tree-based M1–M4 and residual-learning predictions, runs PGLS as `M4-PGLS`, and compares all models on the same species-block holdout test set.

Key outputs:
- `results/explore/test/explore_metrics.csv`
- `results/explore/test/top5_plus_residual_learning_metrics.csv`
- `results/explore/test/model_performance_comparison.pdf`
- `results/explore/test/residual_plot_all_models.pdf`
- `results/explore/explore_species_accuracy.csv`

Current top integrated holdout results (`log10(BMR)`):
- `Residual-XGB`: RMSE 0.3327, MAE 0.2332, R2 0.9511
- `Residual-RF`: RMSE 0.3355, MAE 0.2404, R2 0.9503
- `M3-L`: RMSE 0.3469, MAE 0.2460, R2 0.9468
- `M3-RF`: RMSE 0.3884, MAE 0.2754, R2 0.9333
- `M4-PGLS`: RMSE 0.6956, MAE 0.5894, R2 0.7863

### Step 11: Slope Estimates and Comparison Plots

Script: `code/plot_slope_estimates.py`

Inputs:
- `data/splits/train.csv`
- `data/splits/test/test.csv`
- optional explore_ml / residual-learning prediction CSVs

Output directory:
- `results/plots/`

```bash
python code/plot_slope_estimates.py
```

This estimates the mass-scaling exponent `b` (with 95% CI) by class and writes comparison figures for linear M1–M4 and ML/residual models:
- `results/plots/slope_estimates.pdf` / `.csv`
- `results/plots/m1_m4_linear_comparison.pdf` / `.csv`
- `results/plots/ml_residual_comparison.pdf` / `.csv`

### Optional Step: Block Cross-validation

Script: `code/block_cv.py`

Input:
- `data/merge_phylo.csv`

Default split output directory:
- `data/splits/block_cv/`

Default summary output directory:
- `results/block_cv/`

```bash
python code/block_cv.py
```

This creates two Roberts-style blocked validation datasets without changing the existing fold/holdout split.

Dataset 1: `fair_all`
- Purpose: all classes appear in training and testing, while species are kept as complete blocks.
- Bias control: residual-learning models use class-balanced sample weights, and reports include micro, macro-class, capped weighted macro-class, per-class, and per-block metrics.

Dataset 2: `leave_class_out`
- Purpose: train without one target class, then predict that held-out class.
- Groups:
  - `A`: train without `Teleostei`, predict `Teleostei`.
  - `B`: train without `Mammalia`, predict `Mammalia`.
  - `C`: train without `Insecta`, predict `Insecta`.

Use `--skip-models` to only write train/test CSV files.

## 5. Utility Scripts

### `code/generate_dataset_overview.py`

Summarize observation/species/class counts from the current splits and write:
- `results/dataset_overview/dataset_overview.txt`
- `results/dataset_overview/dataset_class_summary.csv`
- pie-chart figures under `results/dataset_overview/`

### `code/generate_data_compilation_report.py`

Tabulate species/class counts across cleaning pipeline stages into:
- `results/data_compilation/data_flow_table.csv`

### `code/export_xgb_residual_accuracy.py`

Map Residual-XGB species-level accuracy onto the phylogeny and export PhyloXML / HTML / PNG assets under `results/plots/`.

### `code/low_accuracy_class_summary.py`

Summarize low-accuracy species fractions by class for residual-learning predictions.

### `code/class_distribution.py`

Summarize class distributions for one or more CSV files.

Default output:
- `data/cleaning/class_distribution.csv`

Example:

```bash
python code/class_distribution.py --input data/cleaning/standard_data.csv
```

### `code/bmr_models.py`

Standalone experiment script (avg/nonleaky splits × lm/RF/XGB × m1–m4). Not part of the main reproducible pipeline; may require extra packages (`dendropy`, `scipy`).

## 6. Minimal Reproducible Command List

```bash
python code/merge_bmr_mass_temp.py
python code/filter_target_classes.py
python code/export_taxon_names.py
python code/phylogeny.py
python code/merge_phylo_embedding.py
python code/split_train_test_bmr.py
python code/ml_residual_learning.py
python code/explore_ml.py
python code/explore.py
python code/plot_slope_estimates.py
```

Optional overview helpers:

```bash
python code/generate_dataset_overview.py
python code/generate_data_compilation_report.py
```
