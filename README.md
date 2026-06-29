# MSC-RP Workflow Guide

## 1. Project Goal

This repository builds a reproducible workflow for the basal metabolic rate (BMR) analysis. It merges raw BMR, body mass, temperature, taxonomy, and phylogeny data; creates stratified train/test splits; and compares physics-inspired MTE models, tree-based residual-learning models, and a phylogenetic mixed model.

Current processed split:
- Total rows after cleaning and phylogeny merge: 5,672
- Train rows: 3,970
- Test rows: 1,702
- Largest classes in the test split: `Teleostei` 1,190, `Mammalia` 140, `Insecta` 78

## 2. Environment Setup

Install Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

The PGLMM step is run in R and additionally requires:

```r
install.packages(c("ape", "phyr"))
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

### Step 6: Create Stratified Train/Test Split

Script: `code/split_train_test_bmr.py`

Input:
- `data/merge_phylo.csv`

Outputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`
- `data/splits/stratified/class_split_summary.csv`

```bash
python code/split_train_test_bmr.py
```

This step applies final row filters, derives `log_mass`, `log_BMR`, and `inv_kT`, then creates a class-stratified split.

## 4. Model and Result Pipeline

### Step 7: Residual-Learning Benchmark

Script: `code/ml_residual_learning.py`

Input:
- `data/merge_phylo.csv`

Output directory:
- `results/benchmark/`

```bash
python code/ml_residual_learning.py
```

The benchmark fits residual Random Forest and XGBoost models on top of a fixed three-quarter power-law baseline. Following Roberts et al. for structured data, it now creates a class-level species-block validation split: within each class, complete species are assigned either to train or test, so the same `taxon_name` cannot appear in both. Training uses class-balanced sample weights to reduce domination by large classes.

It trains on the all-class species-block training set and reports selected class subsets from the species-block test set:
- `Teleostei`
- `Mammalia`
- `Insecta`

Key outputs:
- `data/splits/train.csv`
- `data/splits/test.csv`
- `data/splits/class_species_block_split_summary.csv`
- `results/benchmark/*/benchmark_metrics.csv`
- `results/benchmark/*/shap_feature_importance.csv`
- `results/benchmark/*/shap_summary_bar.png`
- `results/benchmark/*/shap_summary_beeswarm.png`
- `results/benchmark/benchmark_summary_groups.csv`
- `results/benchmark/class_species_block_split_summary.csv`

Current best benchmark result on the full test set:
- Model: `xgboost`
- RMSE: 1.0325
- MAE: 0.1391
- R2: 0.9459

### Step 8: M0-M4 Tree-Based ML Comparison

Script: `code/explore_ml.py`

Inputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`

Output directory:
- `results/explore/`

```bash
python code/explore_ml.py
```

This step compares Random Forest and XGBoost under M0-M4 feature settings and writes:
- `results/explore/explore_ml_metrics.csv`
- `results/explore/explore_ml_predictions_test.csv`
- `results/explore/explore_ml_model_performance_comparison.png`
- `results/explore/explore_ml_residual_plot.png`

Current best model from this comparison:
- Model: `xgboost_m3`
- RMSE: 1.7923
- MAE: 0.1994
- R2: 0.8370

### Step 9: PGLMM with `phyr`

Script: `code/pglmm_phyr.R`

Inputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`
- `data/phylogeny/unique_taxon_names.nwk`
- `data/phylogeny/phylogenetic_embeddings.csv`

Output directory:
- `results/pglmm_phyr/`

```bash
Rscript code/pglmm_phyr.R
```

This step fits a phylogenetic mixed model with `phyr::pglmm`, removes species under non-positive or missing tree branches, predicts the test set where possible, and writes model summaries and diagnostics.

Key outputs:
- `results/pglmm_phyr/pglmm_test_predictions.csv`
- `results/pglmm_phyr/pglmm_train_fitted.csv`
- `results/pglmm_phyr/pglmm_fixed_effects.csv`
- `results/pglmm_phyr/pglmm_random_variance.csv`
- `results/pglmm_phyr/pglmm_test_metrics.csv`
- `results/pglmm_phyr/pglmm_metrics_summary.txt`
- `results/pglmm_phyr/pglmm_model_summary.txt`
- `results/pglmm_phyr/removed_bad_branch_species.csv`

Current BMR-scale PGLMM test metrics:
- Predicted test rows: 438
- RMSE: 2.3559
- MAE: 0.5066
- R2: -0.0010

### Step 10: Integrated MTE, PGLMM, and ML Comparison

Script: `code/explore.py`

Inputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`
- `results/explore/explore_ml_predictions_test.csv`
- `results/benchmark/all/benchmark_predictions_test.csv`
- `results/pglmm_phyr/pglmm_test_predictions.csv`

Output directory:
- `results/explore/`

```bash
python code/explore.py
```

This step fits linear MTE-style models (`M0-L` to `M3-L`), incorporates the PGLMM result as `M4-L`, imports M0-M4 ML predictions, and compares them with the residual-learning benchmark.

Key outputs:
- `results/explore/explore_metrics.csv`
- `results/explore/top5_plus_residual_learning_metrics.csv`
- `results/explore/model_performance_comparison.png`
- `results/explore/top5_plus_residual_learning_performance.png`
- `results/explore/residual_plot_all_models.png`

Current top integrated results:
- `Residual-XGB`: RMSE 1.0325, MAE 0.1391, R2 0.9459
- `Residual-RF`: RMSE 1.0412, MAE 0.1591, R2 0.9450
- `M3-L`: RMSE 1.0589, MAE 0.2322, R2 0.9431
- `M2-L`: RMSE 1.3830, MAE 0.2631, R2 0.9029

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

This creates two Roberts-style blocked validation datasets without changing the existing stratified split.

Dataset 1: `fair_all`
- Purpose: all classes appear in training and testing, while species are kept as complete blocks.
- Bias control: residual-learning models use class-balanced sample weights, and reports include micro, macro-class, capped weighted macro-class, per-class, and per-block metrics.
- Outputs:
  - `data/splits/block_cv/fair_all/fold_*/train.csv`
  - `data/splits/block_cv/fair_all/fold_*/test.csv`
  - `results/block_cv/fair_all/cv_predictions.csv`
  - `results/block_cv/fair_all/fold_metrics.csv`
  - `results/block_cv/fair_all/metric_summary.csv`
  - `results/block_cv/fair_all/per_class_metrics.csv`
  - `results/block_cv/fair_all/per_block_metrics.csv`

Dataset 2: `leave_class_out`
- Purpose: train without one target class, then predict that held-out class.
- Groups:
  - `A`: train without `Teleostei`, predict `Teleostei`.
  - `B`: train without `Mammalia`, predict `Mammalia`.
  - `C`: train without `Insecta`, predict `Insecta`.
- Outputs:
  - `data/splits/block_cv/leave_class_out/A|B|C/train.csv`
  - `data/splits/block_cv/leave_class_out/A|B|C/test.csv`
  - `results/block_cv/leave_class_out/A|B|C/benchmark_predictions_test.csv`
  - `results/block_cv/leave_class_out/A|B|C/benchmark_metrics.csv`
  - `results/block_cv/leave_class_out/benchmark_summary_groups.csv`

Use `--skip-models` to only write train/test CSV files.

## 5. Utility Script

### `code/class_distribution.py`

Purpose:
- Summarize class distributions for one or more CSV files.

Default output:
- `data/cleaning/class_distribution.csv`

Example:

```bash
python code/class_distribution.py --input data/cleaning/standard_data.csv
```

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
Rscript code/pglmm_phyr.R
python code/explore.py
```

