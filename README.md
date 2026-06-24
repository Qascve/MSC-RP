# MSC-RP Workflow Guide


## 1. Project Goal (Code Perspective)

For the BMR (basal metabolic rate) task, this project merges and cleans multiple raw datasets, builds phylogeny-based embedding features, creates stratified train/test splits, and evaluates both benchmark and exploratory models.

## 2. Environment Setup

Run from the project root:

```bash
pip install -r requirements.txt
```

Notes:
- `requirements.txt` contains Python dependencies.
- `filter_target_classes.py` and `merge_bmr_mass_temp.py` may require internet access (GBIF/pytaxon).

## 3. Recommended Execution Order (Main Pipeline)

Run all commands from the project root.

### Step 1: Merge three raw source datasets

Script: `code/merge_bmr_mass_temp.py`  
Inputs:
- `data/raw/pnas.2303764120.sd01.xlsx`
- `data/raw/observations.xlsx`
- `data/raw/41586_2010_BFnature08920_MOESM90_ESM.xls`

Output:
- `data/cleaning/merged_bmr_mass_temperature.csv`

Run:

```bash
python code/merge_bmr_mass_temp.py
```

Purpose:
- Unify schema (mass, temperature, BMR, taxonomy fields, etc.).
- Remove invalid records and duplicates.
- Fill missing taxonomy fields (`class`/`order`/`family`) using GBIF when possible.

### Step 2: Standardize taxon names + filter classes

Script: `code/filter_target_classes.py`  
Inputs:
- `data/cleaning/merged_bmr_mass_temperature.csv`
- Config: `code/config.json`

Outputs:
- `data/cleaning/standard_data.csv`
- `data/cleaning/filtered_data.csv`

Run:

```bash
python code/filter_target_classes.py
```

Purpose:
- Remove blacklist classes (defined in `EXCLUDED_CLASSES`).
- Standardize species names via pytaxon and create `taxon_name`.
- Apply a final safety filter using GBIF class lookups.

### Step 3: Export taxon list for phylogeny matching

Script: `code/export_taxon_names.py`  
Input:
- `data/cleaning/filtered_data.csv`

Output:
- `data/phylogeny/unique_taxon_names.txt`

Run:

```bash
python code/export_taxon_names.py
```

Purpose:
- Export unique `taxon_name` values (one per line) for tree matching.

### Step 4: Build phylogenetic embedding features from tree

Script: `code/phylogeny.py`  
Inputs:
- Tree file: `data/phylogeny/unique_taxon_names.nwk`
- Species list: `data/phylogeny/unique_taxon_names.txt`

Outputs:
- `data/phylogeny/phylogenetic_embeddings.csv`
- `data/phylogeny/phylogeny_matched_species.csv`

Run:

```bash
python code/phylogeny.py
```

Purpose:
- Match and prune tree tips against the species list.
- Compute patristic distance matrix between species.
- Run PCA on the distance matrix to produce `PC1`-`PC5` embeddings.

### Step 5: Merge observations with phylogenetic embeddings

Script: `code/merge_phylo_embedding.py`  
Inputs:
- `data/phylogeny/phylogenetic_embeddings.csv`
- `data/cleaning/filtered_data.csv`

Output:
- `data/merge_phylo.csv`

Run:

```bash
python code/merge_phylo_embedding.py
```

Purpose:
- Inner-join on `taxon_name` and append `pc1`-`pc5` to filtered observations.

### Step 6: Create stratified train/test split by class

Script: `code/split_train_test_bmr.py`  
Input:
- `data/merge_phylo.csv`

Outputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`
- `data/splits/stratified/class_split_summary.csv`

Run:

```bash
python code/split_train_test_bmr.py
```

Purpose:
- Apply strict row filtering and derive features: `log_mass`, `log_BMR`, `inv_kT`.
- Perform class-stratified split so each class is represented in both train and test when possible.

### Step 7: Train and evaluate benchmark models

Script: `code/benchmark_bmr_models.py`  
Inputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`

Output directory:
- `results/benchmark/`

Key outputs:
- `results/benchmark/all/benchmark_predictions_test.csv`
- `results/benchmark/*/benchmark_metrics.csv`
- `results/benchmark/benchmark_summary_groups.csv`

Run:

```bash
python code/benchmark_bmr_models.py
```

Purpose:
- Evaluate models on full test set and selected class subsets (A/B/C groups).
- Save predictions, error metrics, residual diagnostics, SHAP outputs, and summary tables.

### Step 8: Run exploratory model comparison (MTE + benchmark predictions)

Script: `code/explore.py`  
Inputs:
- `data/splits/stratified/train.csv`
- `data/splits/stratified/test.csv`
- `results/benchmark/all/benchmark_predictions_test.csv`

Output directory:
- `results/explore/`

Run:

```bash
python code/explore.py
```

Purpose:
- Fit and evaluate MTE-style models (`m0`-`m3`).
- Compare against benchmark model predictions and produce integrated metrics/plots.

Note:
- This script includes a `--pgls-r-script` hook to call an external R script. Whether that R file is version-tracked depends on current Git tracking status.

## 4. Utility Script

### `code/class_distribution.py`

Purpose:
- Summarize `class` distribution for one or more CSV files.

Default output:
- `data/cleaning/class_distribution.csv`

Example:

```bash
python code/class_distribution.py --input data/cleaning/standard_data.csv
```

## 5. Minimal Reproducible Command List

```bash
python code/merge_bmr_mass_temp.py
python code/filter_target_classes.py
python code/export_taxon_names.py
python code/phylogeny.py
python code/merge_phylo_embedding.py
python code/split_train_test_bmr.py
python code/benchmark_bmr_models.py
python code/explore.py
```

