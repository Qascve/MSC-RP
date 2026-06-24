#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ape)
  library(caper)
})

required_cols <- c("taxon_name", "log_mass", "inv_kT")
required_train_cols <- c(required_cols, "log_BMR")

args <- commandArgs(trailingOnly = TRUE)

get_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(normalizePath(getwd(), winslash = "/", mustWork = FALSE))
  }
  normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE)
}

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx >= length(args)) {
    if (!is.null(default)) {
      return(default)
    }
    stop(sprintf("Missing required argument: %s", flag), call. = FALSE)
  }
  args[idx + 1]
}

require_existing_file <- function(path_value, label) {
  if (!file.exists(path_value)) {
    stop(sprintf("%s does not exist: %s", label, path_value), call. = FALSE)
  }
  normalizePath(path_value, winslash = "/", mustWork = TRUE)
}

to_phylo_label <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub("\\s+", "_", x)
  x
}

get_valid_tip_labels <- function(phy) {
  if (is.null(phy$edge.length)) {
    return(unique(phy$tip.label))
  }
  tip_count <- length(phy$tip.label)
  edge_child <- phy$edge[, 2]
  tip_edge_idx <- which(edge_child <= tip_count)
  if (length(tip_edge_idx) == 0) {
    return(character(0))
  }
  tip_lengths <- phy$edge.length[tip_edge_idx]
  valid_idx <- is.finite(tip_lengths) & (tip_lengths > 0)
  if (!any(valid_idx)) {
    return(character(0))
  }
  tip_ids <- edge_child[tip_edge_idx][valid_idx]
  unique(phy$tip.label[tip_ids])
}

script_path <- get_script_path()
repo_root <- normalizePath(
  file.path(dirname(script_path), ".."),
  winslash = "/",
  mustWork = FALSE
)

train_path <- get_arg(
  "--train",
  file.path(repo_root, "data", "splits", "stratified", "train.csv")
)
test_path <- get_arg(
  "--test",
  file.path(repo_root, "data", "splits", "stratified", "test.csv")
)
tree_path <- get_arg(
  "--tree",
  file.path(repo_root, "data", "phylogeny", "unique_taxon_names.nwk")
)
output_path <- get_arg(
  "--output",
  file.path(repo_root, "results", "explore", "pgls_predictions_test.csv")
)

train_path <- require_existing_file(train_path, "Train CSV")
test_path <- require_existing_file(test_path, "Test CSV")
tree_path <- require_existing_file(tree_path, "Tree file")

output_dir <- dirname(output_path)
if (!dir.exists(output_dir)) {
  stop(sprintf("Output directory does not exist: %s", output_dir), call. = FALSE)
}

train_df <- read.csv(train_path, stringsAsFactors = FALSE, check.names = FALSE)
test_df <- read.csv(test_path, stringsAsFactors = FALSE, check.names = FALSE)
phy <- read.tree(tree_path)

missing_train <- setdiff(required_train_cols, names(train_df))
missing_test <- setdiff(required_cols, names(test_df))
if (length(missing_train) > 0) {
  stop(sprintf("Train CSV missing columns: %s", paste(missing_train, collapse = ", ")), call. = FALSE)
}
if (length(missing_test) > 0) {
  stop(sprintf("Test CSV missing columns: %s", paste(missing_test, collapse = ", ")), call. = FALSE)
}
if (!("row_id" %in% names(test_df))) {
  test_df$row_id <- seq_len(nrow(test_df)) - 1L
}

train_df$taxon_phylo <- to_phylo_label(train_df$taxon_name)
test_df$taxon_phylo <- to_phylo_label(test_df$taxon_name)

for (col in c("log_mass", "inv_kT", "log_BMR")) {
  train_df[[col]] <- suppressWarnings(as.numeric(train_df[[col]]))
}
for (col in c("log_mass", "inv_kT")) {
  test_df[[col]] <- suppressWarnings(as.numeric(test_df[[col]]))
}
test_df$row_id <- suppressWarnings(as.integer(test_df$row_id))

train_df <- train_df[is.finite(train_df$log_BMR) & is.finite(train_df$log_mass) & is.finite(train_df$inv_kT), ]
test_df <- test_df[!is.na(test_df$row_id), ]

valid_tip_set <- get_valid_tip_labels(phy)
if (length(valid_tip_set) < 10) {
  stop("Too few tree tips have positive, non-missing terminal branch lengths.", call. = FALSE)
}

phy <- drop.tip(phy, setdiff(phy$tip.label, valid_tip_set))
tree_tip_set <- unique(phy$tip.label)
train_df <- train_df[train_df$taxon_phylo %in% tree_tip_set, ]

if (nrow(train_df) < 10) {
  stop("Too few train rows overlap with tree tips for stable PGLS fitting.", call. = FALSE)
}
if (length(unique(train_df$taxon_phylo)) < 10) {
  stop("Too few unique taxa overlap with tree tips for stable PGLS fitting.", call. = FALSE)
}

# Use one row per taxon for caper::pgls to follow comparative-data assumptions.
train_taxon <- aggregate(
  cbind(log_mass, inv_kT, log_BMR) ~ taxon_phylo,
  data = train_df,
  FUN = mean
)

comp <- comparative.data(
  phy = phy,
  data = train_taxon,
  names.col = "taxon_phylo",
  vcv = TRUE,
  na.omit = FALSE,
  warn.dropped = FALSE
)

fit_pgls_robust <- function(formula, comp_data) {
  ml_fit <- try(
    pgls(
      formula = formula,
      data = comp_data,
      lambda = "ML"
    ),
    silent = TRUE
  )
  if (!inherits(ml_fit, "try-error")) {
    return(ml_fit)
  }

  fallback_lambdas <- c(0.99, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.01, 0.00)
  for (lambda_val in fallback_lambdas) {
    lambda_fit <- try(
      pgls(
        formula = formula,
        data = comp_data,
        lambda = lambda_val
      ),
      silent = TRUE
    )
    if (!inherits(lambda_fit, "try-error")) {
      message(sprintf("PGLS ML optimization failed; fallback to fixed lambda = %.2f", lambda_val))
      return(lambda_fit)
    }
  }

  stop(
    sprintf(
      paste(
        "PGLS fitting failed for ML lambda optimization",
        "and all fixed-lambda fallbacks."
      ),
      call. = FALSE
    )
  )
}

pgls_fit <- fit_pgls_robust(
  formula = log_BMR ~ log_mass + inv_kT,
  comp_data = comp
)

coef_vec <- coef(pgls_fit)
test_design <- model.matrix(~ log_mass + inv_kT, data = test_df)
test_ok <- is.finite(test_df$log_mass) & is.finite(test_df$inv_kT) & (test_df$taxon_phylo %in% tree_tip_set)

yhat_log <- rep(NA_real_, nrow(test_df))
if (any(test_ok)) {
  yhat_log[test_ok] <- as.numeric(test_design[test_ok, , drop = FALSE] %*% coef_vec[colnames(test_design)])
}

out_df <- data.frame(
  row_id = test_df$row_id,
  y_pred = exp(yhat_log)
)
out_df <- out_df[order(out_df$row_id), ]

write.csv(out_df, output_path, row.names = FALSE)
