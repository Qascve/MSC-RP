#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_packages <- c("ape", "phyr")
  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      "Missing R packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install with: install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))",
      call. = FALSE
    )
  }
})

find_root <- function(marker = ".gitignore") {
  starts <- c(getwd(), dirname(normalizePath(sys.frame(1)$ofile %||% ".", mustWork = FALSE)))
  for (start in starts) {
    current <- normalizePath(start, winslash = "/", mustWork = FALSE)
    repeat {
      if (file.exists(file.path(current, marker))) {
        return(current)
      }
      parent <- dirname(current)
      if (identical(parent, current)) {
        break
      }
      current <- parent
    }
  }
  stop("Cannot find project root by marker: ", marker, call. = FALSE)
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

parse_args <- function(args) {
  get_value <- function(name, default) {
    idx <- match(name, args)
    if (is.na(idx)) {
      return(default)
    }
    if (idx == length(args)) {
      stop("Argument ", name, " requires a value.", call. = FALSE)
    }
    args[[idx + 1]]
  }

  list(
    train = get_value("--train", "data/splits/stratified/train.csv"),
    test = get_value("--test", "data/splits/stratified/test.csv"),
    tree = get_value("--tree", "data/phylogeny/unique_taxon_names.nwk"),
    embeddings = get_value("--embeddings", "data/phylogeny/phylogenetic_embeddings.csv"),
    out_dir = get_value("--out-dir", "results/pglmm_phyr"),
    reml = tolower(get_value("--reml", "TRUE")) %in% c("true", "t", "1", "yes", "y"),
    write_vcv = tolower(get_value("--write-vcv", "FALSE")) %in% c("true", "t", "1", "yes", "y")
  )
}

resolve_path <- function(root, path) {
  if (grepl("^[A-Za-z]:[/\\\\]|^/", path)) {
    return(normalizePath(path, winslash = "/", mustWork = FALSE))
  }
  normalizePath(file.path(root, path), winslash = "/", mustWork = FALSE)
}

normalize_tip_like_r_sub <- function(x) {
  trimws(sub("_", " ", x, fixed = TRUE))
}

clean_observations <- function(path, need_response = TRUE) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("taxon_name", "wet_Mass_kg", "temperature", "BMR")
  if (!need_response) {
    required <- c("taxon_name", "wet_Mass_kg", "temperature")
  }
  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0) {
    stop(basename(path), " missing columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  df$taxon_name <- trimws(as.character(df$taxon_name))
  df$wet_Mass_kg <- suppressWarnings(as.numeric(df$wet_Mass_kg))
  df$temperature <- suppressWarnings(as.numeric(df$temperature))
  if ("BMR" %in% names(df)) {
    df$BMR <- suppressWarnings(as.numeric(df$BMR))
  }

  keep <- !is.na(df$taxon_name) & nzchar(df$taxon_name) &
    !is.na(df$wet_Mass_kg) & df$wet_Mass_kg > 0 &
    !is.na(df$temperature)
  if (need_response) {
    keep <- keep & !is.na(df$BMR) & df$BMR > 0
  }
  df <- df[keep, , drop = FALSE]
  if (nrow(df) == 0) {
    stop("No valid rows left after cleaning: ", path, call. = FALSE)
  }
  df$row_id <- seq_len(nrow(df)) - 1

  k_boltzmann_ev_per_k <- 8.617e-5
  df$temp_K <- df$temperature + 273.15
  df$inv_kT <- 1 / (k_boltzmann_ev_per_k * df$temp_K)
  df$log_mass <- log(df$wet_Mass_kg)
  if ("BMR" %in% names(df)) {
    df$log_BMR <- log(df$BMR)
  }
  row.names(df) <- NULL
  df
}

load_embeddings <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  emb <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  names(emb) <- tolower(names(emb))
  if (!"taxon_name" %in% names(emb)) {
    first_col <- names(emb)[[1]]
    names(emb)[names(emb) == first_col] <- "taxon_name"
  }
  keep_cols <- intersect(c("taxon_name", "pc1", "pc2", "pc3", "pc4", "pc5"), names(emb))
  emb <- emb[, keep_cols, drop = FALSE]
  emb$taxon_name <- trimws(as.character(emb$taxon_name))
  emb <- emb[!duplicated(emb$taxon_name), , drop = FALSE]
  emb
}

add_embeddings <- function(df, embeddings) {
  if (is.null(embeddings)) {
    return(df)
  }
  existing_pcs <- intersect(c("pc1", "pc2", "pc3", "pc4", "pc5"), names(df))
  if (length(existing_pcs) > 0) {
    df <- df[, setdiff(names(df), existing_pcs), drop = FALSE]
  }
  merge(df, embeddings, by = "taxon_name", all.x = TRUE, sort = FALSE)
}

descendant_tip_labels <- function(phylo, node) {
  n_tip <- length(phylo$tip.label)
  children_by_parent <- split(phylo$edge[, 2], phylo$edge[, 1])
  stack <- as.integer(node)
  tip_ids <- integer(0)

  while (length(stack) > 0) {
    current <- stack[[length(stack)]]
    stack <- stack[-length(stack)]
    if (current <= n_tip) {
      tip_ids <- c(tip_ids, current)
    } else {
      children <- children_by_parent[[as.character(current)]]
      if (!is.null(children)) {
        stack <- c(stack, as.integer(children))
      }
    }
  }

  unique(phylo$tip.label[tip_ids])
}

drop_tips_below_bad_branches <- function(phylo) {
  if (is.null(phylo$edge.length)) {
    stop("The phylogeny has no branch lengths; Brownian-motion PGLMM needs branch lengths.", call. = FALSE)
  }

  bad_edges <- which(is.na(phylo$edge.length) | phylo$edge.length <= 0)
  if (length(bad_edges) == 0) {
    return(list(phylo = phylo, removed_tips = character(0), bad_edge_count = 0))
  }

  bad_child_nodes <- phylo$edge[bad_edges, 2]
  removed_tips <- unique(unlist(lapply(bad_child_nodes, descendant_tip_labels, phylo = phylo)))
  kept_tips <- setdiff(phylo$tip.label, removed_tips)
  if (length(kept_tips) < 2) {
    stop(
      "Dropping tips below non-positive/NA branches leaves fewer than 2 tips.",
      call. = FALSE
    )
  }

  list(
    phylo = ape::keep.tip(phylo, kept_tips),
    removed_tips = removed_tips,
    bad_edge_count = length(bad_edges)
  )
}

build_phylo_vcv <- function(phylo, species) {
  phylo_subset <- ape::keep.tip(phylo, species)
  vcv_raw <- ape::vcv.phylo(phylo_subset, corr = FALSE)
  vcv_raw[species, species, drop = FALSE]
}

fixed_effect_predict_log <- function(fit, newdata) {
  x <- model.matrix(~ log_mass + inv_kT, data = newdata)
  beta <- as.numeric(fit$B[, 1])
  names(beta) <- row.names(fit$B)
  missing_beta <- setdiff(colnames(x), names(beta))
  if (length(missing_beta) > 0) {
    stop("Model coefficient(s) not found for: ", paste(missing_beta, collapse = ", "), call. = FALSE)
  }
  as.numeric(x %*% beta[colnames(x)])
}

fixed_effect_table <- function(fit) {
  data.frame(
    term = row.names(fit$B),
    Value = as.numeric(fit$B[, 1]),
    Std.Error = as.numeric(fit$B.se),
    Zscore = as.numeric(fit$B.zscore),
    Pvalue = as.numeric(fit$B.pvalue),
    row.names = NULL
  )
}

random_variance_table <- function(fit) {
  random_variance <- c(as.numeric(fit$s2r), as.numeric(fit$s2n))
  component <- paste0("random_effect_", seq_along(random_variance))
  data.frame(
    component = c(component, "residual"),
    Variance = c(random_variance, as.numeric(fit$s2resid)),
    Std.Dev = sqrt(c(random_variance, as.numeric(fit$s2resid))),
    row.names = NULL
  )
}

metrics <- function(observed, predicted) {
  ok <- is.finite(observed) & is.finite(predicted)
  observed <- observed[ok]
  predicted <- predicted[ok]
  if (length(observed) == 0) {
    return(data.frame(n = 0, rmse = NA_real_, mae = NA_real_, r2 = NA_real_))
  }
  residual <- observed - predicted
  data.frame(
    n = length(observed),
    rmse = sqrt(mean(residual^2)),
    mae = mean(abs(residual)),
    r2 = 1 - sum(residual^2) / sum((observed - mean(observed))^2)
  )
}

main <- function() {
  root <- find_root()
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  train_path <- resolve_path(root, args$train)
  test_path <- resolve_path(root, args$test)
  tree_path <- resolve_path(root, args$tree)
  embeddings_path <- resolve_path(root, args$embeddings)
  out_dir <- resolve_path(root, args$out_dir)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  train <- clean_observations(train_path, need_response = TRUE)
  test <- clean_observations(test_path, need_response = TRUE)
  embeddings <- load_embeddings(embeddings_path)
  train <- add_embeddings(train, embeddings)
  test <- add_embeddings(test, embeddings)

  phylo <- ape::read.tree(tree_path)
  phylo$tip.label <- normalize_tip_like_r_sub(phylo$tip.label)
  phylo$tip.label <- make.unique(phylo$tip.label)
  bad_branch_filter <- drop_tips_below_bad_branches(phylo)
  phylo <- bad_branch_filter$phylo
  removed_bad_branch_tips <- bad_branch_filter$removed_tips

  tree_species <- phylo$tip.label
  train$removed_due_bad_branch <- train$taxon_name %in% removed_bad_branch_tips
  test$removed_due_bad_branch <- test$taxon_name %in% removed_bad_branch_tips
  train$in_phylogeny <- train$taxon_name %in% tree_species
  test$in_phylogeny <- test$taxon_name %in% tree_species

  train_phylo <- train[train$in_phylogeny, , drop = FALSE]
  if (nrow(train_phylo) < 3) {
    stop("Need at least 3 training rows with species present in the phylogeny.", call. = FALSE)
  }
  if (length(unique(train_phylo$taxon_name)) < 2) {
    stop("Need at least 2 training species present in the phylogeny.", call. = FALSE)
  }

  fit_species <- sort(unique(train_phylo$taxon_name))
  vcv_fit <- build_phylo_vcv(phylo, fit_species)

  fit <- phyr::pglmm(
    log_BMR ~ log_mass + inv_kT + (1 | taxon_name__),
    data = train_phylo,
    family = "gaussian",
    cov_ranef = list(taxon_name = vcv_fit),
    REML = args$reml
  )

  test$seen_in_training <- test$taxon_name %in% unique(train_phylo$taxon_name)
  test$phylo_random_effect_used <- FALSE
  test$prediction_type <- ifelse(
    test$in_phylogeny,
    "fixed_effect_from_pglmm",
    ifelse(
      test$removed_due_bad_branch,
      "not_predicted_removed_due_bad_phylo_branch",
      "not_predicted_species_absent_from_phylogeny"
    )
  )

  test$y_pred_log_BMR <- NA_real_
  test$y_pred_BMR <- NA_real_
  pred_rows <- which(test$in_phylogeny)
  if (length(pred_rows) > 0) {
    pred_log <- fixed_effect_predict_log(fit, test[pred_rows, , drop = FALSE])
    test$y_pred_log_BMR[pred_rows] <- pred_log
    test$y_pred_BMR[pred_rows] <- exp(pred_log)
  }

  all_species <- sort(unique(c(train_phylo$taxon_name, test$taxon_name[test$in_phylogeny])))
  phylo_all <- ape::keep.tip(phylo, all_species)
  vcv_raw <- ape::vcv.phylo(phylo_all, corr = FALSE)
  vcv_corr <- stats::cov2cor(vcv_raw)
  train_species <- unique(train_phylo$taxon_name)
  test$phylo_cov_self <- NA_real_
  test$mean_corr_to_training_species <- NA_real_
  matched_test_species <- intersect(test$taxon_name[test$in_phylogeny], row.names(vcv_raw))
  test$phylo_cov_self[test$taxon_name %in% matched_test_species] <-
    diag(vcv_raw)[test$taxon_name[test$taxon_name %in% matched_test_species]]
  corr_train_species <- intersect(train_species, colnames(vcv_corr))
  if (length(corr_train_species) > 0) {
    for (sp in matched_test_species) {
      test$mean_corr_to_training_species[test$taxon_name == sp] <-
        mean(vcv_corr[sp, corr_train_species], na.rm = TRUE)
    }
  }

  train_phylo$fixed_only_log_BMR <- fixed_effect_predict_log(fit, train_phylo)
  train_phylo$fixed_only_BMR <- exp(train_phylo$fixed_only_log_BMR)
  train_phylo$conditional_log_BMR <- phyr::pglmm_predicted_values(
    fit,
    re.form = NULL,
    type = "link"
  )$Y_hat
  train_phylo$conditional_BMR <- exp(train_phylo$conditional_log_BMR)

  fixed_effects <- fixed_effect_table(fit)
  random_variance <- random_variance_table(fit)

  test_metrics_bmr <- metrics(test$BMR, test$y_pred_BMR)
  test_metrics_log <- metrics(test$log_BMR, test$y_pred_log_BMR)
  metric_out <- rbind(
    data.frame(scale = "BMR", test_metrics_bmr),
    data.frame(scale = "log_BMR", test_metrics_log)
  )
  metric_lines <- c(
    "PGLMM test-set metrics",
    sprintf(
      "BMR: n = %d, RMSE = %.6f, MAE = %.6f, R2 = %.6f",
      test_metrics_bmr$n,
      test_metrics_bmr$rmse,
      test_metrics_bmr$mae,
      test_metrics_bmr$r2
    ),
    sprintf(
      "log_BMR: n = %d, RMSE = %.6f, MAE = %.6f, R2 = %.6f",
      test_metrics_log$n,
      test_metrics_log$rmse,
      test_metrics_log$mae,
      test_metrics_log$r2
    )
  )

  write.csv(test, file.path(out_dir, "pglmm_test_predictions.csv"), row.names = FALSE)
  write.csv(train_phylo, file.path(out_dir, "pglmm_train_fitted.csv"), row.names = FALSE)
  write.csv(fixed_effects, file.path(out_dir, "pglmm_fixed_effects.csv"), row.names = TRUE)
  write.csv(random_variance, file.path(out_dir, "pglmm_random_variance.csv"), row.names = TRUE)
  write.csv(metric_out, file.path(out_dir, "pglmm_test_metrics.csv"), row.names = FALSE)
  writeLines(metric_lines, file.path(out_dir, "pglmm_metrics_summary.txt"))
  write.csv(
    data.frame(taxon_name = removed_bad_branch_tips),
    file.path(out_dir, "removed_bad_branch_species.csv"),
    row.names = FALSE
  )
  writeLines(capture.output(summary(fit)), file.path(out_dir, "pglmm_model_summary.txt"))

  if (isTRUE(args$write_vcv)) {
    write.csv(vcv_raw, file.path(out_dir, "phylogenetic_vcv_raw.csv"), row.names = TRUE)
    write.csv(vcv_corr, file.path(out_dir, "phylogenetic_vcv_correlation.csv"), row.names = TRUE)
  }

  cat("PGLMM finished\n")
  cat("  Bad branch count:", bad_branch_filter$bad_edge_count, "\n")
  cat("  Species removed due bad branches:", length(removed_bad_branch_tips), "\n")
  cat("  Train rows used:", nrow(train_phylo), "\n")
  cat("  Train species used:", length(unique(train_phylo$taxon_name)), "\n")
  cat("  Test rows predicted:", sum(is.finite(test$y_pred_BMR)), "\n")
  cat("  Test RMSE (BMR):", sprintf("%.6f", test_metrics_bmr$rmse), "\n")
  cat("  Test R2 (BMR):", sprintf("%.6f", test_metrics_bmr$r2), "\n")
  cat("  Test RMSE (log_BMR):", sprintf("%.6f", test_metrics_log$rmse), "\n")
  cat("  Test R2 (log_BMR):", sprintf("%.6f", test_metrics_log$r2), "\n")
  cat("  Output directory:", out_dir, "\n")
}

main()
