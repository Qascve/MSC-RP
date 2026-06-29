#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_packages <- c("ape", "nlme", "phytools")
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

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

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
    train = get_value("--train", "data/splits/train.csv"),
    test = get_value("--test", "data/splits/test.csv"),
    tree = get_value("--tree", "data/phylogeny/unique_taxon_names.nwk"),
    out_dir = get_value("--out-dir", "results/pgls_ape"),
    formula = get_value("--formula", "log_BMR ~ log_mass + inv_kT")
  )
}

resolve_path <- function(root, path) {
  if (grepl("^[A-Za-z]:[/\\\\]|^/", path)) {
    return(normalizePath(path, winslash = "/", mustWork = FALSE))
  }
  normalizePath(file.path(root, path), winslash = "/", mustWork = FALSE)
}

normalize_tip_label <- function(x) {
  trimws(gsub("_", " ", x, fixed = TRUE))
}

clean_observations <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("taxon_name", "wet_Mass_kg", "temperature", "BMR")
  missing_cols <- setdiff(required, names(df))
  if (length(missing_cols) > 0) {
    stop(basename(path), " missing columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  df$row_id <- seq_len(nrow(df)) - 1
  df$taxon_name <- trimws(as.character(df$taxon_name))
  df$Species <- df$taxon_name
  df$wet_Mass_kg <- suppressWarnings(as.numeric(df$wet_Mass_kg))
  df$temperature <- suppressWarnings(as.numeric(df$temperature))
  df$BMR <- suppressWarnings(as.numeric(df$BMR))

  keep <- !is.na(df$taxon_name) & nzchar(df$taxon_name) &
    !is.na(df$wet_Mass_kg) & df$wet_Mass_kg > 0 &
    !is.na(df$temperature) &
    !is.na(df$BMR) & df$BMR > 0 &
    (df$temperature + 273.15) > 0
  df <- df[keep, , drop = FALSE]
  if (nrow(df) == 0) {
    stop("No valid rows left after cleaning: ", path, call. = FALSE)
  }

  k_boltzmann_ev_per_k <- 8.617e-5
  df$temp_K <- df$temperature + 273.15
  df$inv_kT <- 1 / (k_boltzmann_ev_per_k * df$temp_K)
  df$log_mass <- log(df$wet_Mass_kg)
  df$log_BMR <- log(df$BMR)
  row.names(df) <- NULL
  df
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
    stop("The phylogeny has no branch lengths; PGLS needs branch lengths.", call. = FALSE)
  }

  bad_edges <- which(is.na(phylo$edge.length) | phylo$edge.length <= 0)
  if (length(bad_edges) == 0) {
    return(list(phylo = phylo, removed_tips = character(0), bad_edge_count = 0))
  }

  bad_child_nodes <- phylo$edge[bad_edges, 2]
  removed_tips <- unique(unlist(lapply(bad_child_nodes, descendant_tip_labels, phylo = phylo)))
  kept_tips <- setdiff(phylo$tip.label, removed_tips)
  if (length(kept_tips) < 2) {
    stop("Dropping bad branches leaves fewer than 2 tips.", call. = FALSE)
  }
  list(
    phylo = ape::keep.tip(phylo, kept_tips),
    removed_tips = removed_tips,
    bad_edge_count = length(bad_edges)
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
  denom <- sum((observed - mean(observed))^2)
  data.frame(
    n = length(observed),
    rmse = sqrt(mean(residual^2)),
    mae = mean(abs(residual)),
    r2 = ifelse(denom > 0, 1 - sum(residual^2) / denom, NA_real_)
  )
}

fit_pgls_models <- function(formula, train, phylo, lambda_start = 0.5) {
  model_fitting_functions <- list(
    pglsModel_Lambda = function() nlme::gls(
      formula,
      data = train,
      correlation = ape::corPagel(lambda_start, phylo, form = ~Species, fixed = FALSE),
      method = "ML"
    ),
    pglsModel_Brownian = function() nlme::gls(
      formula,
      data = train,
      correlation = ape::corBrownian(1, phylo, form = ~Species),
      method = "ML"
    ),
    pglsModel_Martins = function() nlme::gls(
      formula,
      data = train,
      correlation = ape::corMartins(1, phylo, form = ~Species, fixed = FALSE),
      method = "ML"
    ),
    pglsModel_Blomberg = function() nlme::gls(
      formula,
      data = train,
      correlation = ape::corBlomberg(1, phylo, form = ~Species, fixed = FALSE),
      method = "ML"
    ),
    pglsModel_Grafen = function() nlme::gls(
      formula,
      data = train,
      correlation = ape::corGrafen(1, phylo, form = ~Species, fixed = FALSE),
      method = "ML"
    )
  )

  models <- list()
  rows <- list()
  for (model_name in names(model_fitting_functions)) {
    fit_result <- try(model_fitting_functions[[model_name]](), silent = TRUE)
    if (!inherits(fit_result, "try-error")) {
      models[[model_name]] <- fit_result
      rows[[length(rows) + 1]] <- data.frame(
        model = model_name,
        status = "ok",
        AIC = AIC(fit_result),
        logLik = as.numeric(logLik(fit_result)),
        message = "",
        row.names = NULL
      )
    } else {
      rows[[length(rows) + 1]] <- data.frame(
        model = model_name,
        status = "failed",
        AIC = NA_real_,
        logLik = NA_real_,
        message = as.character(fit_result),
        row.names = NULL
      )
    }
  }
  list(models = models, aic = do.call(rbind, rows))
}

run_lambda_signal_test <- function(formula, train, phylo) {
  lm_fit <- stats::lm(formula, data = train)
  train$residual_for_lambda <- stats::residuals(lm_fit)
  species_resid <- stats::aggregate(residual_for_lambda ~ Species, data = train, FUN = mean)
  resid_vec <- species_resid$residual_for_lambda
  names(resid_vec) <- species_resid$Species

  lambda_species <- intersect(phylo$tip.label, names(resid_vec))
  if (length(lambda_species) < 3) {
    return(list(lambda = 0.5, table = data.frame(
      lambda = NA_real_,
      logL = NA_real_,
      P = NA_real_,
      n_species = length(lambda_species),
      source = "lm_residuals",
      message = "Fewer than 3 species available for phytools::phylosig.",
      row.names = NULL
    )))
  }

  phylo_subset <- ape::keep.tip(phylo, lambda_species)
  resid_vec <- resid_vec[phylo_subset$tip.label]
  lambda_test <- try(
    phytools::phylosig(phylo_subset, resid_vec, method = "lambda", test = TRUE),
    silent = TRUE
  )

  if (inherits(lambda_test, "try-error")) {
    return(list(lambda = 0.5, table = data.frame(
      lambda = NA_real_,
      logL = NA_real_,
      P = NA_real_,
      n_species = length(lambda_species),
      source = "lm_residuals",
      message = as.character(lambda_test),
      row.names = NULL
    )))
  }

  lambda_value <- as.numeric(lambda_test$lambda)
  if (!is.finite(lambda_value)) {
    lambda_value <- 0.5
  }
  lambda_start <- min(max(lambda_value, 1e-6), 0.999999)
  list(lambda = lambda_start, table = data.frame(
    lambda = as.numeric(lambda_test$lambda),
    logL = as.numeric(lambda_test$logL),
    P = as.numeric(lambda_test$P),
    n_species = length(lambda_species),
    source = "lm_residuals",
    message = "",
    row.names = NULL
  ))
}

main <- function() {
  root <- find_root()
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  train_path <- resolve_path(root, args$train)
  test_path <- resolve_path(root, args$test)
  tree_path <- resolve_path(root, args$tree)
  out_dir <- resolve_path(root, args$out_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  formula <- stats::as.formula(args$formula)
  train <- clean_observations(train_path)
  test <- clean_observations(test_path)

  phylo <- ape::read.tree(tree_path)
  phylo$tip.label <- make.unique(normalize_tip_label(phylo$tip.label))
  bad_branch_filter <- drop_tips_below_bad_branches(phylo)
  phylo <- bad_branch_filter$phylo

  train$in_phylogeny <- train$Species %in% phylo$tip.label
  test$in_phylogeny <- test$Species %in% phylo$tip.label
  train_phylo <- train[train$in_phylogeny, , drop = FALSE]
  if (nrow(train_phylo) < 3 || length(unique(train_phylo$Species)) < 2) {
    stop("Need at least 3 training rows and 2 species present in the phylogeny.", call. = FALSE)
  }

  fit_species <- sort(unique(train_phylo$Species))
  phylo_fit <- ape::keep.tip(phylo, fit_species)
  lambda_signal <- run_lambda_signal_test(formula, train_phylo, phylo_fit)
  write.csv(
    lambda_signal$table,
    file.path(out_dir, "pgls_lambda_residual_test.csv"),
    row.names = FALSE
  )

  pgls_result <- fit_pgls_models(
    formula,
    train_phylo,
    phylo_fit,
    lambda_start = lambda_signal$lambda
  )
  aic_table <- pgls_result$aic
  ok_aic <- aic_table[is.finite(aic_table$AIC), , drop = FALSE]
  if (nrow(ok_aic) == 0) {
    write.csv(aic_table, file.path(out_dir, "pgls_aic_scores.csv"), row.names = FALSE)
    stop("All PGLS correlation structures failed. See pgls_aic_scores.csv.", call. = FALSE)
  }

  ok_aic <- ok_aic[order(ok_aic$AIC), , drop = FALSE]
  best_model_name <- ok_aic$model[[1]]
  best_fit <- pgls_result$models[[best_model_name]]

  test$prediction_type <- ifelse(
    test$in_phylogeny,
    paste0("fixed_effect_", best_model_name),
    "fixed_effect_species_absent_from_phylogeny"
  )
  test$y_pred_log_BMR <- as.numeric(stats::predict(best_fit, newdata = test))
  test$y_pred_BMR <- exp(test$y_pred_log_BMR)

  train_phylo$y_fitted_log_BMR <- as.numeric(stats::fitted(best_fit))
  train_phylo$y_fitted_BMR <- exp(train_phylo$y_fitted_log_BMR)
  train_phylo$residual_log_BMR <- stats::residuals(best_fit, type = "response")

  test_metrics_bmr <- metrics(test$BMR, test$y_pred_BMR)
  test_metrics_log <- metrics(test$log_BMR, test$y_pred_log_BMR)
  metric_out <- rbind(
    data.frame(scale = "BMR", test_metrics_bmr),
    data.frame(scale = "log_BMR", test_metrics_log)
  )

  aic_table$delta_AIC <- aic_table$AIC - min(aic_table$AIC, na.rm = TRUE)
  aic_table <- aic_table[order(aic_table$AIC), , drop = FALSE]
  write.csv(aic_table, file.path(out_dir, "pgls_aic_scores.csv"), row.names = FALSE)
  write.csv(test, file.path(out_dir, "pgls_test_predictions.csv"), row.names = FALSE)
  write.csv(train_phylo, file.path(out_dir, "pgls_train_fitted.csv"), row.names = FALSE)
  write.csv(metric_out, file.path(out_dir, "pgls_test_metrics.csv"), row.names = FALSE)
  writeLines(capture.output(summary(best_fit)), file.path(out_dir, "pgls_best_model_summary.txt"))
  writeLines(
    c(
      "PGLS model comparison",
      sprintf("Formula: %s", args$formula),
      sprintf("Best model: %s", best_model_name),
      sprintf("Best AIC: %.6f", ok_aic$AIC[[1]]),
      sprintf("Pagel lambda residual test start value: %.6f", lambda_signal$lambda),
      sprintf("BMR test RMSE: %.6f", test_metrics_bmr$rmse),
      sprintf("BMR test R2: %.6f", test_metrics_bmr$r2),
      sprintf("log_BMR test RMSE: %.6f", test_metrics_log$rmse),
      sprintf("log_BMR test R2: %.6f", test_metrics_log$r2),
      sprintf("Bad branch count: %d", bad_branch_filter$bad_edge_count),
      sprintf("Train rows used: %d", nrow(train_phylo)),
      sprintf("Train species used: %d", length(unique(train_phylo$Species))),
      sprintf("Test rows predicted: %d", nrow(test))
    ),
    file.path(out_dir, "pgls_metrics_summary.txt")
  )

  cat("PGLS finished\n")
  cat("  Best model:", best_model_name, "\n")
  cat("  Best AIC:", sprintf("%.6f", ok_aic$AIC[[1]]), "\n")
  cat("  Test RMSE (BMR):", sprintf("%.6f", test_metrics_bmr$rmse), "\n")
  cat("  Test R2 (BMR):", sprintf("%.6f", test_metrics_bmr$r2), "\n")
  cat("  Output directory:", out_dir, "\n")
}

main()
