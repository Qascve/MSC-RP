#!/usr/bin/env Rscript

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  values <- list(
    tree = "data/phylogeny/unique_taxon_names.nwk",
    predictions = "results/benchmark/all/benchmark_predictions_test.csv",
    output_dir = "results/plots"
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!key %in% c("--tree", "--predictions", "--output-dir")) {
      stop("Unknown argument: ", key, call. = FALSE)
    }
    if (i == length(args)) {
      stop("Missing value for argument: ", key, call. = FALSE)
    }
    values[[sub("^--", "", gsub("-", "_", key))]] <- args[[i + 1]]
    i <- i + 2
  }
  values
}

find_root <- function(marker = ".gitignore") {
  file_arg <- commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))]
  script_dir <- if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
  } else {
    getwd()
  }
  starts <- c(getwd(), script_dir)
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

resolve_path <- function(root, path) {
  if (grepl("^([A-Za-z]:)?[\\\\/]", path)) {
    return(path)
  }
  file.path(root, path)
}

require_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Missing R packages: ", paste(missing, collapse = ", "), "\n",
      "Install with:\n",
      "install.packages(c('ggplot2', 'BiocManager'))\n",
      "BiocManager::install(c('ggtree', 'treeio'))",
      call. = FALSE
    )
  }
}

taxon_to_tip_label <- function(x) {
  gsub(" ", "_", trimws(as.character(x)), fixed = TRUE)
}

fold_accuracy <- function(y_true, y_pred) {
  y_true <- suppressWarnings(as.numeric(y_true))
  y_pred <- suppressWarnings(as.numeric(y_pred))
  out <- rep(NA_real_, length(y_true))
  valid <- is.finite(y_true) & is.finite(y_pred) & y_true > 0 & y_pred > 0
  ratio <- y_pred[valid] / y_true[valid]
  out[valid] <- pmin(ratio, 1 / ratio)
  pmin(pmax(out, 0), 1)
}

compute_node_accuracy <- function(tree, tip_accuracy) {
  total_nodes <- length(tree$tip.label) + tree$Nnode
  node_accuracy <- rep(NA_real_, total_nodes)
  names(node_accuracy) <- as.character(seq_len(total_nodes))

  tip_values <- tip_accuracy[tree$tip.label]
  node_accuracy[seq_along(tree$tip.label)] <- as.numeric(tip_values)

  children_by_parent <- split(tree$edge[, 2], tree$edge[, 1])
  compute_one <- function(node) {
    key <- as.character(node)
    if (is.finite(node_accuracy[key])) {
      return(node_accuracy[key])
    }
    children <- children_by_parent[[key]]
    if (is.null(children)) {
      return(node_accuracy[key])
    }
    child_values <- vapply(children, compute_one, numeric(1))
    child_values <- child_values[is.finite(child_values)]
    node_accuracy[key] <<- if (length(child_values) > 0) mean(child_values) else NA_real_
    node_accuracy[key]
  }

  root_node <- length(tree$tip.label) + 1
  compute_one(root_node)
  node_accuracy
}

main <- function() {
  require_packages(c("ape", "ggplot2", "ggtree", "treeio"))
  args <- parse_args()
  root <- find_root()
  tree_path <- resolve_path(root, args$tree)
  predictions_path <- resolve_path(root, args$predictions)
  output_dir <- resolve_path(root, args$output_dir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  tree <- ape::read.tree(tree_path)
  invisible(treeio::read.newick(tree_path))
  predictions <- read.csv(predictions_path, stringsAsFactors = FALSE)
  required <- c("taxon_name", "y_true", "xgboost")
  missing <- setdiff(required, names(predictions))
  if (length(missing) > 0) {
    stop(basename(predictions_path), " missing required columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  predictions$tip_label <- taxon_to_tip_label(predictions$taxon_name)
  predictions$accuracy <- fold_accuracy(predictions$y_true, predictions$xgboost)
  tip_accuracy <- predictions$accuracy
  names(tip_accuracy) <- predictions$tip_label

  predicted_tips <- names(tip_accuracy)[is.finite(tip_accuracy)]
  keep_tips <- intersect(tree$tip.label, predicted_tips)
  if (length(keep_tips) == 0) {
    stop("No prediction rows matched tree tip labels.", call. = FALSE)
  }
  drop_tips <- setdiff(tree$tip.label, keep_tips)
  if (length(drop_tips) > 0) {
    tree <- ape::drop.tip(tree, drop_tips)
  }

  node_accuracy <- compute_node_accuracy(tree, tip_accuracy)
  matched_tips <- sum(is.finite(node_accuracy[seq_along(tree$tip.label)]))

  node_data <- data.frame(
    node = seq_along(node_accuracy),
    accuracy = as.numeric(node_accuracy)
  )

  suppressPackageStartupMessages({
    library(ggplot2)
    library(ggtree)
  })
  plot <- ggtree(tree, layout = "circular", aes(color = accuracy), size = 0.25) %<+% node_data +
    scale_color_gradientn(
      colours = c("#7AD7D3", "#8A8A8A", "#F1055B"),
      limits = c(0, 1),
      na.value = "grey80",
      name = "Prediction\naccuracy"
    ) +
    ggtitle("XGB residual-learning accuracy across test-set species") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      legend.position = "right"
    )

  output_path <- file.path(output_dir, "xgb_residual_phylogeny_accuracy.png")
  ggsave(output_path, plot = plot, width = 10, height = 10, dpi = 300, bg = "white")
  message("Plotted test-set tips: ", matched_tips)
  message("Saved PNG: ", output_path)
}

main()
