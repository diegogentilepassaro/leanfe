#' @importFrom polars pl
NULL


#' Expand factor variables into dummy variables
#'
#' @param df Polars DataFrame
#' @param factor_vars List of list(var, ref) where ref is NULL or reference category
#' @return List with df and dummy_cols
#' @keywords internal
.expand_factors_polars <- function(df, factor_vars) {
  dummy_cols <- c()
  for (fv in factor_vars) {
    var <- fv$var
    ref <- fv$ref
    categories <- as.vector(df$select(pl$col(var)$unique()$sort())$to_series())
    
    # Determine reference category
    if (is.null(ref)) {
      ref_cat <- categories[1]  # Default: first category
    } else {
      # Try to match type
      ref_cat <- ref
      if (length(categories) > 0 && !is.character(categories[1])) {
        ref_cat <- tryCatch(as(ref, class(categories[1])), error = function(e) ref)
      }
      if (!(ref_cat %in% categories)) {
        stop(sprintf("Reference category '%s' not found in %s. Available: %s", 
                     ref, var, paste(categories, collapse = ", ")))
      }
    }
    
    for (cat in categories) {
      if (identical(cat, ref_cat)) next  # Skip reference category
      dummy_name <- paste0(var, "_", cat)
      df <- df$with_columns(
        (pl$col(var) == cat)$cast(pl$Int8)$alias(dummy_name)
      )
      dummy_cols <- c(dummy_cols, dummy_name)
    }
  }
  list(df = df, dummy_cols = dummy_cols)
}


#' Expand interaction terms into dummy variables
#'
#' @param df Polars DataFrame
#' @param interactions List of list(var, factor, ref) where ref is NULL or reference category
#' @return List with df and interaction_cols
#' @keywords internal
.expand_interactions_polars <- function(df, interactions) {
  interaction_cols <- c()
  for (interaction in interactions) {
    var <- interaction$var
    factor <- interaction$factor
    ref <- interaction$ref
    categories <- as.vector(df$select(pl$col(factor)$unique()$sort())$to_series())
    
    # Determine reference category
    if (is.null(ref)) {
      ref_cat <- categories[1]  # Default: first category
    } else {
      # Try to match type
      ref_cat <- ref
      if (length(categories) > 0 && !is.character(categories[1])) {
        ref_cat <- tryCatch(as(ref, class(categories[1])), error = function(e) ref)
      }
      if (!(ref_cat %in% categories)) {
        stop(sprintf("Reference category '%s' not found in %s. Available: %s", 
                     ref, factor, paste(categories, collapse = ", ")))
      }
    }
    
    for (cat in categories) {
      if (identical(cat, ref_cat)) next  # Skip reference category
      col_name <- paste0(var, "_", cat)
      df <- df$with_columns(
        (pl$col(var) * (pl$col(factor) == cat))$cast(pl$Float64)$alias(col_name)
      )
      interaction_cols <- c(interaction_cols, col_name)
    }
  }
  list(df = df, interaction_cols = interaction_cols)
}


.optimize_dtypes_polars <- function(df, fe_cols) {
  schema <- df$schema
  for (col in df$columns) {
    dtype_str <- capture.output(print(schema[[col]]))[1]
    if (grepl("^Int", dtype_str)) {
      col_stats <- df$select(pl$col(col)$min()$alias("min"), 
                             pl$col(col)$max()$alias("max"))
      stats_df <- as.data.frame(col_stats)
      col_min <- stats_df$min[1]
      col_max <- stats_df$max[1]
      if (!is.na(col_min) && !is.na(col_max)) {
        if (col_min >= -128 && col_max <= 127) {
          df <- df$with_columns(pl$col(col)$cast(pl$Int8))
        } else if (col_min >= -32768 && col_max <= 32767) {
          df <- df$with_columns(pl$col(col)$cast(pl$Int16))
        } else if (col_min >= -2147483648 && col_max <= 2147483647) {
          df <- df$with_columns(pl$col(col)$cast(pl$Int32))
        }
      }
    }
    if (col %in% fe_cols && dtype_str == "String") {
      df <- df$with_columns(pl$col(col)$cast(pl$Categorical()))
    }
  }
  df
}


#' Fast Fixed Effects OLS using Polars Backend
#'
#' Polars-based implementation optimized for speed. Use \code{\link{fast_feols}} 
#' for the recommended unified API.
#'
#' @param data Polars DataFrame, R data.frame, or path to parquet file.
#' @param y_col Dependent variable column name (optional if formula provided).
#' @param x_cols Character vector of independent variable column names (optional if formula provided).
#' @param fe_cols Character vector of fixed effect column names (optional if formula provided).
#' @param formula R-style formula supporting:
#'   \itemize{
#'     \item Basic: "y ~ x1 + x2 | fe1 + fe2"
#'     \item Factor variables: "y ~ x + i(region) | fe" or "y ~ x + i(region, ref=R2) | fe"
#'     \item Interactions: "y ~ treatment:i(region) | fe"
#'     \item IV/2SLS: "y ~ x | fe | z" (instruments after second |)
#'   }
#' @param weights Column name for regression weights (WLS).
#' @param demean_tol Convergence tolerance for demeaning (default: 1e-3).
#' @param n_iter Maximum demeaning iterations (default: 100).
#' @param vcov Variance estimator: "iid", "HC1", or "cluster" (default: "iid").
#' @param cluster_cols Character vector of clustering variables (required if vcov="cluster").
#' @param ssc Logical, small sample correction for clustered SEs (default: FALSE).
#' @param sample_frac Fraction of data to sample, e.g., 0.1 for 10% (default: NULL).
#'
#' @return A list containing:
#' \describe{
#'   \item{coefficients}{Named vector of coefficient estimates}
#'   \item{std_errors}{Named vector of standard errors}
#'   \item{n_obs}{Number of observations used}
#'   \item{iterations}{Number of demeaning iterations}
#'   \item{vcov_type}{Type of variance estimator used}
#'   \item{is_iv}{Logical, whether IV/2SLS was used}
#'   \item{n_instruments}{Number of instruments (if IV)}
#'   \item{n_clusters}{Number of clusters (if clustered SEs)}
#' }
#'
#' @seealso \code{\link{fast_feols}}, \code{\link{fast_feols_duckdb}}
#' @export
fast_feols_polars <- function(
  data,
  y_col = NULL,
  x_cols = NULL,
  fe_cols = NULL,
  formula = NULL,
  weights = NULL,
  demean_tol = 1e-3,
  n_iter = 100,
  vcov = "iid",
  cluster_cols = NULL,
  ssc = FALSE,
  sample_frac = NULL
) {
  # Parse formula if provided
  if (!is.null(formula)) {
    parsed <- .parse_formula(formula)
    y_col <- parsed$y
    x_cols <- parsed$X
    fe_cols <- parsed$fe
    factor_vars <- parsed$factor_vars
    interactions <- parsed$interactions
    instruments <- parsed$instruments
  } else if (is.null(y_col) || is.null(x_cols) || is.null(fe_cols)) {
    stop("Must provide either 'formula' or (y_col, x_cols, fe_cols)")
  } else {
    factor_vars <- c()
    interactions <- list()
    instruments <- c()
  }
  
  # Build list of needed columns
  needed_cols <- c(y_col, x_cols, fe_cols, instruments)
  # Add factor variable columns
  for (fv in factor_vars) {
    if (!(fv$var %in% needed_cols)) {
      needed_cols <- c(needed_cols, fv$var)
    }
  }
  # Add interaction variable columns
  for (interaction in interactions) {
    if (!(interaction$var %in% needed_cols)) {
      needed_cols <- c(needed_cols, interaction$var)
    }
    if (!(interaction$factor %in% needed_cols)) {
      needed_cols <- c(needed_cols, interaction$factor)
    }
  }
  if (!is.null(cluster_cols)) {
    needed_cols <- unique(c(needed_cols, cluster_cols))
  }
  if (!is.null(weights)) {
    needed_cols <- unique(c(needed_cols, weights))
  }
  
  # Load data
  if (is.character(data)) {
    df <- pl$scan_parquet(data)$select(needed_cols)$collect()
  } else if ("polars_data_frame" %in% class(data)) {
    df <- data$select(needed_cols)
  } else {
    # Convert R data.frame to polars
    df <- pl$DataFrame(data)$select(needed_cols)
  }
  
  # Expand interactions
  if (length(interactions) > 0) {
    expanded <- .expand_interactions_polars(df, interactions)
    df <- expanded$df
    x_cols <- c(x_cols, expanded$interaction_cols)
  }
  
  # Check for continuous treatment variables
  continuous_treatments <- c()
  for (col in x_cols) {
    dtype <- df$schema[[col]]
    if (identical(dtype, pl$Float64) || identical(dtype, pl$Float32)) {
      n_unique <- as.data.frame(df$select(pl$col(col)$n_unique()))[[1]]
      if (n_unique > 10) {
        continuous_treatments <- c(continuous_treatments, col)
      }
    }
  }
  if (length(continuous_treatments) > 0) {
    warning(sprintf(
      "Continuous treatment variable(s) detected: %s.",
      paste(continuous_treatments, collapse = ", ")
    ), call. = FALSE)
  }
  
  # Optimize types (always done for memory efficiency)
  factor_var_names <- sapply(factor_vars, function(fv) fv$var)
  df <- .optimize_dtypes_polars(df, c(fe_cols, factor_var_names))
  
  # Sample data if requested
  if (!is.null(sample_frac)) {
    df <- df$sample(fraction = sample_frac, seed = 42)
  }
  
  # Expand factor variables into dummies
  if (length(factor_vars) > 0) {
    expanded <- .expand_factors_polars(df, factor_vars)
    df <- expanded$df
    x_cols <- c(x_cols, expanded$dummy_cols)
  }
  
  # Extract weights if provided
  if (!is.null(weights)) {
    w <- as.numeric(as.vector(df$select(weights)$to_series()))
  } else {
    w <- NULL
  }
  
  # Drop singletons
  prev_height <- df$height + 1
  while (df$height < prev_height) {
    prev_height <- df$height
    for (fe in fe_cols) {
      counts <- df$group_by(fe)$agg(pl$len()$alias("cnt"))
      df <- df$join(counts, on = fe, how = "left")$filter(pl$col("cnt") > 1)$drop("cnt")
    }
  }
  
  n_obs <- df$height
  cols_to_demean <- c(y_col, x_cols, instruments)
  
  # FWL demeaning
  for (it in 1:n_iter) {
    for (fe in fe_cols) {
      if (!is.null(weights)) {
        agg_exprs <- lapply(cols_to_demean, function(c) {
          (pl$col(c) * pl$col(weights))$sum()$truediv(pl$col(weights)$sum())$alias(paste0(c, "_mean"))
        })
      } else {
        agg_exprs <- lapply(cols_to_demean, function(c) {
          pl$col(c)$mean()$alias(paste0(c, "_mean"))
        })
      }
      means <- df$group_by(fe)$agg(!!!agg_exprs)
      df <- df$join(means, on = fe, how = "left")
      for (c in cols_to_demean) {
        df <- df$with_columns(
          (pl$col(c) - pl$col(paste0(c, "_mean")))$alias(c)
        )$drop(paste0(c, "_mean"))
      }
    }
    if (it >= 3) {
      max_mean <- max(sapply(fe_cols, function(fe) {
        as.data.frame(df$group_by(fe)$agg(pl$col(y_col)$mean()$alias("m"))$
          select(pl$col("m")$abs()$max()))[[1]]
      }))
      if (max_mean < demean_tol) break
    }
  }
  
  # Extract X and Y for OLS/IV solve
  X <- as.matrix(as.data.frame(df$select(x_cols)))
  Y <- as.data.frame(df$select(y_col))[[1]]
  
  # IV/2SLS or OLS
  is_iv <- length(instruments) > 0
  if (is_iv) {
    Z <- as.matrix(as.data.frame(df$select(instruments)))
    iv_result <- .iv_2sls(X, Y, Z, w)
    beta <- iv_result$beta
    X_hat <- iv_result$X_hat
  } else {
    if (!is.null(w)) {
      sqrt_w <- sqrt(w)
      X_w <- X * sqrt_w
      Y_w <- Y * sqrt_w
      XtX <- crossprod(X_w)
      Xty <- crossprod(X_w, Y_w)
    } else {
      XtX <- crossprod(X)
      Xty <- crossprod(X, Y)
    }
    beta <- solve(XtX, Xty)
    X_hat <- X
  }
  
  # Compute (X'X)^-1 for standard errors
  if (!is.null(w)) {
    sqrt_w <- sqrt(w)
    X_hat_w <- X_hat * sqrt_w
    XtX_inv <- solve(crossprod(X_hat_w))
  } else {
    XtX_inv <- solve(crossprod(X_hat))
  }
  
  # Residuals
  resid <- as.vector(Y - X_hat %*% beta)
  
  # Calculate absorbed degrees of freedom
  fe_counts <- sapply(fe_cols, function(fe) {
    df$select(pl$col(fe))$unique()$height
  })
  absorbed_df <- sum(fe_counts) - length(fe_cols)
  df_resid <- n_obs - length(x_cols) - absorbed_df
  
  # Build cluster IDs if needed
  cluster_ids <- NULL
  if (vcov == "cluster") {
    if (is.null(cluster_cols)) {
      stop("cluster_cols must be provided when vcov='cluster'")
    }
    if (length(cluster_cols) == 1) {
      cluster_ids <- as.vector(df$select(cluster_cols[1])$to_series())
    } else {
      cluster_exprs <- lapply(cluster_cols, function(c) pl$col(c)$cast(pl$String))
      cluster_ids <- as.vector(df$with_columns(
        pl$concat_str(cluster_exprs, separator = "_")$alias("_cluster_id")
      )$select("_cluster_id")$to_series())
    }
  }
  
  # Compute standard errors
  se_result <- .compute_standard_errors(
    XtX_inv = XtX_inv,
    resid = resid,
    n_obs = n_obs,
    df_resid = df_resid,
    vcov = vcov,
    X = X_hat,
    weights = w,
    cluster_ids = cluster_ids,
    ssc = ssc
  )
  
  .build_result(
    x_cols = x_cols,
    beta = beta,
    se = se_result$se,
    n_obs = n_obs,
    iterations = it,
    vcov = vcov,
    is_iv = is_iv,
    n_instruments = if (is_iv) length(instruments) else NULL,
    n_clusters = se_result$n_clusters
  )
}
