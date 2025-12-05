#' @importFrom DBI dbConnect dbDisconnect dbExecute dbGetQuery dbWriteTable
#' @importFrom duckdb duckdb
NULL

#' Fast Fixed Effects OLS using DuckDB Backend
#'
#' DuckDB-based implementation optimized for memory efficiency. Use \code{\link{fast_feols}} 
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
#' @seealso \code{\link{fast_feols}}, \code{\link{fast_feols_polars}}
#' @export
fast_feols_duckdb <- function(
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
  
  # Build needed columns
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
  
  # Setup DuckDB connection
  con <- dbConnect(duckdb())
  on.exit(dbDisconnect(con, shutdown = TRUE))
  
  # Load data efficiently based on type
  if (is.character(data)) {
    col_list <- paste(needed_cols, collapse = ", ")
    dbExecute(con, sprintf("CREATE TABLE data AS SELECT %s FROM read_parquet('%s')", col_list, data))
  } else if ("polars_data_frame" %in% class(data)) {
    data_df <- as.data.frame(data$select(needed_cols))
    dbWriteTable(con, "data", data_df, overwrite = TRUE)
  } else {
    dbWriteTable(con, "data", data[, needed_cols, drop = FALSE], overwrite = TRUE)
  }
  
  # Sample if requested
  if (!is.null(sample_frac)) {
    dbExecute(con, sprintf("CREATE TABLE data_sampled AS SELECT * FROM data USING SAMPLE %.1f%%", sample_frac * 100))
    dbExecute(con, "DROP TABLE data")
    dbExecute(con, "ALTER TABLE data_sampled RENAME TO data")
  }
  
  # Handle interactions
  if (length(interactions) > 0) {
    for (interaction in interactions) {
      var <- interaction$var
      factor <- interaction$factor
      ref <- interaction$ref
      categories <- dbGetQuery(con, sprintf("SELECT DISTINCT %s FROM data ORDER BY %s", factor, factor))[[1]]
      
      # Determine reference category
      if (is.null(ref)) {
        ref_cat <- categories[1]  # Default: first category
      } else {
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
        dbExecute(con, sprintf("ALTER TABLE data ADD COLUMN %s DOUBLE", col_name))
        dbExecute(con, sprintf("UPDATE data SET %s = CASE WHEN %s = '%s' THEN %s ELSE 0 END", col_name, factor, cat, var))
        x_cols <- c(x_cols, col_name)
      }
    }
  }
  
  # Handle factor variables
  if (length(factor_vars) > 0) {
    for (fv in factor_vars) {
      var <- fv$var
      ref <- fv$ref
      categories <- dbGetQuery(con, sprintf("SELECT DISTINCT %s FROM data ORDER BY %s", var, var))[[1]]
      
      # Determine reference category
      if (is.null(ref)) {
        ref_cat <- categories[1]  # Default: first category
      } else {
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
        col_name <- paste0(var, "_", cat)
        dbExecute(con, sprintf("ALTER TABLE data ADD COLUMN %s DOUBLE", col_name))
        dbExecute(con, sprintf("UPDATE data SET %s = CASE WHEN %s = '%s' THEN 1 ELSE 0 END", col_name, var, cat))
        x_cols <- c(x_cols, col_name)
      }
    }
  }
  
  # Drop singletons
  for (fe in fe_cols) {
    dbExecute(con, sprintf(
      "DELETE FROM data WHERE %s IN (SELECT %s FROM data GROUP BY %s HAVING COUNT(*) = 1)",
      fe, fe, fe
    ))
  }
  
  n_obs <- dbGetQuery(con, "SELECT COUNT(*) FROM data")[[1]]
  cols_to_demean <- c(y_col, x_cols, instruments)
  
  # Add demeaned columns
  for (col in cols_to_demean) {
    dbExecute(con, sprintf("ALTER TABLE data ADD COLUMN %s_dm DOUBLE", col))
    dbExecute(con, sprintf("UPDATE data SET %s_dm = %s", col, col))
  }
  
  dm_cols <- paste0(cols_to_demean, "_dm")
  
  # Iterative demeaning
  for (it in 1:n_iter) {
    for (fe in fe_cols) {
      if (!is.null(weights)) {
        for (col in dm_cols) {
          dbExecute(con, sprintf("
            UPDATE data SET %s = %s - (
              SELECT SUM(d2.%s * d2.%s) / SUM(d2.%s)
              FROM data d2 WHERE d2.%s = data.%s
            )
          ", col, col, col, weights, weights, fe, fe))
        }
      } else {
        for (col in dm_cols) {
          dbExecute(con, sprintf("
            WITH fe_means AS (SELECT %s, AVG(%s) as mean_val FROM data GROUP BY %s)
            UPDATE data SET %s = data.%s - fe_means.mean_val
            FROM fe_means WHERE data.%s = fe_means.%s
          ", fe, col, fe, col, col, fe, fe))
        }
      }
    }
    
    if (it >= 3) {
      max_mean <- 0
      for (fe in fe_cols) {
        for (col in dm_cols) {
          mean_val <- dbGetQuery(con, sprintf(
            "SELECT MAX(ABS(avg_val)) FROM (SELECT AVG(%s) as avg_val FROM data GROUP BY %s)",
            col, fe
          ))[[1]]
          max_mean <- max(max_mean, abs(mean_val))
        }
      }
      if (max_mean < demean_tol) break
    }
  }
  
  k <- length(x_cols)
  is_iv <- length(instruments) > 0
  
  if (is_iv) {
    # Extract data for IV estimation
    select_cols <- c(paste0(y_col, "_dm"), paste0(x_cols, "_dm"), paste0(instruments, "_dm"))
    if (!is.null(weights)) select_cols <- c(select_cols, weights)
    if (!is.null(cluster_cols)) select_cols <- c(select_cols, cluster_cols)
    
    result_df <- dbGetQuery(con, sprintf("SELECT %s FROM data", paste(select_cols, collapse = ", ")))
    
    Y <- result_df[[paste0(y_col, "_dm")]]
    X <- as.matrix(result_df[, paste0(x_cols, "_dm"), drop = FALSE])
    Z <- as.matrix(result_df[, paste0(instruments, "_dm"), drop = FALSE])
    w <- if (!is.null(weights)) result_df[[weights]] else NULL
    
    iv_result <- .iv_2sls(X, Y, Z, w)
    beta <- iv_result$beta
    X_hat <- iv_result$X_hat
    resid <- as.vector(Y - X_hat %*% beta)
  } else {
    # OLS via SQL aggregates
    XtX <- matrix(0, k, k)
    Xty <- numeric(k)
    
    for (i in seq_along(x_cols)) {
      col_i_dm <- paste0(x_cols[i], "_dm")
      Xty[i] <- dbGetQuery(con, sprintf("SELECT SUM(%s * %s_dm) FROM data", col_i_dm, y_col))[[1]]
      for (j in i:k) {
        col_j_dm <- paste0(x_cols[j], "_dm")
        val <- dbGetQuery(con, sprintf("SELECT SUM(%s * %s) FROM data", col_i_dm, col_j_dm))[[1]]
        XtX[i, j] <- val
        XtX[j, i] <- val
      }
    }
    
    beta <- solve(XtX, Xty)
    
    # Compute residuals in SQL
    resid_terms <- sapply(seq_along(x_cols), function(i) {
      sprintf("%.15g * %s_dm", beta[i], x_cols[i])
    })
    resid_expr <- sprintf("%s_dm - (%s)", y_col, paste(resid_terms, collapse = " + "))
    dbExecute(con, "ALTER TABLE data ADD COLUMN _resid DOUBLE")
    dbExecute(con, sprintf("UPDATE data SET _resid = %s", resid_expr))
  }
  
  # Degrees of freedom
  n_fe_groups <- sum(sapply(fe_cols, function(fe) {
    dbGetQuery(con, sprintf("SELECT COUNT(DISTINCT %s) FROM data", fe))[[1]]
  }))
  absorbed_df <- n_fe_groups - length(fe_cols)
  df_resid <- n_obs - k - absorbed_df
  
  # Compute XtX_inv
  if (is_iv) {
    if (!is.null(w)) {
      sqrt_w <- sqrt(w)
      X_hat_w <- X_hat * sqrt_w
      XtX_inv <- solve(crossprod(X_hat_w))
    } else {
      XtX_inv <- solve(crossprod(X_hat))
    }
  } else {
    XtX_inv <- solve(XtX)
  }
  
  # Standard errors
  if (is_iv) {
    cluster_ids <- NULL
    if (vcov == "cluster") {
      if (is.null(cluster_cols)) stop("cluster_cols required for vcov='cluster'")
      if (length(cluster_cols) == 1) {
        cluster_ids <- result_df[[cluster_cols[1]]]
      } else {
        cluster_ids <- apply(result_df[, cluster_cols, drop = FALSE], 1, paste, collapse = "_")
      }
    }
    
    se_result <- .compute_standard_errors(
      XtX_inv = XtX_inv, resid = resid, n_obs = n_obs, df_resid = df_resid,
      vcov = vcov, X = X_hat, weights = w, cluster_ids = cluster_ids, ssc = ssc
    )
    se <- se_result$se
    n_clusters <- se_result$n_clusters
  } else {
    n_clusters <- NULL
    if (vcov == "iid") {
      if (!is.null(weights)) {
        sigma2 <- dbGetQuery(con, sprintf("SELECT SUM(%s * _resid * _resid) / %d FROM data", weights, df_resid))[[1]]
      } else {
        sigma2 <- dbGetQuery(con, sprintf("SELECT SUM(_resid * _resid) / %d FROM data", df_resid))[[1]]
      }
      se <- sqrt(sigma2 * diag(XtX_inv))
      
    } else if (vcov == "HC1") {
      meat <- matrix(0, k, k)
      for (i in seq_along(x_cols)) {
        col_i_dm <- paste0(x_cols[i], "_dm")
        for (j in i:k) {
          col_j_dm <- paste0(x_cols[j], "_dm")
          if (!is.null(weights)) {
            val <- dbGetQuery(con, sprintf("SELECT SUM(%s * %s * %s * _resid * _resid) FROM data", weights, col_i_dm, col_j_dm))[[1]]
          } else {
            val <- dbGetQuery(con, sprintf("SELECT SUM(%s * %s * _resid * _resid) FROM data", col_i_dm, col_j_dm))[[1]]
          }
          meat[i, j] <- val
          meat[j, i] <- val
        }
      }
      vcov_matrix <- XtX_inv %*% meat %*% XtX_inv
      se <- sqrt((n_obs / df_resid) * diag(vcov_matrix))
      
    } else if (vcov == "cluster") {
      if (is.null(cluster_cols)) stop("cluster_cols required for vcov='cluster'")
      
      cluster_expr <- if (length(cluster_cols) == 1) cluster_cols[1] else sprintf("CONCAT_WS('_', %s)", paste(cluster_cols, collapse = ", "))
      
      if (!is.null(weights)) {
        score_exprs <- sapply(seq_along(x_cols), function(i) sprintf("SUM(%s_dm * _resid * %s) AS score_%d", x_cols[i], weights, i))
      } else {
        score_exprs <- sapply(seq_along(x_cols), function(i) sprintf("SUM(%s_dm * _resid) AS score_%d", x_cols[i], i))
      }
      
      cluster_scores <- dbGetQuery(con, sprintf(
        "SELECT %s AS cluster_id, %s FROM data GROUP BY %s",
        cluster_expr, paste(score_exprs, collapse = ", "), cluster_expr
      ))
      n_clusters <- nrow(cluster_scores)
      
      S <- as.matrix(cluster_scores[, paste0("score_", seq_along(x_cols)), drop = FALSE])
      meat <- crossprod(S)
      
      adj <- if (ssc) (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid) else n_clusters / (n_clusters - 1)
      vcov_matrix <- adj * XtX_inv %*% meat %*% XtX_inv
      se <- sqrt(diag(vcov_matrix))
    } else {
      stop(paste("Unknown vcov:", vcov))
    }
  }
  
  .build_result(
    x_cols = x_cols, beta = beta, se = se, n_obs = n_obs, iterations = it,
    vcov = vcov, is_iv = is_iv, n_instruments = if (is_iv) length(instruments) else NULL, n_clusters = n_clusters
  )
}
