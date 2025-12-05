#' Common utilities shared between Polars and DuckDB backends
#'
#' Contains formula parsing, IV estimation, and other shared logic.

#' Parse i() term with optional reference category
#'
#' @param term i() term string like "i(var)" or "i(var, ref=value)"
#' @return List with var and ref (NULL if not specified)
#' @keywords internal
.parse_i_term <- function(term) {
  # Match i(var) or i(var, ref=value) or i(var, ref="value")
  match <- regmatches(term, regexec("^i\\((\\w+)(?:\\s*,\\s*ref\\s*=\\s*[\"']?([^\"')\\s]+)[\"']?)?\\)$", term))[[1]]
  if (length(match) >= 2) {
    ref <- if (length(match) == 3 && nchar(match[3]) > 0) match[3] else NULL
    return(list(var = match[2], ref = ref))
  }
  stop(sprintf("Invalid i() syntax: %s. Use i(var) or i(var, ref=value)", term))
}


#' Parse R-style formula into components
#'
#' Supports:
#' - Basic: 'y ~ x1 + x2 | fe1 + fe2'
#' - Factor variables: 'y ~ x1 + i(region) | fe1'
#' - Factor with reference: 'y ~ x1 + i(region, ref=R1) | fe1'
#' - Interactions: 'y ~ x1 + treatment:i(region) | fe1'
#' - Interactions with reference: 'y ~ x1 + treatment:i(region, ref=R1) | fe1'
#' - IV/2SLS: 'y ~ x1 + x2 | fe1 + fe2 | z1 + z2'
#'
#' @param formula R-style formula string
#' @return List with y, X, fe, factor_vars (list of var/ref), interactions (list of var/factor/ref), instruments
#' @keywords internal
.parse_formula <- function(formula) {
  parts <- strsplit(formula, "\\|")[[1]]
  if (length(parts) < 2) {
    stop("Formula must include fixed effects: 'y ~ x | fe1 + fe2'")
  }
  if (length(parts) > 3) {
    stop("Formula has too many parts. Use: 'y ~ x | fe' or 'y ~ x | fe | z' (IV)")
  }
  
  # Parse left-hand side (y ~ x terms)
  lhs_rhs <- strsplit(trimws(parts[1]), "~")[[1]]
  if (length(lhs_rhs) != 2) {
    stop("Formula must have exactly one '~' separating y and x variables")
  }
  
  y <- trimws(lhs_rhs[1])
  
  # Parse x terms
  x_terms <- trimws(strsplit(lhs_rhs[2], "\\+")[[1]])
  X <- c()
  factor_vars <- list()
  interactions <- list()
  
  for (term in x_terms) {
    if (grepl(":i\\(.+\\)$", term)) {
      # Interaction term: var:i(factor) or var:i(factor, ref=value)
      match <- regmatches(term, regexec("^(\\w+):i\\((\\w+)(?:\\s*,\\s*ref\\s*=\\s*[\"']?([^\"')\\s]+)[\"']?)?\\)$", term))[[1]]
      if (length(match) >= 3) {
        ref <- if (length(match) == 4 && nchar(match[4]) > 0) match[4] else NULL
        interactions[[length(interactions) + 1]] <- list(var = match[2], factor = match[3], ref = ref)
      } else {
        stop(sprintf("Invalid interaction syntax: %s", term))
      }
    } else if (grepl("^i\\(.+\\)$", term)) {
      # Factor variable: i(var) or i(var, ref=value)
      parsed <- .parse_i_term(term)
      factor_vars[[length(factor_vars) + 1]] <- parsed
    } else {
      # Regular variable
      X <- c(X, term)
    }
  }
  
  # Parse fixed effects
  fe <- trimws(strsplit(parts[2], "\\+")[[1]])
  
  # Parse instruments (if present)
  instruments <- c()
  if (length(parts) == 3) {
    instruments <- trimws(strsplit(parts[3], "\\+")[[1]])
  }
  
  list(
    y = y, 
    X = X, 
    fe = fe, 
    factor_vars = factor_vars, 
    interactions = interactions,
    instruments = instruments
  )
}


#' Two-Stage Least Squares (2SLS) estimation
#'
#' @param X Endogenous regressors matrix (n x k)
#' @param Y Dependent variable vector (n)
#' @param Z Instruments matrix (n x m) where m >= k
#' @param weights Optional regression weights (n)
#' @return List with beta coefficients and X_hat fitted values
#' @keywords internal
.iv_2sls <- function(X, Y, Z, weights = NULL) {
  if (ncol(Z) < ncol(X)) {
    stop(sprintf("Under-identified: %d instruments for %d endogenous variables", ncol(Z), ncol(X)))
  }
  
  if (!is.null(weights)) {
    sqrt_w <- sqrt(weights)
    X_w <- X * sqrt_w
    Y_w <- Y * sqrt_w
    Z_w <- Z * sqrt_w
    
    # First stage: X = Z @ gamma
    ZtZ <- crossprod(Z_w)
    ZtX <- crossprod(Z_w, X_w)
    ZtZ_inv <- solve(ZtZ)
    gamma <- ZtZ_inv %*% ZtX
    X_hat <- Z %*% gamma
    
    # Second stage: Y = X_hat @ beta
    X_hat_w <- X_hat * sqrt_w
    beta <- solve(crossprod(X_hat_w), crossprod(X_hat_w, Y_w))
  } else {
    # First stage
    ZtZ <- crossprod(Z)
    ZtX <- crossprod(Z, X)
    ZtZ_inv <- solve(ZtZ)
    gamma <- ZtZ_inv %*% ZtX
    X_hat <- Z %*% gamma
    
    # Second stage
    beta <- solve(crossprod(X_hat), crossprod(X_hat, Y))
  }
  
  list(beta = beta, X_hat = X_hat)
}


#' Compute standard errors for OLS/IV coefficients
#'
#' @param XtX_inv Inverse of X'X matrix (k x k)
#' @param resid Residuals vector (n)
#' @param n_obs Number of observations
#' @param df_resid Residual degrees of freedom
#' @param vcov Type of variance estimator: "iid", "HC1", or "cluster"
#' @param X Regressor matrix (n x k) - use X_hat for IV
#' @param weights Optional regression weights (n)
#' @param cluster_ids Optional cluster identifiers (n)
#' @param ssc Small sample correction for clustered SEs
#' @return List with se (standard errors) and n_clusters
#' @keywords internal
.compute_standard_errors <- function(
  XtX_inv,
  resid,
  n_obs,
  df_resid,
  vcov,
  X,
  weights = NULL,
  cluster_ids = NULL,
  ssc = FALSE
) {
  n_clusters <- NULL
  
  if (vcov == "iid") {
    if (!is.null(weights)) {
      sigma2 <- sum(weights * resid^2) / df_resid
    } else {
      sigma2 <- sum(resid^2) / df_resid
    }
    se <- sqrt(diag(XtX_inv) * sigma2)
    
  } else if (vcov == "HC1") {
    if (!is.null(weights)) {
      meat <- crossprod(X, X * as.vector(weights * resid^2))
    } else {
      meat <- crossprod(X, X * as.vector(resid^2))
    }
    vcov_matrix <- XtX_inv %*% meat %*% XtX_inv
    adjustment <- n_obs / df_resid
    se <- sqrt(diag(vcov_matrix) * adjustment)
    
  } else if (vcov == "cluster") {
    if (is.null(cluster_ids)) {
      stop("cluster_ids required for vcov='cluster'")
    }
    
    # Get unique clusters
    unique_clusters <- unique(cluster_ids)
    n_clusters <- length(unique_clusters)
    k <- ncol(X)
    
    # Compute cluster scores: score_g = sum(X_i * e_i) for each cluster g
    scores <- matrix(0, n_clusters, k)
    for (g in seq_along(unique_clusters)) {
      mask <- cluster_ids == unique_clusters[g]
      if (!is.null(weights)) {
        scores[g, ] <- colSums(X[mask, , drop = FALSE] * (resid[mask] * weights[mask]))
      } else {
        scores[g, ] <- colSums(X[mask, , drop = FALSE] * resid[mask])
      }
    }
    
    # Meat matrix: S'S
    meat <- crossprod(scores)
    
    # Cluster adjustment
    if (ssc) {
      adjustment <- (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
    } else {
      adjustment <- n_clusters / (n_clusters - 1)
    }
    
    vcov_matrix <- XtX_inv %*% meat %*% XtX_inv * adjustment
    se <- sqrt(diag(vcov_matrix))
    
  } else {
    stop(sprintf("vcov must be 'iid', 'HC1', or 'cluster', got '%s'", vcov))
  }
  
  list(se = se, n_clusters = n_clusters)
}


#' Build standardized result list
#'
#' @param x_cols Variable names
#' @param beta Coefficient vector
#' @param se Standard error vector
#' @param n_obs Number of observations
#' @param iterations Number of demeaning iterations
#' @param vcov Variance estimator type
#' @param is_iv Whether IV was used
#' @param n_instruments Number of instruments (if IV)
#' @param n_clusters Number of clusters (if clustered)
#' @return Standardized result list
#' @keywords internal
.build_result <- function(
  x_cols,
  beta,
  se,
  n_obs,
  iterations,
  vcov,
  is_iv,
  n_instruments,
  n_clusters
) {
  list(
    coefficients = setNames(as.vector(beta), x_cols),
    std_errors = setNames(as.vector(se), x_cols),
    n_obs = n_obs,
    iterations = iterations,
    vcov_type = vcov,
    is_iv = is_iv,
    n_instruments = if (is_iv) n_instruments else NULL,
    n_clusters = n_clusters
  )
}
