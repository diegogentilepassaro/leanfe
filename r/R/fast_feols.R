#' Fast High-Dimensional Fixed Effects Regression
#'
#' Main entry point for fixed effects regression using either Polars or DuckDB backend.
#' By default uses Polars for speed; use DuckDB for very large datasets.
#'
#' @description
#' This is the recommended entry point for the package. It provides a unified API
#' that dispatches to either the Polars or DuckDB backend based on the `backend` parameter.
#'
#' @param data Polars DataFrame, R data.frame, or path to parquet file.
#' @param y_col Dependent variable column name (optional if formula provided).
#' @param x_cols Character vector of independent variable column names (optional if formula provided).
#' @param fe_cols Character vector of fixed effect column names (optional if formula provided).
#' @param formula R-style formula: "y ~ x1 + x2 | fe1 + fe2" or "y ~ x | fe | z" (IV).
#'   Supports:
#'   \itemize{
#'     \item Regular variables: x1, x2
#'     \item Factor variables: i(region) - automatic dummy expansion
#'     \item Interactions: treatment:i(region) - heterogeneous effects
#'     \item Instruments (IV): third part after second |
#'   }
#' @param weights Column name for regression weights (WLS).
#' @param demean_tol Convergence tolerance for iterative demeaning (default: 1e-3).
#' @param n_iter Maximum iterations for demeaning (default: 100).
#' @param vcov Variance-covariance estimator: "iid", "HC1", or "cluster" (default: "iid").
#' @param cluster_cols Character vector of clustering variables (required if vcov="cluster").
#' @param ssc Logical, small sample correction for clustered SEs (default: FALSE).
#' @param sample_frac Fraction of data to sample, e.g., 0.1 for 10% (default: NULL).
#' @param backend Computation backend: "polars" (default) or "duckdb".
#'   \itemize{
#'     \item "polars": Faster, better for data that fits in memory
#'     \item "duckdb": Lower memory usage, better for very large datasets
#'   }
#'
#' @return A list containing:
#' \describe{
#'   \item{coefficients}{Named vector of coefficient estimates}
#'   \item{std_errors}{Named vector of standard errors}
#'   \item{n_obs}{Number of observations used}
#'   \item{iterations}{Number of demeaning iterations}
#'   \item{vcov_type}{Type of variance estimator used}
#'   \item{n_clusters}{Number of clusters (if clustered SEs)}
#' }
#'
#' @details
#' \strong{Choosing a Backend:}
#'
#' Use Polars (default) when:
#' \itemize{
#'   \item Speed is the priority
#'   \item Data fits comfortably in memory
#'   \item You need the fastest possible execution
#' }
#'
#' Use DuckDB when:
#' \itemize{
#'   \item Dataset is larger than available RAM
#'   \item Memory is constrained
#'   \item Reading directly from parquet files without loading into memory
#' }
#'
#' \strong{Performance Comparison (12.7M obs, 4 FEs):}
#' \itemize{
#'   \item Polars: ~15s, ~1 GB memory
#'   \item DuckDB: ~18s, ~700 MB memory
#' }
#'
#' @examples
#' \dontrun{
#' library(polars)
#'
#' # Basic usage with formula
#' df <- pl$read_parquet("data.parquet")
#' result <- fast_feols(df, formula = "y ~ treatment | customer + product")
#' print(result$coefficients["treatment"])
#'
#' # With clustered standard errors
#' result <- fast_feols(
#'   df,
#'   formula = "y ~ treatment | customer + product",
#'   vcov = "cluster",
#'   cluster_cols = c("customer")
#' )
#'
#' # Using DuckDB for large datasets
#' result <- fast_feols(
#'   "large_data.parquet",
#'   formula = "y ~ treatment | fe1 + fe2",
#'   backend = "duckdb"
#' )
#'
#' # Difference-in-Differences
#' result <- fast_feols(
#'   df,
#'   formula = "y ~ treated_post | state + year",
#'   vcov = "cluster",
#'   cluster_cols = c("state")
#' )
#' }
#'
#' @export
fast_feols <- function(
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
  sample_frac = NULL,
  backend = "polars"
) {
  if (!backend %in% c("polars", "duckdb")) {
    stop(sprintf("backend must be 'polars' or 'duckdb', got '%s'", backend))
  }
  
  if (backend == "polars") {
    fast_feols_polars(
      data = data,
      y_col = y_col,
      x_cols = x_cols,
      fe_cols = fe_cols,
      formula = formula,
      weights = weights,
      demean_tol = demean_tol,
      n_iter = n_iter,
      vcov = vcov,
      cluster_cols = cluster_cols,
      ssc = ssc,
      sample_frac = sample_frac
    )
  } else {
    fast_feols_duckdb(
      data = data,
      y_col = y_col,
      x_cols = x_cols,
      fe_cols = fe_cols,
      formula = formula,
      weights = weights,
      demean_tol = demean_tol,
      n_iter = n_iter,
      vcov = vcov,
      cluster_cols = cluster_cols,
      ssc = ssc,
      sample_frac = sample_frac
    )
  }
}
