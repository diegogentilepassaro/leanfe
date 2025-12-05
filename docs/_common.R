# Common R setup for leanfe documentation
# Source this file at the beginning of R code blocks to load leanfe functions

# Get the package R directory relative to docs
pkg_dir <- normalizePath(file.path(dirname(getwd()), "r", "R"), mustWork = FALSE)
if (!dir.exists(pkg_dir)) {
  # Try alternative path (when running from docs directory)
  pkg_dir <- normalizePath("../r/R", mustWork = FALSE)
}

# Source all R files
if (dir.exists(pkg_dir)) {
  source(file.path(pkg_dir, "common.R"))
  source(file.path(pkg_dir, "duckdb.R"))
  source(file.path(pkg_dir, "polars.R"))
  source(file.path(pkg_dir, "fast_feols.R"))
} else {
  warning("Could not find leanfe R package directory")
}
