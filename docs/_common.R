# Common R setup for leanfe documentation
# Source this file at the beginning of R code blocks to load leanfe functions

# Configure reticulate to use system Python (for CI environments)
if (Sys.getenv("RETICULATE_PYTHON") != "") {
  library(reticulate)
  use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
}

# Find the project root by looking for _quarto.yml
.leanfe_find_project_root <- function(start_dir = getwd()) {
  dir <- normalizePath(start_dir, mustWork = FALSE)
  while (dir != dirname(dir)) {  # Stop at filesystem root
    if (file.exists(file.path(dir, "_quarto.yml"))) {
      return(dir)
    }
    dir <- dirname(dir)
  }
  NULL
}

.leanfe_project_root <- .leanfe_find_project_root()

if (!is.null(.leanfe_project_root)) {
  # Project root is docs/, R package is at ../r/R relative to docs/
  .leanfe_pkg_dir <- file.path(.leanfe_project_root, "..", "r", "R")
} else {
  # Fallback: try relative paths from working directory
  .leanfe_pkg_dir <- NULL
  .leanfe_possible_paths <- c(
    "../../r/R",        # From docs/benchmarks/ or docs/tutorials/ etc.
    "../r/R",           # From docs/ directory
    "r/R",              # From package/ directory
    "../../../r/R"      # From deeper subdirectories
  )
  for (.leanfe_path in .leanfe_possible_paths) {
    if (dir.exists(.leanfe_path) && file.exists(file.path(.leanfe_path, "common.R"))) {
      .leanfe_pkg_dir <- .leanfe_path
      break
    }
  }
}

# Source all R files in correct dependency order
if (!is.null(.leanfe_pkg_dir) && dir.exists(.leanfe_pkg_dir)) {
  .leanfe_pkg_dir <- normalizePath(.leanfe_pkg_dir)
  source(file.path(.leanfe_pkg_dir, "common.R"))
  source(file.path(.leanfe_pkg_dir, "compress.R"))
  source(file.path(.leanfe_pkg_dir, "duckdb.R"))
  source(file.path(.leanfe_pkg_dir, "polars.R"))
  source(file.path(.leanfe_pkg_dir, "leanfe.R"))
} else {
  stop(paste(
    "Could not find leanfe R package directory.",
    "Working dir:", getwd(),
    "Project root:", .leanfe_project_root
  ))
}

# Clean up temporary variables
rm(list = ls(pattern = "^\\.leanfe_"))
