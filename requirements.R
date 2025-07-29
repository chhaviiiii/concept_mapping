# =============================================================================
# R Package Requirements for Concept Mapping Analysis
# =============================================================================
#
# This file contains all required R packages for the concept mapping analysis
# toolkit. Each package includes version requirements and brief descriptions.
#
# Installation Instructions:
# 1. Run the installation commands below
# 2. Or use: source("requirements.R")
# 3. For CRAN packages: install.packages(c("package1", "package2"))
# 4. For Bioconductor: BiocManager::install(c("package1", "package2"))
#
# Author: BCCS AI Workshop Team
# Date: July 27, 2025
# =============================================================================

# =============================================================================
# Core Data Manipulation Packages
# =============================================================================

# dplyr: Grammar of data manipulation
# - Provides intuitive functions for data transformation
# - Essential for data cleaning and preparation
# - Version: >= 1.0.0
if (!require(dplyr, quietly = TRUE)) {
  install.packages("dplyr", dependencies = TRUE)
}

# readr: Fast and friendly data reading
# - Efficient CSV/TSV file reading
# - Better performance than base R read.csv
# - Version: >= 2.0.0
if (!require(readr, quietly = TRUE)) {
  install.packages("readr", dependencies = TRUE)
}

# tidyr: Data tidying
# - Reshape data between wide and long formats
# - Essential for data transformation
# - Version: >= 1.0.0
if (!require(tidyr, quietly = TRUE)) {
  install.packages("tidyr", dependencies = TRUE)
}

# tibble: Modern data frames
# - Enhanced data frame functionality
# - Better printing and subsetting
# - Version: >= 3.0.0
if (!require(tibble, quietly = TRUE)) {
  install.packages("tibble", dependencies = TRUE)
}

# =============================================================================
# Visualization Packages
# =============================================================================

# ggplot2: Grammar of graphics
# - Publication-quality plotting system
# - Consistent and customizable themes
# - Version: >= 3.3.0
if (!require(ggplot2, quietly = TRUE)) {
  install.packages("ggplot2", dependencies = TRUE)
}

# ggrepel: Label positioning
# - Prevents label overlap in plots
# - Essential for concept maps
# - Version: >= 0.9.0
if (!require(ggrepel, quietly = TRUE)) {
  install.packages("ggrepel", dependencies = TRUE)
}

# gridExtra: Arrange multiple plots
# - Combine multiple ggplot objects
# - Create complex plot layouts
# - Version: >= 2.3
if (!require(gridExtra, quietly = TRUE)) {
  install.packages("gridExtra", dependencies = TRUE)
}

# viridis: Color palettes
# - Colorblind-friendly palettes
# - Consistent color schemes
# - Version: >= 0.6.0
if (!require(viridis, quietly = TRUE)) {
  install.packages("viridis", dependencies = TRUE)
}

# RColorBrewer: Color schemes
# - Additional color palettes
# - Professional color schemes
# - Version: >= 1.1.0
if (!require(RColorBrewer, quietly = TRUE)) {
  install.packages("RColorBrewer", dependencies = TRUE)
}

# =============================================================================
# Statistical Analysis Packages
# =============================================================================

# MASS: Multidimensional scaling
# - Classical MDS implementation
# - Essential for concept mapping
# - Version: >= 7.3.0
if (!require(MASS, quietly = TRUE)) {
  install.packages("MASS", dependencies = TRUE)
}

# cluster: Clustering algorithms
# - K-means and hierarchical clustering
# - Silhouette analysis
# - Version: >= 2.1.0
if (!require(cluster, quietly = TRUE)) {
  install.packages("cluster", dependencies = TRUE)
}

# factoextra: Extract and visualize clustering results
# - Enhanced clustering visualizations
# - Optimal cluster number determination
# - Version: >= 1.0.7
if (!require(factoextra, quietly = TRUE)) {
  install.packages("factoextra", dependencies = TRUE)
}

# corrplot: Correlation matrix visualization
# - Heatmap creation for similarity matrices
# - Professional correlation plots
# - Version: >= 0.92
if (!require(corrplot, quietly = TRUE)) {
  install.packages("corrplot", dependencies = TRUE)
}

# =============================================================================
# Utility Packages
# =============================================================================

# stringr: String manipulation
# - Consistent string processing functions
# - Pattern matching and replacement
# - Version: >= 1.4.0
if (!require(stringr, quietly = TRUE)) {
  install.packages("stringr", dependencies = TRUE)
}

# purrr: Functional programming
# - Map, reduce, and other functional tools
# - Consistent iteration patterns
# - Version: >= 0.3.0
if (!require(purrr, quietly = TRUE)) {
  install.packages("purrr", dependencies = TRUE)
}

# knitr: Report generation
# - Dynamic report creation
# - HTML and PDF output
# - Version: >= 1.40
if (!require(knitr, quietly = TRUE)) {
  install.packages("knitr", dependencies = TRUE)
}

# =============================================================================
# Optional Packages (for advanced features)
# =============================================================================

# data.table: Fast data manipulation
# - High-performance data operations
# - Useful for large datasets
# - Version: >= 1.14.0
if (!require(data.table, quietly = TRUE)) {
  install.packages("data.table", dependencies = TRUE)
}

# parallel: Parallel processing
# - Multi-core computation support
# - Performance optimization
# - Version: Built into R
# No installation needed

# doParallel: Parallel backend
# - Parallel processing with foreach
# - Performance optimization
# - Version: >= 1.0.0
if (!require(doParallel, quietly = TRUE)) {
  install.packages("doParallel", dependencies = TRUE)
}

# =============================================================================
# Installation Function
# =============================================================================

install_required_packages <- function() {
  """
  Install all required packages for concept mapping analysis.
  
  This function checks for and installs all necessary packages
  with appropriate version requirements.
  """
  
  cat("Installing required packages for concept mapping analysis...\n")
  
  # Core packages
  core_packages <- c(
    "dplyr", "readr", "tidyr", "tibble",
    "ggplot2", "ggrepel", "gridExtra", "viridis", "RColorBrewer",
    "MASS", "cluster", "factoextra", "corrplot",
    "stringr", "purrr", "knitr"
  )
  
  # Optional packages
  optional_packages <- c(
    "data.table", "doParallel"
  )
  
  # Install core packages
  cat("Installing core packages...\n")
  for (pkg in core_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      install.packages(pkg, dependencies = TRUE)
    } else {
      cat("✓", pkg, "already installed\n")
    }
  }
  
  # Install optional packages
  cat("\nInstalling optional packages...\n")
  for (pkg in optional_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      install.packages(pkg, dependencies = TRUE)
    } else {
      cat("✓", pkg, "already installed\n")
    }
  }
  
  cat("\n✅ Package installation completed!\n")
  cat("You can now run the concept mapping analysis.\n")
}

# =============================================================================
# Version Check Function
# =============================================================================

check_package_versions <- function() {
  """
  Check installed package versions against requirements.
  
  This function verifies that all required packages are installed
  with appropriate versions.
  """
  
  cat("Checking package versions...\n")
  
  # Define required versions
  required_versions <- list(
    dplyr = "1.0.0",
    readr = "2.0.0",
    tidyr = "1.0.0",
    tibble = "3.0.0",
    ggplot2 = "3.3.0",
    ggrepel = "0.9.0",
    gridExtra = "2.3",
    viridis = "0.6.0",
    RColorBrewer = "1.1.0",
    MASS = "7.3.0",
    cluster = "2.1.0",
    factoextra = "1.0.7",
    corrplot = "0.92",
    stringr = "1.4.0",
    purrr = "0.3.0",
    knitr = "1.40"
  )
  
  # Check each package
  for (pkg in names(required_versions)) {
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
      current_version <- packageVersion(pkg)
      required_version <- required_versions[[pkg]]
      
      if (current_version >= required_version) {
        cat("✓", pkg, "version", current_version, ">= required", required_version, "\n")
      } else {
        cat("⚠", pkg, "version", current_version, "< required", required_version, "\n")
        cat("  Consider updating with: install.packages('", pkg, "', dependencies = TRUE)\n", sep = "")
      }
    } else {
      cat("✗", pkg, "not installed\n")
    }
  }
}

# =============================================================================
# Load All Packages Function
# =============================================================================

load_all_packages <- function() {
  """
  Load all required packages for concept mapping analysis.
  
  This function loads all necessary packages and provides
  status information for each.
  """
  
  cat("Loading required packages...\n")
  
  # Core packages to load
  packages_to_load <- c(
    "dplyr", "readr", "ggplot2", "ggrepel", "cluster", "factoextra", 
    "MASS", "corrplot", "viridis", "RColorBrewer", "gridExtra", 
    "tidyr", "stringr", "purrr", "tibble"
  )
  
  # Load each package
  for (pkg in packages_to_load) {
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("✓ Loaded", pkg, "\n")
    } else {
      cat("✗ Failed to load", pkg, "\n")
      cat("  Install with: install.packages('", pkg, "')\n", sep = "")
    }
  }
  
  cat("\n✅ Package loading completed!\n")
}

# =============================================================================
# Main Execution
# =============================================================================

# Uncomment the line below to automatically install packages when sourcing this file
# install_required_packages()

# Uncomment the line below to check package versions
# check_package_versions()

# Uncomment the line below to load all packages
# load_all_packages()

cat("R requirements file loaded.\n")
cat("To install packages, run: install_required_packages()\n")
cat("To check versions, run: check_package_versions()\n")
cat("To load packages, run: load_all_packages()\n") 