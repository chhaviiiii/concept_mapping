# R Package Requirements for Concept Mapping Analysis
# Run this script to install all required packages

# Core data manipulation packages
install.packages(c(
  "dplyr",      # Data manipulation
  "readr",      # Fast data reading
  "stringr",    # String manipulation
  "tidyr",      # Data reshaping
  "data.table", # Fast data operations
  "purrr",      # Functional programming
  "tibble"      # Modern data frames
))

# Visualization packages
install.packages(c(
  "ggplot2",    # Grammar of graphics
  "ggrepel",    # Non-overlapping labels
  "viridis",    # Color palettes
  "RColorBrewer", # Color palettes
  "gridExtra"   # Arrange multiple plots
))

# Statistical analysis packages
install.packages(c(
  "cluster",    # Clustering algorithms
  "factoextra", # Multivariate analysis
  "MASS",       # Multidimensional scaling
  "corrplot"    # Correlation visualization
))

# Report generation packages
install.packages(c(
  "knitr",      # Dynamic report generation
  "rmarkdown"   # R Markdown support
))

cat("All required packages have been installed!\n")
cat("You can now run the concept mapping analysis.\n") 