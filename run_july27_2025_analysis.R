#!/usr/bin/env Rscript

# Master Script for BCCS AI Workshop July 27, 2025 Concept Mapping Analysis
# This script runs the complete pipeline:
# 1. Transform raw Qualtrics data to RCMap format
# 2. Perform simplified concept mapping analysis (rating-based)
# 3. Generate visualizations and reports

cat("=== BCCS AI Workshop July 27, 2025 Concept Mapping Analysis ===\n")
cat("Starting complete analysis pipeline...\n\n")

# Step 1: Transform data
cat("Step 1: Transforming raw data to RCMap format...\n")
source("transform_july27_2025_to_rcmap.R")

# Step 2: Perform simplified concept mapping analysis
cat("\nStep 2: Performing simplified concept mapping analysis...\n")
source("simplified_concept_mapping_analysis.R")

cat("\n=== Analysis Pipeline Complete ===\n")
cat("All results have been generated and saved.\n")
cat("Check the following directories for outputs:\n")
cat("- data/rcmap_july27_2025/ (transformed data files)\n")
cat("- Figures/july27_2025_analysis/ (visualizations and reports)\n")
cat("- analysis_report.html (comprehensive HTML report)\n") 