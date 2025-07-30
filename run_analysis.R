#!/usr/bin/env Rscript

# =============================================================================
# Master Script for R Concept Mapping Analysis
# =============================================================================
#
# This script orchestrates the complete concept mapping analysis workflow
# including data transformation, analysis, visualization, and reporting.
#
# Author: Concept Mapping Analysis Team
# Date: 2025
# License: Educational and Research Use
# =============================================================================

# Load required packages
required_packages <- c(
  "dplyr", "readr", "ggplot2", "ggrepel", "cluster", "factoextra", 
  "MASS", "corrplot", "viridis", "RColorBrewer", "gridExtra", 
  "tidyr", "stringr", "purrr", "tibble"
)

# Check and install missing packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, dependencies = TRUE)
  }
}

# Load packages
for (pkg in required_packages) {
  library(pkg, character.only = TRUE)
}

# =============================================================================
# Main Analysis Workflow
# =============================================================================

cat("=" * 60, "\n")
cat("CONCEPT MAPPING ANALYSIS - R IMPLEMENTATION\n")
cat("=" * 60, "\n")

# Step 1: Data Transformation
cat("\n📊 Step 1: Data Transformation\n")
cat("=" * 40, "\n")

if (file.exists("transform_data_to_rcmap.R")) {
  source("transform_data_to_rcmap.R")
  cat("✅ Data transformation completed\n")
} else {
  cat("⚠️  Data transformation script not found\n")
  cat("   Make sure transform_data_to_rcmap.R exists\n")
}

# Step 2: Main Analysis
cat("\n🔬 Step 2: Concept Mapping Analysis\n")
cat("=" * 40, "\n")

if (file.exists("concept_mapping_analysis.R")) {
  source("concept_mapping_analysis.R")
  cat("✅ Main analysis completed\n")
} else {
  cat("⚠️  Main analysis script not found\n")
  cat("   Make sure concept_mapping_analysis.R exists\n")
}

# Step 3: Custom Visualizations
cat("\n📈 Step 3: Custom Visualizations\n")
cat("=" * 40, "\n")

if (file.exists("create_custom_graphs.R")) {
  source("create_custom_graphs.R")
  cat("✅ Custom visualizations completed\n")
} else {
  cat("⚠️  Custom visualizations script not found\n")
  cat("   Make sure create_custom_graphs.R exists\n")
}

# Step 4: HTML Report Generation
cat("\n📋 Step 4: HTML Report Generation\n")
cat("=" * 40, "\n")

if (file.exists("create_html_report.R")) {
  source("create_html_report.R")
  cat("✅ HTML report generated\n")
} else {
  cat("⚠️  HTML report script not found\n")
  cat("   Make sure create_html_report.R exists\n")
}

# =============================================================================
# Final Summary
# =============================================================================

cat("\n" + "=" * 60, "\n")
cat("ANALYSIS WORKFLOW COMPLETED!\n")
cat("=" * 60, "\n")
cat("📊 Check the 'Figures' directory for all results\n")
cat("📋 Check the 'data' directory for transformed data\n")
cat("🌐 HTML report should be available if create_html_report.R exists\n")
cat("\n🎉 Concept mapping analysis is ready for interpretation!\n") 