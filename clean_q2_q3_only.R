#!/usr/bin/env Rscript

# Clean Qualtrics Data: Keep only Q2 and Q3 columns, remove all Q1 columns

library(dplyr)
library(readr)
library(stringr)

# Configuration
input_file <- "data/BCCS AI Workshop_July 6, 2025_15.44.csv"
output_file <- "data/cleaned_q2_q3_only.csv"

cat("Reading Qualtrics data...\n")
data <- read_csv(input_file, show_col_types = FALSE)

cat("Original columns:", ncol(data), "\n")

# Identify Q1, Q2, Q3 columns
data_cols <- names(data)
q1_cols <- grep("^Q1_", data_cols, value = TRUE)
q2_cols <- grep("^Q2_", data_cols, value = TRUE)
q3_cols <- grep("^Q3_", data_cols, value = TRUE)

# Identify metadata columns (not Q1, Q2, Q3)
meta_cols <- setdiff(data_cols, c(q1_cols, q2_cols, q3_cols))

# Keep only Q2, Q3, and metadata columns
keep_cols <- c(meta_cols, q2_cols, q3_cols)
cleaned_data <- data[, keep_cols]

cat("Columns after removing Q1:", ncol(cleaned_data), "\n")

write_csv(cleaned_data, output_file)
cat("Cleaned data saved to:", output_file, "\n") 