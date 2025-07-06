#!/usr/bin/env Rscript

# Simple Qualtrics Data Cleaning Script
# This script handles complex Qualtrics CSV files with multiple headers and empty columns

library(dplyr)
library(readr)
library(stringr)

# Configuration
input_file <- "data/BCCS AI Workshop_July 6, 2025_15.44.csv"
output_dir <- "data/simple_cleaned"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Starting simple Qualtrics data cleaning...\n")
cat("Input file:", input_file, "\n")
cat("Output directory:", output_dir, "\n\n")

# First, let's examine the file structure more carefully
cat("Examining file structure...\n")
raw_lines <- readLines(input_file, n = 50)

# Find where the actual data starts
data_start_line <- 1
for (i in 1:length(raw_lines)) {
  line <- raw_lines[i]
  # Look for a line that has many commas (indicating data columns)
  comma_count <- str_count(line, ",")
  if (comma_count > 100) {  # Assuming data rows have many columns
    data_start_line <- i
    break
  }
}

cat("Data appears to start at line:", data_start_line, "\n")

# Read the file with a more flexible approach
cat("Reading data with flexible parsing...\n")

# Try to read the file with different approaches
tryCatch({
  # First attempt: read with default settings
  data <- read_csv(input_file, 
                   skip = data_start_line - 1,
                   show_col_types = FALSE,
                   progress = TRUE)
}, error = function(e) {
  cat("First attempt failed, trying alternative approach...\n")
  # Second attempt: read with more flexible parsing
  data <- read_csv(input_file, 
                   skip = data_start_line - 1,
                   show_col_types = FALSE,
                   progress = TRUE,
                   guess_max = 0)  # Don't guess column types
})

cat("Dataset loaded successfully!\n")
cat("Dimensions:", nrow(data), "rows x", ncol(data), "columns\n\n")

# Get column names
col_names <- names(data)
total_cols <- length(col_names)

cat("Total columns found:", total_cols, "\n")

# Function to check if a column has meaningful data
has_meaningful_data <- function(x) {
  # Remove NA, empty strings, and common Qualtrics placeholders
  clean_values <- x[!is.na(x) & x != "" & x != "NA" & x != "N/A"]
  
  # Check if we have any non-empty values
  if (length(clean_values) == 0) return(FALSE)
  
  # Check if the values look like actual data (not just headers or instructions)
  # Look for patterns that suggest actual responses
  has_responses <- any(
    str_detect(clean_values, "^[1-5]$") |  # Single digits 1-5
    str_detect(clean_values, "^[A-Za-z]") |  # Text responses
    str_detect(clean_values, "^\\d+$")  # Any numeric responses
  )
  
  return(has_responses)
}

# Identify meaningful columns
cat("Identifying meaningful columns...\n")
meaningful_columns <- character()
empty_columns <- character()

# Sample the data to check columns (faster than checking all rows)
sample_size <- min(100, nrow(data))
sample_data <- data %>% slice_sample(n = sample_size)

for (col in col_names) {
  if (has_meaningful_data(sample_data[[col]])) {
    meaningful_columns <- c(meaningful_columns, col)
  } else {
    empty_columns <- c(empty_columns, col)
  }
}

cat("Meaningful columns found:", length(meaningful_columns), "\n")
cat("Empty/irrelevant columns found:", length(empty_columns), "\n\n")

# Create cleaned dataset with only meaningful columns
if (length(meaningful_columns) > 0) {
  cleaned_data <- data %>% select(all_of(meaningful_columns))
  
  cat("Cleaned dataset created!\n")
  cat("Dimensions:", nrow(cleaned_data), "rows x", ncol(cleaned_data), "columns\n\n")
  
  # Save the cleaned dataset
  output_file <- file.path(output_dir, "cleaned_qualtrics_data.csv")
  write_csv(cleaned_data, output_file)
  cat("Cleaned dataset saved to:", output_file, "\n\n")
  
  # Analyze column patterns
  group_cols <- meaningful_columns[str_detect(meaningful_columns, "_GROUP$")]
  rank_cols <- meaningful_columns[str_detect(meaningful_columns, "_RANK$")]
  rating_cols <- meaningful_columns[str_detect(meaningful_columns, "_RATING$")]
  
  cat("Column pattern analysis:\n")
  cat("  GROUP columns:", length(group_cols), "\n")
  cat("  RANK columns:", length(rank_cols), "\n")
  cat("  RATING columns:", length(rating_cols), "\n")
  cat("  Other columns:", length(meaningful_columns) - length(group_cols) - length(rank_cols) - length(rating_cols), "\n\n")
  
  # Show sample of the data
  cat("Sample of cleaned data (first 3 rows, first 10 columns):\n")
  sample_display <- cleaned_data %>% 
    select(1:min(10, ncol(cleaned_data))) %>% 
    slice_head(n = 3)
  print(sample_display)
  cat("\n")
  
  # Create a summary report
  summary_file <- file.path(output_dir, "data_summary.txt")
  sink(summary_file)
  cat("Simple Qualtrics Data Cleaning Summary\n")
  cat("=====================================\n\n")
  cat("Original file:", input_file, "\n")
  cat("Original dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
  cat("Cleaned dimensions:", nrow(cleaned_data), "rows x", ncol(cleaned_data), "columns\n\n")
  cat("Column analysis:\n")
  cat("  Meaningful columns:", length(meaningful_columns), "\n")
  cat("  Empty/irrelevant columns removed:", length(empty_columns), "\n\n")
  cat("Column pattern breakdown:\n")
  cat("  GROUP columns:", length(group_cols), "\n")
  cat("  RANK columns:", length(rank_cols), "\n")
  cat("  RATING columns:", length(rating_cols), "\n")
  cat("  Other columns:", length(meaningful_columns) - length(group_cols) - length(rank_cols) - length(rating_cols), "\n\n")
  cat("Sample meaningful columns:\n")
  for (col in head(meaningful_columns, 20)) {
    cat("  ", col, "\n")
  }
  if (length(meaningful_columns) > 20) {
    cat("  ... and", length(meaningful_columns) - 20, "more\n")
  }
  sink()
  
  cat("Summary report saved to:", summary_file, "\n\n")
  
  # Create a sample of the cleaned data for inspection
  sample_file <- file.path(output_dir, "sample_cleaned_data.csv")
  sample_data <- cleaned_data %>% slice_head(n = 10)
  write_csv(sample_data, sample_file)
  cat("Sample data (first 10 rows) saved to:", sample_file, "\n\n")
  
} else {
  cat("No meaningful columns found! The data structure may be very different than expected.\n")
  
  # Let's try a different approach - look at the raw data
  cat("Examining raw data structure...\n")
  raw_sample <- readLines(input_file, n = 20)
  for (i in 1:length(raw_sample)) {
    cat("Line", i, ":", substr(raw_sample[i], 1, 100), "...\n")
  }
}

cat("Data cleaning completed!\n") 