#!/usr/bin/env Rscript

# Qualtrics Data Cleaning and Organization Script v2
# This script properly handles Qualtrics CSV files with multiple header rows

library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(purrr)

# Configuration
input_file <- "data/BCCS AI Workshop_July 6, 2025_15.44.csv"
output_dir <- "data/cleaned_v2"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Starting Qualtrics data cleaning process (v2)...\n")
cat("Input file:", input_file, "\n")
cat("Output directory:", output_dir, "\n\n")

# Function to check if a column is empty or mostly empty
is_empty_column <- function(x, threshold = 0.95) {
  if (is.numeric(x)) {
    return(sum(is.na(x)) / length(x) >= threshold)
  } else {
    return(sum(is.na(x) | x == "" | x == "NA") / length(x) >= threshold)
  }
}

# Function to identify column types
identify_column_types <- function(col_names) {
  metadata_patterns <- c(
    "^StartDate$", "^EndDate$", "^Status$", "^IPAddress$", "^Progress$",
    "^Duration", "^Finished$", "^RecordedDate$", "^ResponseId$",
    "^Recipient", "^ExternalReference$", "^Location", "^Distribution",
    "^UserLanguage$"
  )
  
  group_patterns <- c("_GROUP$")
  rank_patterns <- c("_RANK$")
  rating_patterns <- c("_RATING$")
  
  column_types <- list(
    metadata = character(),
    group = character(),
    rank = character(),
    rating = character(),
    other = character()
  )
  
  for (col in col_names) {
    if (any(str_detect(col, metadata_patterns))) {
      column_types$metadata <- c(column_types$metadata, col)
    } else if (str_detect(col, group_patterns)) {
      column_types$group <- c(column_types$group, col)
    } else if (str_detect(col, rank_patterns)) {
      column_types$rank <- c(column_types$rank, col)
    } else if (str_detect(col, rating_patterns)) {
      column_types$rating <- c(column_types$rating, col)
    } else {
      column_types$other <- c(column_types$other, col)
    }
  }
  
  return(column_types)
}

# First, let's examine the file structure
cat("Examining file structure...\n")
raw_lines <- readLines(input_file, n = 20)
cat("First 10 lines of the file:\n")
for (i in 1:min(10, length(raw_lines))) {
  cat("Line", i, ":", substr(raw_lines[i], 1, 100), "...\n")
}
cat("\n")

# Count total lines
total_lines <- length(readLines(input_file))
cat("Total lines in file:", total_lines, "\n")

# Read the data properly, skipping the first 3 rows (Qualtrics headers)
cat("Reading data with proper header handling...\n")
data <- read_csv(input_file, 
                 skip = 3,  # Skip the first 3 rows (Qualtrics headers)
                 show_col_types = FALSE,
                 progress = TRUE)

cat("Dataset loaded successfully!\n")
cat("Dimensions:", nrow(data), "rows x", ncol(data), "columns\n\n")

# Get column names
col_names <- names(data)
total_cols <- length(col_names)

cat("Total columns found:", total_cols, "\n")

# Identify column types
column_types <- identify_column_types(col_names)

cat("Column type breakdown:\n")
cat("  Metadata columns:", length(column_types$metadata), "\n")
cat("  Group columns:", length(column_types$group), "\n")
cat("  Rank columns:", length(column_types$rank), "\n")
cat("  Rating columns:", length(column_types$rating), "\n")
cat("  Other columns:", length(column_types$other), "\n\n")

# Check for empty columns
cat("Analyzing columns for empty data...\n")
empty_columns <- character()
non_empty_columns <- character()

# Sample the data to check for empty columns (faster than checking all rows)
sample_size <- min(1000, nrow(data))
sample_data <- data %>% slice_sample(n = sample_size)

for (col in col_names) {
  if (is_empty_column(sample_data[[col]])) {
    empty_columns <- c(empty_columns, col)
  } else {
    non_empty_columns <- c(non_empty_columns, col)
  }
}

cat("Empty columns found:", length(empty_columns), "\n")
cat("Non-empty columns found:", length(non_empty_columns), "\n\n")

# Filter to only relevant columns (exclude empty and metadata)
relevant_columns <- non_empty_columns[!non_empty_columns %in% column_types$metadata]

cat("Relevant data columns:", length(relevant_columns), "\n\n")

# Create cleaned dataset with only relevant columns
cleaned_data <- data %>% select(all_of(relevant_columns))

cat("Cleaned dataset created!\n")
cat("Dimensions:", nrow(cleaned_data), "rows x", ncol(cleaned_data), "columns\n\n")

# Save the cleaned dataset
output_file <- file.path(output_dir, "cleaned_qualtrics_data.csv")
write_csv(cleaned_data, output_file)
cat("Cleaned dataset saved to:", output_file, "\n\n")

# Analyze the structure of GROUP and RANK columns
group_cols <- relevant_columns[str_detect(relevant_columns, "_GROUP$")]
rank_cols <- relevant_columns[str_detect(relevant_columns, "_RANK$")]

cat("Analysis of data structure:\n")
cat("  Group columns:", length(group_cols), "\n")
cat("  Rank columns:", length(rank_cols), "\n\n")

# Extract statement numbers from column names
extract_statement_numbers <- function(col_names) {
  # Extract numbers from patterns like Q1_0_GROUP, Q1_0_1_RANK
  numbers <- str_extract_all(col_names, "\\d+")
  statement_nums <- map(numbers, function(x) {
    if (length(x) >= 2) {
      return(as.numeric(x[2]))  # Second number is usually the statement number
    } else {
      return(NA)
    }
  })
  return(unlist(statement_nums))
}

if (length(group_cols) > 0) {
  group_statement_nums <- extract_statement_numbers(group_cols)
  unique_group_statements <- unique(na.omit(group_statement_nums))
  cat("Unique statements in GROUP columns:", length(unique_group_statements), "\n")
  cat("Statement numbers:", paste(head(unique_group_statements, 10), collapse = ", "), "...\n\n")
}

if (length(rank_cols) > 0) {
  rank_statement_nums <- extract_statement_numbers(rank_cols)
  unique_rank_statements <- unique(na.omit(rank_statement_nums))
  cat("Unique statements in RANK columns:", length(unique_rank_statements), "\n")
  cat("Statement numbers:", paste(head(unique_rank_statements, 10), collapse = ", "), "...\n\n")
}

# Show some sample data
cat("Sample of cleaned data (first 3 rows, first 10 columns):\n")
sample_display <- cleaned_data %>% 
  select(1:min(10, ncol(cleaned_data))) %>% 
  slice_head(n = 3)
print(sample_display)
cat("\n")

# Create a summary report
summary_file <- file.path(output_dir, "data_summary.txt")
sink(summary_file)
cat("Qualtrics Data Cleaning Summary (v2)\n")
cat("===================================\n\n")
cat("Original file:", input_file, "\n")
cat("Original dimensions:", total_lines, "total lines\n")
cat("Cleaned dimensions:", nrow(cleaned_data), "rows x", ncol(cleaned_data), "columns\n\n")
cat("Column type breakdown:\n")
cat("  Metadata columns:", length(column_types$metadata), "\n")
cat("  Group columns:", length(column_types$group), "\n")
cat("  Rank columns:", length(column_types$rank), "\n")
cat("  Rating columns:", length(column_types$rating), "\n")
cat("  Other columns:", length(column_types$other), "\n\n")
cat("Empty columns removed:", length(empty_columns), "\n")
cat("Relevant columns kept:", length(relevant_columns), "\n\n")
cat("Data structure analysis:\n")
if (length(group_cols) > 0) {
  cat("  Group columns found:", length(group_cols), "\n")
  cat("  Unique statements in groups:", length(unique_group_statements), "\n")
}
if (length(rank_cols) > 0) {
  cat("  Rank columns found:", length(rank_cols), "\n")
  cat("  Unique statements in ranks:", length(unique_rank_statements), "\n")
}
sink()

cat("Summary report saved to:", summary_file, "\n\n")

# Create a sample of the cleaned data for inspection
sample_file <- file.path(output_dir, "sample_cleaned_data.csv")
sample_data <- cleaned_data %>% slice_head(n = 10)
write_csv(sample_data, sample_file)
cat("Sample data (first 10 rows) saved to:", sample_file, "\n\n")

cat("Data cleaning completed successfully!\n")
cat("Next steps:\n")
cat("1. Review the sample data to understand the structure\n")
cat("2. Create transformation scripts to convert to RCMap format\n")
cat("3. Generate the required CSV files (Statements.csv, SortedCards.csv, Ratings.csv, Demographics.csv)\n") 