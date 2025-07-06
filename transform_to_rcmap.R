#!/usr/bin/env Rscript

# Transform Qualtrics Data to RCMap Format
# This script converts the cleaned Qualtrics data into the required RCMap CSV files

library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(purrr)

# Configuration
input_file <- "data/simple_cleaned/cleaned_qualtrics_data.csv"
output_dir <- "data/rcmap_format"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Starting transformation to RCMap format...\n")
cat("Input file:", input_file, "\n")
cat("Output directory:", output_dir, "\n\n")

# Read the cleaned data
cat("Reading cleaned data...\n")
data <- read_csv(input_file, show_col_types = FALSE)

# Remove the first two header rows and keep only actual data
cat("Removing header rows...\n")
data <- data %>% slice(3:n())  # Keep rows 3 onwards (actual data)

cat("Data dimensions after removing headers:", nrow(data), "rows x", ncol(data), "columns\n\n")

# Get column names
col_names <- names(data)

# Identify column types
group_cols <- col_names[str_detect(col_names, "_GROUP$")]
rank_cols <- col_names[str_detect(col_names, "_RANK$")]
metadata_cols <- col_names[!col_names %in% c(group_cols, rank_cols)]

cat("Column breakdown:\n")
cat("  Metadata columns:", length(metadata_cols), "\n")
cat("  Group columns:", length(group_cols), "\n")
cat("  Rank columns:", length(rank_cols), "\n\n")

# Extract statement numbers from column names
extract_statement_numbers <- function(col_names) {
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

# Get unique statement numbers
group_statement_nums <- extract_statement_numbers(group_cols)
rank_statement_nums <- extract_statement_numbers(rank_cols)

unique_group_statements <- unique(na.omit(group_statement_nums))
unique_rank_statements <- unique(na.omit(rank_statement_nums))

cat("Statement analysis:\n")
cat("  Unique statements in GROUP columns:", length(unique_group_statements), "\n")
cat("  Unique statements in RANK columns:", length(unique_rank_statements), "\n")
cat("  Statement numbers range:", min(unique_group_statements), "to", max(unique_group_statements), "\n\n")

# 1. Create Statements.csv
cat("Creating Statements.csv...\n")
statements_data <- data.frame(
  StatementID = unique_group_statements,
  StatementText = paste("Statement", unique_group_statements),  # Placeholder text
  stringsAsFactors = FALSE
)

# Sort by statement ID
statements_data <- statements_data %>% arrange(StatementID)

write_csv(statements_data, file.path(output_dir, "Statements.csv"))
cat("Statements.csv created with", nrow(statements_data), "statements\n\n")

# 2. Create SortedCards.csv (from GROUP columns)
cat("Creating SortedCards.csv...\n")
sorted_cards_data <- data.frame(
  ParticipantID = character(),
  StatementID = numeric(),
  PileID = numeric(),
  stringsAsFactors = FALSE
)

# Process each participant's grouping data
for (row_idx in 1:nrow(data)) {
  participant_id <- paste0("P", row_idx)
  
  # Get the grouping data for this participant
  participant_groups <- data[row_idx, group_cols, drop = FALSE]
  
  for (col in group_cols) {
    statement_num <- extract_statement_numbers(col)[1]
    group_value <- participant_groups[[col]]
    
    # Only include if there's a valid group assignment
    if (!is.na(group_value) && group_value != "" && group_value != "NA") {
      # Try to convert to numeric pile ID
      pile_id <- as.numeric(group_value)
      if (!is.na(pile_id)) {
        sorted_cards_data <- rbind(sorted_cards_data, data.frame(
          ParticipantID = participant_id,
          StatementID = statement_num,
          PileID = pile_id,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

write_csv(sorted_cards_data, file.path(output_dir, "SortedCards.csv"))
cat("SortedCards.csv created with", nrow(sorted_cards_data), "sorting assignments\n\n")

# 3. Create Ratings.csv (from RANK columns)
cat("Creating Ratings.csv...\n")
ratings_data <- data.frame(
  ParticipantID = character(),
  StatementID = numeric(),
  Rating = numeric(),
  stringsAsFactors = FALSE
)

# Process each participant's rating data
for (row_idx in 1:nrow(data)) {
  participant_id <- paste0("P", row_idx)
  
  # Get the rating data for this participant
  participant_ratings <- data[row_idx, rank_cols, drop = FALSE]
  
  for (col in rank_cols) {
    statement_num <- extract_statement_numbers(col)[1]
    rating_value <- participant_ratings[[col]]
    
    # Only include if there's a valid rating
    if (!is.na(rating_value) && rating_value != "" && rating_value != "NA") {
      # Try to convert to numeric rating
      rating <- as.numeric(rating_value)
      if (!is.na(rating)) {
        ratings_data <- rbind(ratings_data, data.frame(
          ParticipantID = participant_id,
          StatementID = statement_num,
          Rating = rating,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

write_csv(ratings_data, file.path(output_dir, "Ratings.csv"))
cat("Ratings.csv created with", nrow(ratings_data), "ratings\n\n")

# 4. Create Demographics.csv
cat("Creating Demographics.csv...\n")
demographics_data <- data.frame(
  ParticipantID = paste0("P", 1:nrow(data)),
  stringsAsFactors = FALSE
)

# Add relevant metadata columns as demographics
relevant_metadata <- c("StartDate", "EndDate", "Progress", "Duration (in seconds)", "Finished")
available_metadata <- intersect(relevant_metadata, metadata_cols)

for (col in available_metadata) {
  demographics_data[[col]] <- data[[col]]
}

write_csv(demographics_data, file.path(output_dir, "Demographics.csv"))
cat("Demographics.csv created with", nrow(demographics_data), "participants\n\n")

# Create a summary report
summary_file <- file.path(output_dir, "transformation_summary.txt")
sink(summary_file)
cat("Qualtrics to RCMap Transformation Summary\n")
cat("=========================================\n\n")
cat("Input file:", input_file, "\n")
cat("Input dimensions:", nrow(data), "participants x", ncol(data), "columns\n\n")
cat("Output files created:\n")
cat("  Statements.csv:", nrow(statements_data), "statements\n")
cat("  SortedCards.csv:", nrow(sorted_cards_data), "sorting assignments\n")
cat("  Ratings.csv:", nrow(ratings_data), "ratings\n")
cat("  Demographics.csv:", nrow(demographics_data), "participants\n\n")
cat("Data structure:\n")
cat("  Unique statements:", length(unique_group_statements), "\n")
cat("  Statement range:", min(unique_group_statements), "to", max(unique_group_statements), "\n")
cat("  Participants:", nrow(data), "\n\n")
cat("Column mapping:\n")
cat("  GROUP columns -> SortedCards.csv\n")
cat("  RANK columns -> Ratings.csv\n")
cat("  Metadata columns -> Demographics.csv\n")
cat("  Statement IDs generated from column names\n\n")
cat("Sample data:\n")
cat("First 5 statements:", paste(head(statements_data$StatementID, 5), collapse = ", "), "\n")
cat("First 5 sorting assignments:\n")
print(head(sorted_cards_data, 5))
cat("\nFirst 5 ratings:\n")
print(head(ratings_data, 5))
sink()

cat("Transformation summary saved to:", summary_file, "\n\n")

# Show sample of each output file
cat("Sample of generated files:\n\n")

cat("Statements.csv (first 10 rows):\n")
print(head(statements_data, 10))
cat("\n")

cat("SortedCards.csv (first 10 rows):\n")
print(head(sorted_cards_data, 10))
cat("\n")

cat("Ratings.csv (first 10 rows):\n")
print(head(ratings_data, 10))
cat("\n")

cat("Demographics.csv (first 5 rows):\n")
print(head(demographics_data, 5))
cat("\n")

cat("Transformation completed successfully!\n")
cat("Files created in:", output_dir, "\n")
cat("You can now use these files with the RCMap system.\n") 