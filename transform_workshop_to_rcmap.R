#!/usr/bin/env Rscript

# Transform Workshop Qualtrics Data to RCMap Format (Q1 = grouping, Q2 = ratings)

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(data.table)

# Configuration
input_file <- "data/BCCS AI Workshop_July 8, 2025_00.34.csv"
output_dir <- "data/rcmap_workshop"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Reading Workshop CSV, using first line for variable names, second for statement text, and data from line 3...\n")
# Read variable names and statement text using proper CSV parsing
header_data <- read.csv(input_file, nrows = 2, header = FALSE, stringsAsFactors = FALSE)
var_names <- as.character(header_data[1, ])
question_row <- as.character(header_data[2, ])

# Read data, skipping first two lines
data <- fread(input_file, skip = 2, header = FALSE, data.table = FALSE)
names(data) <- trimws(var_names)

# Print first 20 column names for debugging
cat("First 20 column names after setting manually:\n")
print(names(data)[1:20])

# Identify columns (robust)
col_names <- names(data)
q1_cols <- col_names[grepl("^Q1_", col_names)]
q2_cols <- col_names[grepl("^Q2\\.", col_names)]
meta_cols <- setdiff(col_names, c(q1_cols, q2_cols))

# --- Statements.csv ---
cat("Extracting statement text from line 2...\n")
statements <- sapply(q1_cols, function(col) {
  idx <- which(col_names == col)
  txt <- question_row[idx]
  # Extract statement text after the dash and number
  txt <- sub(".*- ", "", txt)
  # Remove trailing ":Right" or similar
  txt <- sub(":.*$", "", txt)
  trimws(txt)
})

# Print first few statements for debugging
cat("First 5 statements extracted:\n")
print(statements[1:5])

statements_df <- data.frame(
  StatementID = seq_along(statements),
  StatementText = statements,
  stringsAsFactors = FALSE
)
write_csv(statements_df, file.path(output_dir, "Statements.csv"))

# --- SortedCards.csv (Q1) ---
cat("Creating SortedCards.csv from Q1 columns...\n")
if (length(q1_cols) > 0) {
  sorted_cards <- data %>%
    mutate(ParticipantID = row_number()) %>%
    select(ParticipantID, all_of(q1_cols)) %>%
    pivot_longer(
      cols = all_of(q1_cols),
      names_to = "QCol",
      values_to = "PileID"
    ) %>%
    mutate(
      StatementID = as.integer(str_extract(QCol, "\\d+")),
      PileID = as.integer(PileID)
    ) %>%
    select(ParticipantID, StatementID, PileID) %>%
    filter(!is.na(PileID))
  write_csv(sorted_cards, file.path(output_dir, "SortedCards.csv"))
} else {
  cat("No Q1 columns found! Skipping SortedCards.csv\n")
}

# --- Ratings.csv (Q2) ---
cat("Creating Ratings.csv from Q2 columns...\n")
if (length(q2_cols) > 0) {
  ratings <- data %>%
    mutate(ParticipantID = row_number()) %>%
    select(ParticipantID, all_of(q2_cols)) %>%
    pivot_longer(
      cols = all_of(q2_cols),
      names_to = "QCol",
      values_to = "Rating"
    ) %>%
    mutate(
      StatementID = as.integer(str_extract(QCol, "\\d+$")),
      Rating = as.numeric(Rating)
    ) %>%
    select(ParticipantID, StatementID, Rating) %>%
    filter(!is.na(Rating))
  write_csv(ratings, file.path(output_dir, "Ratings.csv"))
} else {
  cat("No Q2 columns found! Skipping Ratings.csv\n")
}

# --- Demographics.csv ---
cat("Creating Demographics.csv from metadata columns...\n")
demographics <- data %>%
  mutate(ParticipantID = row_number()) %>%
  select(ParticipantID, all_of(meta_cols))
write_csv(demographics, file.path(output_dir, "Demographics.csv"))

cat("All RCMap input files created in:", output_dir, "\n")
cat("You can now run RCMap as usual on this data.\n") 