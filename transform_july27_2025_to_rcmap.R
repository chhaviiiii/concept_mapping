#!/usr/bin/env Rscript

# Transform July 27, 2025 BCCS AI Workshop Data to RCMap Format
# This script processes the Qualtrics survey data and extracts:
# - 100 statements about AI in cancer care
# - Grouping data (Q1 columns) - how participants grouped statements
# - Rating data (Q2 columns) - importance and feasibility ratings

library(dplyr)
library(readr)
library(stringr)
library(tidyr)
library(data.table)
library(purrr)

# Configuration - only use files that actually exist
input_files <- c(
  "data/BCCS AI Workshop_July 27, 2025_15.23.csv",
  "data/BCCS AI Workshop_July 27, 2025_15.26_utf8.tsv"
)
output_dir <- "data/rcmap_july27_2025"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Function to extract statements from question text
extract_statement_text <- function(question_text) {
  # Extract text after the dash and number
  txt <- sub(".*- ", "", question_text)
  # Remove trailing ":Right" or similar
  txt <- sub(":.*$", "", txt)
  # Clean up any remaining artifacts
  txt <- trimws(txt)
  return(txt)
}

# Function to process CSV file
process_csv_file <- function(file_path) {
  cat("Processing file:", file_path, "\n")
  
  # Read variable names and statement text using proper CSV parsing
  header_data <- read.csv(file_path, nrows = 2, header = FALSE, stringsAsFactors = FALSE)
  var_names <- as.character(header_data[1, ])
  question_row <- as.character(header_data[2, ])
  
  # Read data, skipping first two lines
  data <- fread(file_path, skip = 2, header = FALSE, data.table = FALSE)
  names(data) <- trimws(var_names)
  
  # Identify columns
  col_names <- names(data)
  q1_cols <- col_names[grepl("^Q1_", col_names)]
  q2_1_cols <- col_names[grepl("^Q2\\.1_", col_names)]  # Importance ratings
  q2_2_cols <- col_names[grepl("^Q2\\.2_", col_names)]  # Feasibility ratings
  meta_cols <- setdiff(col_names, c(q1_cols, q2_1_cols, q2_2_cols))
  
  # Extract statements from Q1 columns
  statements <- sapply(q1_cols, function(col) {
    idx <- which(col_names == col)
    txt <- question_row[idx]
    extract_statement_text(txt)
  })
  
  # Create statements dataframe
  statements_df <- data.frame(
    StatementID = seq_along(statements),
    StatementText = statements,
    stringsAsFactors = FALSE
  )
  
  # Process grouping data (Q1)
  sorted_cards <- NULL
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
        PileID = as.character(PileID)  # Keep as character to handle "Group X" format
      ) %>%
      select(ParticipantID, StatementID, PileID) %>%
      filter(!is.na(PileID) & PileID != "")
  }
  
  # Process importance ratings (Q2.1)
  importance_ratings <- NULL
  if (length(q2_1_cols) > 0) {
    importance_ratings <- data %>%
      mutate(ParticipantID = row_number()) %>%
      select(ParticipantID, all_of(q2_1_cols)) %>%
      pivot_longer(
        cols = all_of(q2_1_cols),
        names_to = "QCol",
        values_to = "Rating"
      ) %>%
      mutate(
        StatementID = as.integer(str_extract(QCol, "\\d+$")),
        Rating = as.numeric(str_extract(Rating, "\\d+")),  # Extract numeric rating
        RatingType = "Importance"
      ) %>%
      select(ParticipantID, StatementID, Rating, RatingType) %>%
      filter(!is.na(Rating))
  }
  
  # Process feasibility ratings (Q2.2)
  feasibility_ratings <- NULL
  if (length(q2_2_cols) > 0) {
    feasibility_ratings <- data %>%
      mutate(ParticipantID = row_number()) %>%
      select(ParticipantID, all_of(q2_2_cols)) %>%
      pivot_longer(
        cols = all_of(q2_2_cols),
        names_to = "QCol",
        values_to = "Rating"
      ) %>%
      mutate(
        StatementID = as.integer(str_extract(QCol, "\\d+$")),
        Rating = as.numeric(str_extract(Rating, "\\d+")),  # Extract numeric rating
        RatingType = "Feasibility"
      ) %>%
      select(ParticipantID, StatementID, Rating, RatingType) %>%
      filter(!is.na(Rating))
  }
  
  # Combine ratings
  all_ratings <- bind_rows(importance_ratings, feasibility_ratings)
  
  # Extract participant metadata
  participant_metadata <- data %>%
    mutate(ParticipantID = row_number()) %>%
    select(ParticipantID, all_of(meta_cols)) %>%
    filter(!is.na(ParticipantID))
  
  return(list(
    statements = statements_df,
    sorted_cards = sorted_cards,
    ratings = all_ratings,
    metadata = participant_metadata
  ))
}

# Function to process TSV file
process_tsv_file <- function(file_path) {
  cat("Processing TSV file:", file_path, "\n")
  
  # Read variable names and statement text
  header_data <- read.delim(file_path, nrows = 2, header = FALSE, stringsAsFactors = FALSE)
  var_names <- as.character(header_data[1, ])
  question_row <- as.character(header_data[2, ])
  
  # Read data, skipping first two lines
  data <- fread(file_path, skip = 2, header = FALSE, data.table = FALSE, sep = "\t")
  names(data) <- trimws(var_names)
  
  # Process the same way as CSV
  return(process_csv_file_data(data, var_names, question_row))
}

# Common processing function for data frames
process_csv_file_data <- function(data, var_names, question_row) {
  # Identify columns
  col_names <- names(data)
  q1_cols <- col_names[grepl("^Q1_", col_names)]
  q2_1_cols <- col_names[grepl("^Q2\\.1_", col_names)]
  q2_2_cols <- col_names[grepl("^Q2\\.2_", col_names)]
  meta_cols <- setdiff(col_names, c(q1_cols, q2_1_cols, q2_2_cols))
  
  # Extract statements from Q1 columns
  statements <- sapply(q1_cols, function(col) {
    idx <- which(col_names == col)
    txt <- question_row[idx]
    extract_statement_text(txt)
  })
  
  # Create statements dataframe
  statements_df <- data.frame(
    StatementID = seq_along(statements),
    StatementText = statements,
    stringsAsFactors = FALSE
  )
  
  # Process grouping data (Q1)
  sorted_cards <- NULL
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
        PileID = as.character(PileID)
      ) %>%
      select(ParticipantID, StatementID, PileID) %>%
      filter(!is.na(PileID) & PileID != "")
  }
  
  # Process importance ratings (Q2.1)
  importance_ratings <- NULL
  if (length(q2_1_cols) > 0) {
    importance_ratings <- data %>%
      mutate(ParticipantID = row_number()) %>%
      select(ParticipantID, all_of(q2_1_cols)) %>%
      pivot_longer(
        cols = all_of(q2_1_cols),
        names_to = "QCol",
        values_to = "Rating"
      ) %>%
      mutate(
        StatementID = as.integer(str_extract(QCol, "\\d+$")),
        Rating = as.numeric(str_extract(Rating, "\\d+")),
        RatingType = "Importance"
      ) %>%
      select(ParticipantID, StatementID, Rating, RatingType) %>%
      filter(!is.na(Rating))
  }
  
  # Process feasibility ratings (Q2.2)
  feasibility_ratings <- NULL
  if (length(q2_2_cols) > 0) {
    feasibility_ratings <- data %>%
      mutate(ParticipantID = row_number()) %>%
      select(ParticipantID, all_of(q2_2_cols)) %>%
      pivot_longer(
        cols = all_of(q2_2_cols),
        names_to = "QCol",
        values_to = "Rating"
      ) %>%
      mutate(
        StatementID = as.integer(str_extract(QCol, "\\d+$")),
        Rating = as.numeric(str_extract(Rating, "\\d+")),
        RatingType = "Feasibility"
      ) %>%
      select(ParticipantID, StatementID, Rating, RatingType) %>%
      filter(!is.na(Rating))
  }
  
  # Combine ratings
  all_ratings <- bind_rows(importance_ratings, feasibility_ratings)
  
  # Extract participant metadata
  participant_metadata <- data %>%
    mutate(ParticipantID = row_number()) %>%
    select(ParticipantID, all_of(meta_cols)) %>%
    filter(!is.na(ParticipantID))
  
  return(list(
    statements = statements_df,
    sorted_cards = sorted_cards,
    ratings = all_ratings,
    metadata = participant_metadata
  ))
}

# Process all files
all_data <- list()

for (file_path in input_files) {
  if (file.exists(file_path)) {
    file_ext <- tools::file_ext(file_path)
    
    if (file_ext == "csv") {
      all_data[[file_path]] <- process_csv_file(file_path)
    } else if (file_ext == "tsv") {
      all_data[[file_path]] <- process_tsv_file(file_path)
    }
  } else {
    cat("File not found:", file_path, "\n")
  }
}

# Combine data from all files
if (length(all_data) > 0) {
  # Use the first file's statements as the master list
  master_statements <- all_data[[1]]$statements
  
  # Combine sorted cards with offset participant IDs
  all_sorted_cards <- list()
  all_ratings <- list()
  all_metadata <- list()
  
  participant_offset <- 0
  
  for (i in seq_along(all_data)) {
    file_data <- all_data[[i]]
    
    # Adjust participant IDs to be unique across files
    if (!is.null(file_data$sorted_cards)) {
      adjusted_sorted_cards <- file_data$sorted_cards %>%
        mutate(ParticipantID = ParticipantID + participant_offset)
      all_sorted_cards[[i]] <- adjusted_sorted_cards
    }
    
    if (!is.null(file_data$ratings)) {
      adjusted_ratings <- file_data$ratings %>%
        mutate(ParticipantID = ParticipantID + participant_offset)
      all_ratings[[i]] <- adjusted_ratings
    }
    
    if (!is.null(file_data$metadata)) {
      adjusted_metadata <- file_data$metadata %>%
        mutate(ParticipantID = ParticipantID + participant_offset)
      all_metadata[[i]] <- adjusted_metadata
    }
    
    participant_offset <- participant_offset + max(file_data$metadata$ParticipantID, na.rm = TRUE)
  }
  
  # Combine all data
  combined_sorted_cards <- bind_rows(all_sorted_cards)
  combined_ratings <- bind_rows(all_ratings)
  combined_metadata <- bind_rows(all_metadata)
  
  # Write output files
  write_csv(master_statements, file.path(output_dir, "Statements.csv"))
  write_csv(combined_sorted_cards, file.path(output_dir, "SortedCards.csv"))
  write_csv(combined_ratings, file.path(output_dir, "Ratings.csv"))
  write_csv(combined_metadata, file.path(output_dir, "Demographics.csv"))
  
  # Print summary
  cat("\n=== Data Processing Summary ===\n")
  cat("Number of statements:", nrow(master_statements), "\n")
  cat("Number of participants:", length(unique(combined_sorted_cards$ParticipantID)), "\n")
  cat("Number of grouping responses:", nrow(combined_sorted_cards), "\n")
  cat("Number of rating responses:", nrow(combined_ratings), "\n")
  
  # Print first few statements
  cat("\n=== First 10 Statements ===\n")
  print(head(master_statements, 10))
  
  cat("\nData successfully transformed and saved to:", output_dir, "\n")
  
} else {
  cat("No data files were successfully processed.\n")
} 