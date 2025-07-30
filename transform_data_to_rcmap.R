#!/usr/bin/env Rscript

# =============================================================================
# Transform Concept Mapping Data to RCMap Format
# =============================================================================
#
# This script converts Qualtrics survey data into RCMap format for
# concept mapping analysis. It handles CSV and TSV files and extracts
# statements, ratings, and demographics for analysis.
#
# This implementation is designed for researchers conducting concept mapping studies
# in healthcare, education, business, or any domain requiring structured analysis
# of complex ideas and their relationships.
#
# Author: Concept Mapping Analysis Team
# Date: 2025
# License: Educational and Research Use
# =============================================================================

# Load required libraries
library(dplyr)
library(readr)
library(tidyr)
library(stringr)

# =============================================================================
# Configuration
# =============================================================================

# Set input and output directories
input_dir <- "data"
output_dir <- "data/rcmap_analysis"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("✅ Created output directory:", output_dir, "\n")
}

# =============================================================================
# Data Processing Functions
# =============================================================================

process_csv_file <- function(file_path) {
  # Process CSV file from Qualtrics survey.
  #
  # This function extracts statements, ratings, and demographics from
  # a Qualtrics CSV export file.
  #
  # Args:
  #   file_path: Path to the CSV file
  #
  # Returns:
  #   List containing statements, ratings, and demographics data
  
  cat("Processing CSV file:", file_path, "\n")
  
  # Read the CSV file - skip first 2 rows to get to the data
  df <- read_csv(file_path, skip = 2, show_col_types = FALSE)
  
  cat("Column names after skipping 2 rows:\n")
  print(head(names(df), 20))  # Show first 20 column names
  
  # Extract statements from Q1_x columns
  statements <- data.frame()
  for (i in 1:100) {
    col_name <- paste0("Q1_", i)
    if (col_name %in% names(df)) {
      statement_text <- df[[col_name]][1]  # First row contains statement text
      if (!is.na(statement_text) && str_trim(statement_text) != "") {
        statements <- rbind(statements, data.frame(
          StatementID = i,
          StatementText = str_trim(statement_text)
        ))
      }
    }
  }
  
  # Extract ratings data
  ratings <- data.frame()
  demographics <- data.frame()
  
  for (i in 1:nrow(df)) {
    participant_id <- paste0("P", i)
    
    # Extract demographics (if available)
    demographics <- rbind(demographics, data.frame(
      ParticipantID = participant_id,
      Age = ifelse("Q6" %in% names(df), df$Q6[i], ""),
      Gender = ifelse("Q7" %in% names(df), df$Q7[i], ""),
      Role = "",
      Experience = ""
    ))
    
    # Extract importance and feasibility ratings
    for (j in 1:100) {
      importance_col <- paste0("Q2.1_", j)  # Importance rating
      feasibility_col <- paste0("Q2.2_", j)  # Feasibility rating
      
      if (importance_col %in% names(df) && feasibility_col %in% names(df)) {
        importance <- df[[importance_col]][i]
        feasibility <- df[[feasibility_col]][i]
        
        if (!is.na(importance) && !is.na(feasibility)) {
          tryCatch({
            importance_val <- as.numeric(importance)
            feasibility_val <- as.numeric(feasibility)
            
            if (!is.na(importance_val) && !is.na(feasibility_val)) {
              ratings <- rbind(ratings, data.frame(
                ParticipantID = participant_id,
                StatementID = j,
                RatingType = "Importance",
                Rating = importance_val
              ))
              ratings <- rbind(ratings, data.frame(
                ParticipantID = participant_id,
                StatementID = j,
                RatingType = "Feasibility",
                Rating = feasibility_val
              ))
            }
          }, error = function(e) {
            # Skip non-numeric ratings
          })
        }
      }
    }
  }
  
  return(list(
    statements = statements,
    ratings = ratings,
    demographics = demographics
  ))
}

process_tsv_file <- function(file_path) {
  # Process TSV file from Qualtrics survey.
  #
  # This function extracts statements, ratings, and demographics from
  # a Qualtrics TSV export file.
  #
  # Args:
  #   file_path: Path to the TSV file
  #
  # Returns:
  #   List containing statements, ratings, and demographics data
  
  cat("Processing TSV file:", file_path, "\n")
  
  # Read the TSV file
  df <- read_tsv(file_path, show_col_types = FALSE)
  
  # Extract statements from the first few rows
  statements <- data.frame()
  for (i in 1:100) {  # 100 statements
    col_name <- paste0("Q", i)
    if (col_name %in% names(df)) {
      statement_text <- df[[col_name]][1]  # First row contains statement text
      if (!is.na(statement_text) && str_trim(statement_text) != "") {
        statements <- rbind(statements, data.frame(
          StatementID = i,
          StatementText = str_trim(statement_text)
        ))
      }
    }
  }
  
  # Extract ratings data
  ratings <- data.frame()
  demographics <- data.frame()
  
  for (i in 3:nrow(df)) {  # Skip header rows
    participant_id <- paste0("P", i-2)
    
    # Extract demographics
    demographics <- rbind(demographics, data.frame(
      ParticipantID = participant_id,
      Age = ifelse("Q101" %in% names(df), df$Q101[i], ""),
      Gender = ifelse("Q102" %in% names(df), df$Q102[i], ""),
      Role = ifelse("Q103" %in% names(df), df$Q103[i], ""),
      Experience = ifelse("Q104" %in% names(df), df$Q104[i], "")
    ))
    
    # Extract importance and feasibility ratings
    for (j in 1:100) {
      importance_col <- paste0("Q", j, "_1")  # Importance rating
      feasibility_col <- paste0("Q", j, "_2")  # Feasibility rating
      
      if (importance_col %in% names(df) && feasibility_col %in% names(df)) {
        importance <- df[[importance_col]][i]
        feasibility <- df[[feasibility_col]][i]
        
        if (!is.na(importance) && !is.na(feasibility)) {
          ratings <- rbind(ratings, data.frame(
            ParticipantID = participant_id,
            StatementID = j,
            RatingType = "Importance",
            Rating = as.numeric(importance)
          ))
          ratings <- rbind(ratings, data.frame(
            ParticipantID = participant_id,
            StatementID = j,
            RatingType = "Feasibility",
            Rating = as.numeric(feasibility)
          ))
        }
      }
    }
  }
  
  return(list(
    statements = statements,
    ratings = ratings,
    demographics = demographics
  ))
}

save_transformed_data <- function(statements, ratings, demographics, output_dir) {
  # Save transformed data to CSV files.
  #
  # Args:
  #   statements: Dataframe of statements
  #   ratings: Dataframe of ratings
  #   demographics: Dataframe of demographics
  #   output_dir: Output directory for transformed data
  
  # Save to CSV files
  write_csv(statements, file.path(output_dir, "Statements.csv"))
  write_csv(ratings, file.path(output_dir, "Ratings.csv"))
  write_csv(demographics, file.path(output_dir, "Demographics.csv"))
  
  # Create a placeholder sorted cards file (empty for now)
  sorted_cards <- data.frame(
    ParticipantID = character(),
    StatementID = integer(),
    GroupID = integer(),
    stringsAsFactors = FALSE
  )
  write_csv(sorted_cards, file.path(output_dir, "SortedCards.csv"))
  
  cat("✅ Transformed data saved to", output_dir, "\n")
  cat("   -", nrow(statements), "statements\n")
  cat("   -", nrow(ratings), "ratings\n")
  cat("   -", nrow(demographics), "participants\n")
}

# =============================================================================
# Main Transformation Workflow
# =============================================================================

main <- function() {
  # Main function to transform concept mapping data.
  #
  # This function processes Qualtrics survey data and converts it to
  # RCMap format for analysis.
  
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("CONCEPT MAPPING DATA TRANSFORMATION - R\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  # Check if input directory exists
  if (!dir.exists(input_dir)) {
    cat("❌ Input directory '", input_dir, "' not found!\n", sep = "")
    cat("Please place your Qualtrics CSV/TSV files in the 'data' directory.\n")
    return()
  }
  
  # Find CSV and TSV files
  csv_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)
  tsv_files <- list.files(input_dir, pattern = "\\.tsv$", full.names = TRUE)
  
  if (length(csv_files) == 0 && length(tsv_files) == 0) {
    cat("❌ No CSV or TSV files found in '", input_dir, "'!\n", sep = "")
    cat("Please add your Qualtrics export files to the data directory.\n")
    return()
  }
  
  # Process the first file found
  if (length(csv_files) > 0) {
    file_path <- csv_files[1]
    cat("Found CSV file:", file_path, "\n")
    result <- process_csv_file(file_path)
  } else if (length(tsv_files) > 0) {
    file_path <- tsv_files[1]
    cat("Found TSV file:", file_path, "\n")
    result <- process_tsv_file(file_path)
  }
  
  # Save transformed data
  save_transformed_data(
    result$statements,
    result$ratings,
    result$demographics,
    output_dir
  )
  
  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat("DATA TRANSFORMATION COMPLETED!\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat("📁 Transformed data is ready for R analysis\n")
  cat("🚀 Run: Rscript concept_mapping_analysis.R\n")
}

# Run the main function
main() 