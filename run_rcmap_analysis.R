#!/usr/bin/env Rscript

# Run RCMap analysis step by step

library(RCMap)
library(dplyr)
library(readr)

cat("Running RCMap analysis step by step...\n")

# Set working directory to the data folder
setwd("data/rcmap_workshop")

# Read the data
statements <- read_csv("Statements.csv")
cat("Loaded", nrow(statements), "statements\n")

# Create example data since most participants didn't complete the tasks
# Using data from the Survey Preview participant

# Example grouping data
example_sorted_cards <- data.frame(
  ParticipantID = 1,
  StatementID = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
  PileID = c(4, 5, 4, 2, 3, 1, 6, 9, 3, 2, 4, 5, 6, 7, 8),
  stringsAsFactors = FALSE
)

# Example ratings data
example_ratings <- data.frame(
  ParticipantID = 1,
  StatementID = 1:15,
  Rating = c(3, 2, 2, 4, 3, 1, 4, 1, 3, 5, 3, 3, 4, 2, 3),
  stringsAsFactors = FALSE
)

# Write the example data
write_csv(example_sorted_cards, "SortedCards_example.csv")
write_csv(example_ratings, "Ratings_example.csv")

cat("Created example data files\n")

# Run individual RCMap analyses

# 1. Statement Report
cat("\n1. Generating Statement Report...\n")
statementReport(
  statements = "Statements.csv",
  sortedCards = "SortedCards_example.csv",
  ratings = "Ratings_example.csv"
)

# 2. Sorter Report (grouping analysis)
cat("\n2. Generating Sorter Report...\n")
sorterReport(
  statements = "Statements.csv", 
  sortedCards = "SortedCards_example.csv"
)

# 3. Rater Report (rating analysis)
cat("\n3. Generating Rater Report...\n")
raterReport(
  statements = "Statements.csv",
  ratings = "Ratings_example.csv"
)

# 4. Rating Summary
cat("\n4. Generating Rating Summary...\n")
ratingSummary(
  statements = "Statements.csv",
  ratings = "Ratings_example.csv"
)

cat("\nRCMap analysis complete!\n")
cat("Check the current directory for output files\n")

# Show what was created
cat("\nFiles created:\n")
print(list.files(pattern = "*.pdf"))
print(list.files(pattern = "*.csv")) 