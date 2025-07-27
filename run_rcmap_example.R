#!/usr/bin/env Rscript

# Run RCMap with the Workshop data

library(RCMap)
library(dplyr)
library(readr)

cat("Running RCMap with Workshop data...\n")

# Set working directory to the data folder
setwd("data/rcmap_workshop")

# Check what files we have
cat("Available files:\n")
print(list.files())

# Read the statements
statements <- read_csv("Statements.csv")
cat("\nNumber of statements:", nrow(statements), "\n")

# Since most participants didn't complete Q1 (grouping), let's create a simple example
# with the data we do have from the last row (Survey Preview)

cat("\nCreating example data from Survey Preview participant...\n")

# Example grouping data from the Survey Preview participant
example_sorted_cards <- data.frame(
  ParticipantID = 1,
  StatementID = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
  PileID = c(4, 5, 4, 2, 3, 1, 6, 9, 3, 2, 4, 5, 6, 7, 8),
  stringsAsFactors = FALSE
)

# Example ratings data (1-5 scale)
example_ratings <- data.frame(
  ParticipantID = 1,
  StatementID = 1:15,
  Rating = c(3, 2, 2, 4, 3, 1, 4, 1, 3, 5, 3, 3, 4, 2, 3),
  stringsAsFactors = FALSE
)

# Write the example data
write_csv(example_sorted_cards, "SortedCards_example.csv")
write_csv(example_ratings, "Ratings_example.csv")

cat("Created example files with data from Survey Preview participant\n")

# Now run RCMap with the example data
cat("\nRunning RCMap analysis...\n")

# Run RCMap with the example data
result <- RCMap(
  statements = "Statements.csv",
  sortedCards = "SortedCards_example.csv", 
  ratings = "Ratings_example.csv",
  demographics = "Demographics.csv"
)

cat("RCMap analysis complete!\n")
cat("Results saved in the current directory\n")

# Show what was created
cat("\nFiles created by RCMap:\n")
print(list.files(pattern = "*.pdf"))
print(list.files(pattern = "*.csv")) 