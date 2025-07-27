#!/usr/bin/env Rscript

# Create a fully consistent RCMap sample dataset for 5 participants and 10 statements

library(dplyr)
library(readr)

cat("Creating fully consistent RCMap sample dataset (10 statements)...\n")

setwd("data")

n_participants <- 5
n_statements <- 10
n_piles_min <- 3
n_piles_max <- 5

# Demographics.csv (tabular, with headers)
demographics <- data.frame(
  RaterID = 1:n_participants,
  CompanySize = sample(c("large", "medium", "small"), n_participants, replace = TRUE),
  Years = round(runif(n_participants, 1, 10), 1)
)
write_csv(demographics, "Demographics.csv")

# Ratings.csv (tabular, with headers, long format)
ratings <- expand.grid(
  RaterID = 1:n_participants,
  StatementID = 1:n_statements
) %>%
  mutate(
    Feasibility = sample(1:5, n(), replace = TRUE),
    Importance = sample(1:5, n(), replace = TRUE)
  )
write_csv(ratings, "Ratings.csv")

# SortedCards.csv (pile matrix, no header, all statements assigned, no duplicates, correct padding)
set.seed(42)
pile_rows <- list()
for (pid in 1:n_participants) {
  n_piles <- sample(n_piles_min:n_piles_max, 1)
  # Randomly assign statements to piles
  statement_ids <- sample(1:n_statements)
  pile_sizes <- rep(floor(n_statements / n_piles), n_piles)
  pile_sizes[1:(n_statements %% n_piles)] <- pile_sizes[1:(n_statements %% n_piles)] + 1
  idx <- 1
  for (pile in 1:n_piles) {
    cards <- statement_ids[idx:(idx + pile_sizes[pile] - 1)]
    idx <- idx + pile_sizes[pile]
    # Each row: participant, pile number, card numbers, padded with blanks
    row <- c(pid, pile, cards)
    # Pad with blanks to max possible columns
    row <- c(row, rep("", n_statements + 2 - length(row)))
    pile_rows[[length(pile_rows) + 1]] <- row
  }
}
# Write pile matrix to CSV (no header)
pile_matrix <- do.call(rbind, pile_rows)
write.table(pile_matrix, "SortedCards.csv", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

cat("Consistent RCMap files created in data/ directory!\n")

# Also copy to rcmap_workshop directory
file.copy("Demographics.csv", "rcmap_workshop/Demographics.csv", overwrite = TRUE)
file.copy("Ratings.csv", "rcmap_workshop/Ratings.csv", overwrite = TRUE)
file.copy("SortedCards.csv", "rcmap_workshop/SortedCards.csv", overwrite = TRUE)

cat("Consistent RCMap files also copied to data/rcmap_workshop/!\n")

cat("\nReady to run RCMap! Use:\n")
cat("library(RCMap)\n")
cat("RCMapMenu()\n")
cat("Then select the 'data' or 'data/rcmap_workshop' folder\n") 