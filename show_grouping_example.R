#!/usr/bin/env Rscript

# Show how grouping works with actual data example

library(dplyr)
library(readr)

# Example from the actual data (last row - Survey Preview)
# This participant grouped statements like this:
example_grouping <- data.frame(
  ParticipantID = 1,
  StatementID = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  PileID = c(4, 5, 4, 2, 3, 1, 6, 9, 3, 2),
  stringsAsFactors = FALSE
)

# Add some more examples to show grouping patterns
example_grouping <- rbind(
  example_grouping,
  data.frame(
    ParticipantID = 2,
    StatementID = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    PileID = c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
    stringsAsFactors = FALSE
  )
)

# Show the grouping
cat("EXAMPLE: How participants group statements together\n")
cat("==================================================\n\n")

cat("Participant 1 grouped statements like this:\n")
print(example_grouping[example_grouping$ParticipantID == 1, ])

cat("\nThis means:\n")
cat("- Statement 1 and Statement 3 both went to Group 4\n")
cat("- Statement 2 went to Group 5\n")
cat("- Statement 4 went to Group 2\n")
cat("- etc.\n\n")

cat("Participant 2 grouped statements like this:\n")
print(example_grouping[example_grouping$ParticipantID == 2, ])

cat("\nThis means:\n")
cat("- Statement 1 and Statement 2 both went to Group 1\n")
cat("- Statement 3 and Statement 4 both went to Group 2\n")
cat("- Statement 5 and Statement 6 both went to Group 3\n")
cat("- etc.\n\n")

# Show how RCMap would analyze this
cat("RCMap would then analyze:\n")
cat("- Which statements are frequently grouped together\n")
cat("- What themes emerge from the groupings\n")
cat("- How different participants think about the statements\n")
cat("- Create concept maps showing relationships between statements\n") 