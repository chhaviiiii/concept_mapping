#!/usr/bin/env Rscript

# Run RCMap interactive menu

library(RCMap)

cat("Starting RCMap interactive menu...\n")
cat("This will open a GUI where you can:\n")
cat("1. Select your data files\n")
cat("2. Choose which analyses to run\n")
cat("3. Generate reports and visualizations\n\n")

cat("Your data files are in: data/rcmap_workshop/\n")
cat("- Statements.csv (100 statements)\n")
cat("- SortedCards_example.csv (example grouping data)\n")
cat("- Ratings_example.csv (example rating data)\n")
cat("- Demographics.csv (participant info)\n\n")

cat("Opening RCMap menu...\n")

# Open the interactive menu
RCMapMenu()

cat("RCMap analysis complete!\n") 