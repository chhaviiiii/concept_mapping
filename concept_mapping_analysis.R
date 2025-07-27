#!/usr/bin/env Rscript

# Comprehensive Concept Mapping Analysis for BCCS AI Workshop July 27, 2025
# This script performs complete concept mapping analysis on the 100 statements
# about AI in cancer care, including:
# - Grouping analysis (how participants grouped statements)
# - Rating analysis (importance and feasibility)
# - Multidimensional scaling and clustering
# - Visualization of concept maps
# - Statistical analysis of patterns

library(dplyr)
library(readr)
library(ggplot2)
library(ggrepel)
library(cluster)
library(factoextra)
library(MASS)
library(corrplot)
library(viridis)
library(RColorBrewer)
library(gridExtra)
library(knitr)
library(kableExtra)
library(tidyr)
library(stringr)
library(purrr)

# Load RCMap functions
source("R/RCMap.R")

# Configuration
data_dir <- "data/rcmap_july27_2025"
output_dir <- "Figures/july27_2025_analysis"

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Load data
cat("Loading transformed data...\n")
statements <- read_csv(file.path(data_dir, "Statements.csv"))
sorted_cards <- read_csv(file.path(data_dir, "SortedCards.csv"))
ratings <- read_csv(file.path(data_dir, "Ratings.csv"))
demographics <- read_csv(file.path(data_dir, "Demographics.csv"))

# Data validation and cleaning
cat("Validating and cleaning data...\n")

# Check for missing data
cat("Data summary:\n")
cat("Statements:", nrow(statements), "\n")
cat("Participants:", length(unique(sorted_cards$ParticipantID)), "\n")
cat("Grouping responses:", nrow(sorted_cards), "\n")
cat("Rating responses:", nrow(ratings), "\n")

# Clean pile IDs - convert "Group X" to numeric
sorted_cards_clean <- sorted_cards %>%
  mutate(
    PileID = str_extract(PileID, "\\d+"),
    PileID = as.integer(PileID)
  ) %>%
  filter(!is.na(PileID))

# Create participant dictionary for RCMap
participants <- unique(sorted_cards_clean$ParticipantID)
sorters_dict <- data.frame(
  orig_ID = participants,
  seq_ID = seq_along(participants),
  stringsAsFactors = FALSE
)

# Create card names for RCMap
card_names <- data.frame(
  CardID = statements$StatementID,
  CardName = statements$StatementText,
  stringsAsFactors = FALSE
)

# Prepare pile sorting data for RCMap
pile_data <- sorted_cards_clean %>%
  dplyr::select(ParticipantID, PileID, StatementID) %>%
  arrange(ParticipantID, PileID, StatementID)

# Convert to RCMap format (ParticipantID, PileName, Card1, Card2, ...)
pile_data_rcmap <- pile_data %>%
  group_by(ParticipantID, PileID) %>%
  summarise(
    cards = list(StatementID),
    .groups = 'drop'
  ) %>%
  mutate(
    max_cards = max(sapply(cards, length)),
    cards_padded = lapply(cards, function(x) {
      if (length(x) < max_cards) {
        c(x, rep(NA, max_cards - length(x)))
      } else {
        x
      }
    })
  ) %>%
  unnest(cards_padded) %>%
  group_by(ParticipantID, PileID) %>%
  mutate(card_col = paste0("Card", row_number())) %>%
  ungroup() %>%
  pivot_wider(
    names_from = card_col,
    values_from = cards_padded
  ) %>%
  dplyr::select(ParticipantID, PileID, starts_with("Card"))

# Function to perform concept mapping analysis
perform_concept_mapping <- function(pile_data, card_names, sorters_dict) {
  cat("Performing concept mapping analysis...\n")
  
  # Get adjacency matrices
  adj_result <- getAdjMatrices(pile_data, card_names, sorters_dict, showWarnings = TRUE)
  adj_matrices <- adj_result$mat.list
  
  # Create distance matrix
  dist_matrix <- distanceMatrix(adj_matrices)
  
  # Perform MDS
  mds_result <- cmdscale(dist_matrix, k = 2)
  
  # Perform clustering on MDS coordinates
  # Determine optimal number of clusters
  wss <- fviz_nbclust(mds_result, kmeans, method = "wss", k.max = 15)
  silhouette <- fviz_nbclust(mds_result, kmeans, method = "silhouette", k.max = 15)
  
  # Use gap statistic to find optimal clusters
  gap_stat <- fviz_nbclust(mds_result, kmeans, method = "gap_stat", k.max = 15)
  
  # Perform k-means clustering (using elbow method to determine k)
  elbow_data <- wss$data
  optimal_k <- which.min(diff(elbow_data$y)) + 1
  optimal_k <- min(max(optimal_k, 3), 10)  # Between 3 and 10 clusters
  
  cat("Optimal number of clusters:", optimal_k, "\n")
  
  kmeans_result <- kmeans(mds_result, centers = optimal_k, nstart = 25)
  
  return(list(
    mds_coords = mds_result,
    dist_matrix = dist_matrix,
    clusters = kmeans_result$cluster,
    optimal_k = optimal_k,
    wss_plot = wss,
    silhouette_plot = silhouette,
    gap_plot = gap_stat
  ))
}

# Function to analyze ratings
analyze_ratings <- function(ratings, statements) {
  cat("Analyzing ratings...\n")
  
  # Calculate summary statistics by statement and rating type
  rating_summary <- ratings %>%
    group_by(StatementID, RatingType) %>%
    summarise(
      mean_rating = mean(Rating, na.rm = TRUE),
      median_rating = median(Rating, na.rm = TRUE),
      sd_rating = sd(Rating, na.rm = TRUE),
      n_ratings = n(),
      .groups = 'drop'
    ) %>%
    left_join(statements, by = "StatementID")
  
  # Calculate correlation between importance and feasibility
  importance_feasibility <- ratings %>%
    pivot_wider(
      names_from = RatingType,
      values_from = Rating,
      values_fn = mean
    ) %>%
    group_by(StatementID) %>%
    summarise(
      Importance = mean(Importance, na.rm = TRUE),
      Feasibility = mean(Feasibility, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    left_join(statements, by = "StatementID")
  
  return(list(
    rating_summary = rating_summary,
    importance_feasibility = importance_feasibility
  ))
}

# Function to create visualizations
create_visualizations <- function(cm_result, rating_analysis, statements, output_dir) {
  cat("Creating visualizations...\n")
  
  # 1. Concept Map (MDS with clusters)
  mds_df <- data.frame(
    X = cm_result$mds_coords[, 1],
    Y = cm_result$mds_coords[, 2],
    StatementID = statements$StatementID,
    StatementText = statements$StatementText,
    Cluster = as.factor(cm_result$clusters)
  )
  
  # Create concept map
  concept_map <- ggplot(mds_df, aes(x = X, y = Y, color = Cluster)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_text_repel(
      aes(label = StatementID),
      size = 3,
      max.overlaps = 20,
      box.padding = 0.5
    ) +
    scale_color_viridis_d() +
    labs(
      title = "Concept Map: AI in Cancer Care Statements",
      subtitle = "Multidimensional Scaling with K-means Clustering",
      x = "Dimension 1",
      y = "Dimension 2",
      color = "Cluster"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom"
    )
  
  ggsave(file.path(output_dir, "concept_map.png"), concept_map, 
         width = 12, height = 10, dpi = 300)
  
  # 2. Importance vs Feasibility Plot
  importance_feasibility_plot <- ggplot(rating_analysis$importance_feasibility, 
                                       aes(x = Importance, y = Feasibility)) +
    geom_point(size = 3, alpha = 0.7, color = "steelblue") +
    geom_text_repel(
      aes(label = StatementID),
      size = 3,
      max.overlaps = 15
    ) +
    geom_hline(yintercept = mean(rating_analysis$importance_feasibility$Feasibility, na.rm = TRUE), 
               linetype = "dashed", color = "red") +
    geom_vline(xintercept = mean(rating_analysis$importance_feasibility$Importance, na.rm = TRUE), 
               linetype = "dashed", color = "red") +
    labs(
      title = "Importance vs Feasibility of AI in Cancer Care Statements",
      subtitle = "Red lines indicate mean values",
      x = "Importance Rating (1-5)",
      y = "Feasibility Rating (1-5)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"))
  
  ggsave(file.path(output_dir, "importance_vs_feasibility.png"), 
         importance_feasibility_plot, width = 12, height = 10, dpi = 300)
  
  # 3. Rating Distribution by Type
  rating_dist_plot <- ggplot(rating_analysis$rating_summary, 
                            aes(x = mean_rating, fill = RatingType)) +
    geom_histogram(binwidth = 0.2, alpha = 0.7, position = "identity") +
    scale_fill_brewer(palette = "Set1") +
    labs(
      title = "Distribution of Mean Ratings by Type",
      x = "Mean Rating",
      y = "Count",
      fill = "Rating Type"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"))
  
  ggsave(file.path(output_dir, "rating_distribution.png"), 
         rating_dist_plot, width = 10, height = 6, dpi = 300)
  
  # 4. Cluster Analysis Plots
  # WSS plot
  ggsave(file.path(output_dir, "wss_plot.png"), 
         cm_result$wss_plot, width = 8, height = 6, dpi = 300)
  
  # Silhouette plot
  ggsave(file.path(output_dir, "silhouette_plot.png"), 
         cm_result$silhouette_plot, width = 8, height = 6, dpi = 300)
  
  # Gap statistic plot
  ggsave(file.path(output_dir, "gap_stat_plot.png"), 
         cm_result$gap_plot, width = 8, height = 6, dpi = 300)
  
  return(list(
    concept_map = concept_map,
    importance_feasibility_plot = importance_feasibility_plot,
    rating_dist_plot = rating_dist_plot
  ))
}

# Function to generate reports
generate_reports <- function(cm_result, rating_analysis, statements, output_dir) {
  cat("Generating reports...\n")
  
  # Create cluster summary
  cluster_summary <- data.frame(
    StatementID = statements$StatementID,
    StatementText = statements$StatementText,
    Cluster = cm_result$clusters,
    stringsAsFactors = FALSE
  ) %>%
    arrange(Cluster, StatementID)
  
  # Create rating summary by cluster
  cluster_ratings <- cluster_summary %>%
    left_join(rating_analysis$importance_feasibility, by = c("StatementID", "StatementText")) %>%
    group_by(Cluster) %>%
    summarise(
      n_statements = n(),
      mean_importance = mean(Importance, na.rm = TRUE),
      mean_feasibility = mean(Feasibility, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # Write reports
  write_csv(cluster_summary, file.path(output_dir, "cluster_summary.csv"))
  write_csv(cluster_ratings, file.path(output_dir, "cluster_ratings.csv"))
  write_csv(rating_analysis$importance_feasibility, 
            file.path(output_dir, "importance_feasibility_summary.csv"))
  
  # Create HTML report
  html_report <- paste0('
<!DOCTYPE html>
<html>
<head>
    <title>BCCS AI Workshop Concept Mapping Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #2c3e50; }
        .cluster { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }
        .statement { margin: 5px 0; padding: 5px; background-color: #f8f9fa; }
        .stats { background-color: #ecf0f1; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>BCCS AI Workshop Concept Mapping Analysis</h1>
    <p><strong>Date:</strong> July 27, 2025</p>
    <p><strong>Total Statements:</strong> ', nrow(statements), '</p>
    <p><strong>Total Participants:</strong> ', length(unique(sorted_cards$ParticipantID)), '</p>
    <p><strong>Optimal Clusters:</strong> ', cm_result$optimal_k, '</p>
    
    <h2>Cluster Summary</h2>
    <div class="stats">
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><th>Cluster</th><th>Statements</th><th>Mean Importance</th><th>Mean Feasibility</th></tr>')
  
  # Add cluster rows
  for (i in 1:nrow(cluster_ratings)) {
    html_report <- paste0(html_report, '
            <tr><td>', cluster_ratings$Cluster[i], '</td><td>', 
                 cluster_ratings$n_statements[i], '</td><td>', 
                 round(cluster_ratings$mean_importance[i], 2), '</td><td>', 
                 round(cluster_ratings$mean_feasibility[i], 2), '</td></tr>')
  }
  
  html_report <- paste0(html_report, '
        </table>
    </div>
    
    <h2>Statements by Cluster</h2>')
  
  # Add clusters
  for (cluster_id in 1:cm_result$optimal_k) {
    cluster_statements <- cluster_summary[cluster_summary$Cluster == cluster_id, ]
    html_report <- paste0(html_report, '
        <div class="cluster">
            <h3>Cluster ', cluster_id, ' (', nrow(cluster_statements), ' statements)</h3>')
    
    for (i in 1:nrow(cluster_statements)) {
      html_report <- paste0(html_report, '
            <div class="statement"><strong>', cluster_statements$StatementID[i], 
                           ':</strong> ', cluster_statements$StatementText[i], '</div>')
    }
    
    html_report <- paste0(html_report, '
        </div>')
  }
  
  # Add top statements
  top_important <- head(rating_analysis$importance_feasibility[order(-rating_analysis$importance_feasibility$Importance), ], 10)
  top_feasible <- head(rating_analysis$importance_feasibility[order(-rating_analysis$importance_feasibility$Feasibility), ], 10)
  
  html_report <- paste0(html_report, '
    
    <h2>Top Statements by Importance</h2>
    <div class="stats">')
  
  for (i in 1:nrow(top_important)) {
    html_report <- paste0(html_report, '
        <div class="statement"><strong>', top_important$StatementID[i], 
                       ':</strong> ', top_important$StatementText[i], ' (Importance: ', 
                       round(top_important$Importance[i], 2), ')</div>')
  }
  
  html_report <- paste0(html_report, '
    </div>
    
    <h2>Top Statements by Feasibility</h2>
    <div class="stats">')
  
  for (i in 1:nrow(top_feasible)) {
    html_report <- paste0(html_report, '
        <div class="statement"><strong>', top_feasible$StatementID[i], 
                       ':</strong> ', top_feasible$StatementText[i], ' (Feasibility: ', 
                       round(top_feasible$Feasibility[i], 2), ')</div>')
  }
  
  html_report <- paste0(html_report, '
    </div>
</body>
</html>')
  
  writeLines(html_report, file.path(output_dir, "analysis_report.html"))
  
  return(list(
    cluster_summary = cluster_summary,
    cluster_ratings = cluster_ratings
  ))
}

# Main analysis
cat("Starting comprehensive concept mapping analysis...\n")

# Perform concept mapping
cm_result <- perform_concept_mapping(pile_data_rcmap, card_names, sorters_dict)

# Analyze ratings
rating_analysis <- analyze_ratings(ratings, statements)

# Create visualizations
visualizations <- create_visualizations(cm_result, rating_analysis, statements, output_dir)

# Generate reports
reports <- generate_reports(cm_result, rating_analysis, statements, output_dir)

# Print summary
cat("\n=== Analysis Complete ===\n")
cat("Output files saved to:", output_dir, "\n")
cat("Number of clusters:", cm_result$optimal_k, "\n")
cat("Correlation between importance and feasibility:", 
    cor(rating_analysis$importance_feasibility$Importance, 
        rating_analysis$importance_feasibility$Feasibility, use = "complete.obs"), "\n")

# Print top statements
cat("\n=== Top 5 Most Important Statements ===\n")
top_important <- head(rating_analysis$importance_feasibility[order(-rating_analysis$importance_feasibility$Importance), ], 5)
for (i in 1:nrow(top_important)) {
  cat(i, ". Statement", top_important$StatementID[i], ": ", top_important$StatementText[i], 
      " (Importance:", round(top_important$Importance[i], 2), ")\n")
}

cat("\n=== Top 5 Most Feasible Statements ===\n")
top_feasible <- head(rating_analysis$importance_feasibility[order(-rating_analysis$importance_feasibility$Feasibility), ], 5)
for (i in 1:nrow(top_feasible)) {
  cat(i, ". Statement", top_feasible$StatementID[i], ": ", top_feasible$StatementText[i], 
      " (Feasibility:", round(top_feasible$Feasibility[i], 2), ")\n")
}

cat("\nAnalysis complete! Check the output directory for all results.\n") 