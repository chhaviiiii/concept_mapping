#!/usr/bin/env Rscript

# Simplified Concept Mapping Analysis for BCCS AI Workshop July 27, 2025
# This script performs analysis focusing on the rating data since grouping data is incomplete
# - Rating analysis (importance and feasibility)
# - Statement similarity based on rating patterns
# - Clustering and visualization
# - Statistical analysis

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
library(tidyr)
library(stringr)
library(purrr)
library(tibble)

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
ratings <- read_csv(file.path(data_dir, "Ratings.csv"))
demographics <- read_csv(file.path(data_dir, "Demographics.csv"))

# Data validation and cleaning
cat("Validating and cleaning data...\n")

# Check for missing data
cat("Data summary:\n")
cat("Statements:", nrow(statements), "\n")
cat("Participants:", length(unique(ratings$ParticipantID)), "\n")
cat("Rating responses:", nrow(ratings), "\n")

# Function to analyze ratings and create statement similarity matrix
analyze_ratings_and_similarity <- function(ratings, statements) {
  cat("Analyzing ratings and creating similarity matrix...\n")
  
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
  
  # Create wide format for importance and feasibility
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
  
  # Create similarity matrix based on rating patterns
  # Pivot so each row is a statement, columns are participant-ratingtype
  rating_matrix <- ratings %>%
    mutate(ParticipantRatingType = paste0("P", ParticipantID, "_", RatingType)) %>%
    as.data.frame() %>%
    dplyr::select(StatementID, ParticipantRatingType, Rating) %>%
    arrange(StatementID) %>%
    pivot_wider(names_from = ParticipantRatingType, values_from = Rating) %>%
    column_to_rownames("StatementID") %>%
    as.matrix()

  # Calculate correlation matrix between statements
  similarity_matrix <- cor(t(rating_matrix), use = "pairwise.complete.obs")
  
  # Convert to distance matrix
  distance_matrix <- 1 - abs(similarity_matrix)
  diag(distance_matrix) <- 0
  
  # Handle NA values by replacing with 0
  distance_matrix[is.na(distance_matrix)] <- 0
  
  # Ensure the matrix is symmetric and valid
  distance_matrix <- (distance_matrix + t(distance_matrix)) / 2
  diag(distance_matrix) <- 0
  
  return(list(
    rating_summary = rating_summary,
    importance_feasibility = importance_feasibility,
    similarity_matrix = similarity_matrix,
    distance_matrix = distance_matrix
  ))
}

# Function to perform clustering analysis
perform_clustering_analysis <- function(distance_matrix, statements) {
  cat("Performing clustering analysis...\n")
  
  # Perform MDS on distance matrix
  mds_result <- cmdscale(distance_matrix, k = 2)
  
  # Determine optimal number of clusters
  wss <- fviz_nbclust(mds_result, kmeans, method = "wss", k.max = 15)
  silhouette <- fviz_nbclust(mds_result, kmeans, method = "silhouette", k.max = 15)
  gap_stat <- fviz_nbclust(mds_result, kmeans, method = "gap_stat", k.max = 15)
  
  # Use elbow method to determine optimal k
  elbow_data <- wss$data
  optimal_k <- which.min(diff(elbow_data$y)) + 1
  optimal_k <- min(max(optimal_k, 3), 10)  # Between 3 and 10 clusters
  
  cat("Optimal number of clusters:", optimal_k, "\n")
  
  # Perform k-means clustering
  kmeans_result <- kmeans(mds_result, centers = optimal_k, nstart = 25)
  
  return(list(
    mds_coords = mds_result,
    clusters = kmeans_result$cluster,
    optimal_k = optimal_k,
    wss_plot = wss,
    silhouette_plot = silhouette,
    gap_plot = gap_stat
  ))
}

# Function to create visualizations
create_visualizations <- function(clustering_result, rating_analysis, statements, output_dir) {
  cat("Creating visualizations...\n")
  
  # 1. Concept Map (MDS with clusters)
  mds_df <- data.frame(
    X = clustering_result$mds_coords[, 1],
    Y = clustering_result$mds_coords[, 2],
    StatementID = statements$StatementID,
    StatementText = statements$StatementText,
    Cluster = as.factor(clustering_result$clusters)
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
      subtitle = "Based on Rating Pattern Similarity",
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
  ggsave(file.path(output_dir, "wss_plot.png"), 
         clustering_result$wss_plot, width = 8, height = 6, dpi = 300)
  
  ggsave(file.path(output_dir, "silhouette_plot.png"), 
         clustering_result$silhouette_plot, width = 8, height = 6, dpi = 300)
  
  ggsave(file.path(output_dir, "gap_stat_plot.png"), 
         clustering_result$gap_plot, width = 8, height = 6, dpi = 300)
  
  return(list(
    concept_map = concept_map,
    importance_feasibility_plot = importance_feasibility_plot,
    rating_dist_plot = rating_dist_plot
  ))
}

# Function to generate reports
generate_reports <- function(clustering_result, rating_analysis, statements, output_dir) {
  cat("Generating reports...\n")
  
  # Create cluster summary
  cluster_summary <- data.frame(
    StatementID = statements$StatementID,
    StatementText = statements$StatementText,
    Cluster = clustering_result$clusters,
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
    <p><strong>Total Participants:</strong> ', length(unique(ratings$ParticipantID)), '</p>
    <p><strong>Optimal Clusters:</strong> ', clustering_result$optimal_k, '</p>
    <p><strong>Analysis Method:</strong> Rating pattern similarity clustering</p>
    
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
  for (cluster_id in 1:clustering_result$optimal_k) {
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
cat("Starting simplified concept mapping analysis...\n")

# Analyze ratings and create similarity matrix
rating_analysis <- analyze_ratings_and_similarity(ratings, statements)

# Perform clustering analysis
clustering_result <- perform_clustering_analysis(rating_analysis$distance_matrix, statements)

# Create visualizations
visualizations <- create_visualizations(clustering_result, rating_analysis, statements, output_dir)

# Generate reports
reports <- generate_reports(clustering_result, rating_analysis, statements, output_dir)

# Print summary
cat("\n=== Analysis Complete ===\n")
cat("Output files saved to:", output_dir, "\n")
cat("Number of clusters:", clustering_result$optimal_k, "\n")
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